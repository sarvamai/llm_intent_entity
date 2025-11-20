# imports
from openai import OpenAI
from typing import Optional
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request
import queue
import threading
import uuid
from pydantic import BaseModel
from pydantic_core import from_json
import traceback
import re
from tqdm import tqdm
from joblib import Parallel, delayed
import jsonlines
from collections import Counter
import time
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# helper functions
def write_to_file(buffer, output_file_path, file_lock, data=None, flush=False, chunk_size=1000):
    if data:
        buffer.append(data)
    if (flush and buffer) or len(buffer) >= chunk_size:
        write_buffer = list(buffer)
        buffer.clear()
        if file_lock:
            with file_lock:
                with jsonlines.open(output_file_path, mode='a') as writer:
                    writer.write_all(write_buffer)
        else:
            with jsonlines.open(output_file_path, mode='a') as writer:
                writer.write_all(write_buffer)
                
def json_string_to_python_dict(content: str) -> dict | None:
    try:
        cleaned_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        json_content_str = None
        for pattern in [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'(?:json|JSON|output|Output):\s*(\{.*?\})',
            r'\{.*\}'
        ]:
            match = re.search(pattern, cleaned_content, re.DOTALL)
            if match:
                json_content_str = match.group(1) if match.groups() else match.group(0)
                break
        
        if json_content_str:
            return from_json(json_content_str, allow_partial=True)
        return None
    except Exception as e:
        return None
    
def flatten_responses(response_list):
    records = []
    for response in response_list:
        record = {}
        if isinstance(response['key'], dict):
            record.update(response['key'])
        elif isinstance(response['key'], str):
            key_dict = json_string_to_python_dict(response['key'])
            record.update(key_dict) if key_dict else record.update({'key': response['key']})
        elif isinstance(response['key'], (list, tuple)):
            record.update({f"key_{i}": val for i, val in enumerate(response['key'])})
        else: record['key'] = response['key']
        
        if isinstance(response['response'], dict):
            record.update(response['response'])
        elif isinstance(response['response'], str):
            response_dict = json_string_to_python_dict(response['response'])
            record.update(response_dict) if response_dict else record.update({'response': response['response']})
        else: record['response'] = response['response']
        records.append(record)
    return records

def validate_response_with_schema(response_content, key, schema):
    try:
        if isinstance(response_content, str):
            response_to_validate = json_string_to_python_dict(response_content)
        else:
            response_to_validate = response_content
        if response_to_validate is None:
                raise ValueError("Could not parse JSON from response string.")
        if not isinstance(response_to_validate, dict):
            raise ValueError(f"Expected dict, got {type(response_to_validate).__name__}")
        validated_response = schema.model_validate(response_to_validate)
        return {"key": key, "response": validated_response.model_dump(), "status": "success"}
    except Exception as e:
        return {"key": key, "response": response_content, "status": "parse_error", "error": f"Schema validation failed: {str(e)}", "traceback": traceback.format_exc()}

class ChatCompletionsAPI:
    def __init__(
        self, 
        model_name: str,
        api_key: str, 
        base_url: str, 
        num_workers: int = 500, 
        max_retries: int = 2, 
        system_prompt: Optional[str] = None, 
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        gemini: bool = False,
        creds_path: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        report_usage: bool = False,
        delay: float = 0.0,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.delay = delay
        
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.report_usage = report_usage
        self.max_tokens = max_tokens
        
        self.num_workers = num_workers
        self.max_retries = max_retries
        self.request_queue = queue.Queue()
        self.semaphore = threading.Semaphore(self.num_workers)

        if not gemini:
            self.client = OpenAI(
                api_key=api_key, 
                base_url=base_url, 
                timeout=timeout
            )
        else:
            if not creds_path or not location:
                raise ValueError("`creds_path` and `location` must be provided for Gemini models.")
            credentials = Credentials.from_service_account_file(
                creds_path, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            if not project_id:
                project_id = credentials.project_id
            if not project_id:
                raise ValueError("For Gemini models, project_id is required. It can be passed directly or be present in the credentials file.")
            credentials.refresh(Request())
            self.client = OpenAI(
                api_key=credentials.token, 
                base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
                timeout=timeout
            )
        self.is_parse_endpoint_supported()
        print(f"API Initialized - Model: {self.model_name}, Workers: {self.num_workers}, "
              f"Temperature: {self.temperature}, Timeout: {self.timeout}, Delay: {self.delay}, "
              f"Parse Endpoint: {self.parse_endpoint_supported}")

    def is_parse_endpoint_supported(self):
        class ResponseModel(BaseModel):
            greeting: str
            response: str
            mood: str
        try:
            self.client.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always responds in JSON format."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                response_format=ResponseModel
            )
            print("Parse endpoint supported!")
            self.parse_endpoint_supported = True
        except Exception as e:
            print(f"Error checking parse endpoint support: {e}")
            print(f"Falling back to create endpoint!")
            self.parse_endpoint_supported = False

    def generate_single_response(self, prompt, key=None, schema=None, temperature=None, system_prompt=None):
       if key is None: key = {'uuid': str(uuid.uuid4())}
       with self.semaphore:
           try:
                messages = []
                system_content = system_prompt or self.system_prompt
                if system_content:
                    messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": prompt})                
                params = {
                    "model": self.model_name,
                    "messages": messages,
                }
                if temperature or self.temperature:
                    params["temperature"] = temperature if temperature is not None else self.temperature
                if self.max_tokens:
                    params["max_tokens"] = self.max_tokens
                if self.timeout:
                    params["timeout"] = self.timeout

                if isinstance(self.base_url, str) and 'aiplatform.googleapis.com' in self.base_url:
                    params.setdefault("extra_body", {})
                    params["extra_body"]["safety_settings"] = [
                        {"category": cat.name, "threshold": HarmBlockThreshold.BLOCK_NONE.name} for cat in [
                            HarmCategory.HARM_CATEGORY_HARASSMENT,
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
                        ]
                    ]

                response_content = None
                if schema is not None:
                    if self.parse_endpoint_supported:
                        params["response_format"] = schema
                        response = self.client.beta.chat.completions.parse(**params)
                        response_content = response.choices[0].message.parsed
                        usage = response.usage
                        result = {"key": key, "response": response_content.model_dump(), "status": "success"} # type: ignore
                        if self.report_usage and usage:
                            result["input_tokens"] = usage.prompt_tokens
                            result["output_tokens"] = usage.completion_tokens
                        return result
                    else:
                        params["extra_body"] = params.get("extra_body", {})
                        params["extra_body"]["guided_json"] = schema.model_json_schema()
                        response = self.client.chat.completions.create(**params)
                        response_content = response.choices[0].message.content
                        usage = response.usage
                        result = validate_response_with_schema(response_content, key, schema)
                        if self.report_usage and usage:
                            result['input_tokens'] = getattr(usage, "prompt_tokens", 0)
                            result['output_tokens'] = getattr(usage, "completion_tokens", 0)
                        return result
                else:
                    response = self.client.chat.completions.create(**params)
                    response_content = response.choices[0].message.content
                    usage = response.usage
                    result = {"key": key, "response": response_content, "status": "success_no_schema"}
                    if self.report_usage and usage:
                        result["input_tokens"] = usage.prompt_tokens
                        result["output_tokens"] = usage.completion_tokens
                    return result
           except Exception as e:
                return {"key": key, "response": None, "status": "api_error", "error": str(e), "traceback": traceback.format_exc()}
    
    def append_to_request_queue(self, prompt, key=None, schema=None, temperature=None, system_prompt=None, audio_path=None):
        if audio_path:
            raise NotImplementedError("audio_path is not supported for ChatCompletionsAPI")
        item = {
            'prompt': prompt,
            'key': key,
            'schema': schema,
            'temperature': temperature,
            'system_prompt': system_prompt  
        }
        self.request_queue.put(item)
    
    def generate_responses_from_queue(self, output_file_path=None, retry_temp: float = 0.5, input_file_path=None, delete_input_file=False):
        if input_file_path or delete_input_file:
            raise ValueError("input_file_path and delete_input_file are not supported for ChatCompletionsAPI")
        print(f"Generating responses from queue with {self.num_workers} workers, {self.max_retries} retries, and {self.timeout} timeout")
        requests_to_process = []
        while not self.request_queue.empty():
            requests_to_process.append(self.request_queue.get())
        successful_responses, failed_responses = [], []
        file_lock = threading.Lock() if output_file_path and self.num_workers > 1 else None
        buffer = []
        try:
            for i in range(self.max_retries + 1):
                if not requests_to_process:
                    break
                if i > 0:
                    backoff_time = (self.timeout if self.timeout else 10) * (2 ** (i - 1))
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                print(f"Attempt {i + 1}/{self.max_retries + 1}: {len(requests_to_process)} requests")
                results = []
                with tqdm(total=len(requests_to_process), desc=f"Attempt {i+1}") as pbar:
                    def process_request(req):
                        result = self.generate_single_response(**req)
                        pbar.update(1)
                        return result, req

                    if self.num_workers > 1 and self.delay > 0:
                        chunk_size = self.num_workers
                        request_chunks = [requests_to_process[j:j + chunk_size] for j in range(0, len(requests_to_process), chunk_size)]
                        
                        for idx, chunk in enumerate(request_chunks):
                            with Parallel(n_jobs=len(chunk), prefer="threads") as parallel:
                                chunk_results = parallel(delayed(process_request)(req) for req in chunk)
                            results.extend(chunk_results)
                            if self.delay > 0 and idx < len(request_chunks) - 1:
                                time.sleep(self.delay)
                    else:
                        if self.num_workers > 1:
                            with Parallel(n_jobs=min(self.num_workers, len(requests_to_process)), prefer="threads") as parallel:
                                results = parallel(delayed(process_request)(req) for req in requests_to_process)
                        else:
                            results = [process_request(req) for req in requests_to_process]

                requests_to_process_next = []
                status_counts = Counter()
                if results:
                    is_last_attempt = (i == self.max_retries)
                    for result, req in results: # type: ignore
                        status = result['status']
                        status_counts.update([status])
                        should_retry = status in ['api_error', 'parse_error'] and not is_last_attempt
                        if should_retry:
                            if status == 'parse_error':
                                req['temperature'] = retry_temp
                            requests_to_process_next.append(req)
                        else:
                            if status in ['success', 'success_no_schema']:
                                successful_responses.append(result)
                            else:
                                failed_responses.append(result)
                            if output_file_path:
                                write_to_file(buffer, output_file_path, file_lock, data=result)
                print(f"Completed with {status_counts['success']+status_counts['success_no_schema']} successful responses, {status_counts['parse_error']} parsing errors, and {status_counts['api_error']} API errors.")
                requests_to_process = requests_to_process_next
        finally:
            if output_file_path:
                write_to_file(buffer, output_file_path, file_lock, flush=True)

        if self.report_usage:
            total_prompt_tokens = 0
            total_completion_tokens = 0
            for r in successful_responses:
                total_prompt_tokens += r.get("input_tokens", 0)
                total_completion_tokens += r.get("output_tokens", 0)
            
            num_successful = len(successful_responses)
            avg_input_tokens = total_prompt_tokens / num_successful if num_successful > 0 else 0
            avg_output_tokens = total_completion_tokens / num_successful if num_successful > 0 else 0

            print(f"Avg Input Tokens: {avg_input_tokens:.2f}, Avg Output Tokens: {avg_output_tokens:.2f}")
        
        return successful_responses, failed_responses