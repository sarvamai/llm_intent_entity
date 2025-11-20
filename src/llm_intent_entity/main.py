from pydantic import BaseModel
from .llm_api import ChatCompletionsAPI
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from .utilities import IndicNormalizer, push_to_sheet, calculate_intent_accuracy, calculate_entity_metrics, calculate_combined_score

logger = logging.getLogger(__name__)

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    PROMPT_TEMPLATE = (PROJECT_ROOT / "prompt_template.txt").read_text()
except FileNotFoundError:
    raise FileNotFoundError(f"prompt_template.txt not found at {PROJECT_ROOT / 'prompt_template.txt'}. Please ensure it exists in the project root.")

class IntentEntityResponse(BaseModel):
    index: int
    intent_score: int
    intent_explanation: str
    entity_score: float
    ground_truth_entities: str
    preserved_entities: str
    missing_entities: str
    entity_explanation: str

def build_prompt(item: Dict[str, Any]) -> str:
    """Build prompt for intent entity evaluation"""
    prompt = PROMPT_TEMPLATE + "\n\n**INPUT:**\n"
    json_object = {
        "index": item["index"],
        "hypothesis": item["hypothesis"],
        "ground_truth": item["ground_truth"],
        "context": item.get("context", "")
    }
    prompt += json.dumps(json_object, indent=2, ensure_ascii=False)
    return prompt

def load_and_validate_dataset(
    dataset_path: str, required_cols: set
) -> pd.DataFrame:
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist")

    if path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path_obj)
    elif path_obj.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path_obj, lines=True)
    else:
        raise ValueError("Unsupported dataset format. Only CSV or JSONL accepted.")

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {', '.join(missing_cols)}")
    
    return df

def normalize_texts_in_dataframe(df: pd.DataFrame, ref_col: str, pred_col: str, lang_col: str) -> pd.DataFrame:
    """Normalize text columns using IndicNormalizer"""
    normalizer = IndicNormalizer()
    df["norm_reference"] = normalizer.normalize_texts(df[ref_col].astype(str).to_list(), df[lang_col].astype(str).to_list())
    df["norm_prediction"] = normalizer.normalize_texts(df[pred_col].astype(str).to_list(), df[lang_col].astype(str).to_list())
    return df

def prepare_evaluation_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Prepare items for LLM evaluation"""
    items = []
    for idx, row in df.iterrows():
        item = {
            "index": idx,
            "hypothesis": row["norm_prediction"],
            "ground_truth": row["norm_reference"],
            "context": row.get("context", "")
        }
        items.append(item)
    return items

def query_llm_for_intent_entity_evaluation(
    evaluation_items: List[Dict[str, Any]],
    dataset_name: str,
    api: ChatCompletionsAPI,
    ignore_cache: bool = False,
) -> Tuple[list, list]:
    """Query LLM for intent entity evaluation"""
    cache_dir = PROJECT_ROOT / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_intent_entity_cache.jsonl"

    if ignore_cache and cache_file.exists():
        cache_file.unlink()

    cached_responses_map = {}
    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    cached_item = json.loads(line)
                    key = cached_item.get("key", {})
                    if key and "index" in key:
                        cache_key = (key["hypothesis"], key["ground_truth"], key.get("context", ""))
                        cached_responses_map[cache_key] = cached_item
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed cache line in {cache_file}: '{line.strip()}'. Error: {e}")

    items_to_query = []
    for item in evaluation_items:
        cache_key = (item["hypothesis"], item["ground_truth"], item.get("context", ""))
        if cache_key not in cached_responses_map:
            items_to_query.append(item)
    
    successful_from_cache = [
        item for cache_key, item in cached_responses_map.items()
        if any(
            (eval_item["hypothesis"], eval_item["ground_truth"], eval_item.get("context", "")) == cache_key
            for eval_item in evaluation_items
        )
    ]

    newly_successful = []
    failed = []

    if items_to_query:
        print(f"Found {len(successful_from_cache)} cached responses. Querying LLM for {len(items_to_query)} new items.")
        for item in items_to_query:
            prompt = build_prompt(item)
            api.append_to_request_queue(prompt=prompt, key=item, schema=IntentEntityResponse)
        
        newly_successful, failed = api.generate_responses_from_queue(output_file_path=cache_file)
    else:
        print(f"All {len(successful_from_cache)} required items found in cache. No new API calls needed.")
    
    all_successful = successful_from_cache + newly_successful
    return all_successful, failed

def process_llm_responses(successful_responses: list, df: pd.DataFrame) -> pd.DataFrame:
    """Process LLM responses and add results to dataframe"""
    # Create a mapping from index to response
    response_map = {}
    for item in successful_responses:
        key = item.get("key", {})
        response = item.get("response", {})
        if key and "index" in key:
            response_map[key["index"]] = response

    # Add response columns to dataframe
    intent_scores = []
    intent_explanations = []
    entity_scores = []
    ground_truth_entities = []
    preserved_entities = []
    missing_entities = []
    entity_explanations = []

    for idx in df.index:
        if idx in response_map:
            resp = response_map[idx]
            intent_scores.append(resp.get("intent_score", -1))
            intent_explanations.append(resp.get("intent_explanation", ""))
            entity_scores.append(resp.get("entity_score", -1.0))
            ground_truth_entities.append(resp.get("ground_truth_entities", ""))
            preserved_entities.append(resp.get("preserved_entities", ""))
            missing_entities.append(resp.get("missing_entities", ""))
            entity_explanations.append(resp.get("entity_explanation", ""))
        else:
            # Fill with error values for missing responses
            intent_scores.append(-1)
            intent_explanations.append("ERROR: No response")
            entity_scores.append(-1.0)
            ground_truth_entities.append("ERROR")
            preserved_entities.append("ERROR")
            missing_entities.append("ERROR")
            entity_explanations.append("ERROR: No response")

    df["intent_score"] = intent_scores
    df["intent_explanation"] = intent_explanations
    df["entity_score"] = entity_scores
    df["ground_truth_entities"] = ground_truth_entities
    df["preserved_entities"] = preserved_entities
    df["missing_entities"] = missing_entities
    df["entity_explanation"] = entity_explanations

    return df

def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate intent and entity metrics"""
    # Filter out error responses
    valid_rows = df[(df["intent_score"] >= 0) & (df["entity_score"] >= 0)]
    
    if len(valid_rows) == 0:
        return {
            "total_samples": len(df),
            "valid_samples": 0,
            "intent_accuracy": 0.0,
            "entity_metrics": {"mean": 0.0, "median": 0.0, "std": 0.0},
            "combined_score": 0.0
        }

    intent_accuracy = calculate_intent_accuracy(valid_rows["intent_score"].tolist())
    entity_metrics = calculate_entity_metrics(valid_rows["entity_score"].tolist())
    combined_score = calculate_combined_score(
        valid_rows["intent_score"].tolist(),
        valid_rows["entity_score"].tolist()
    )

    return {
        "total_samples": len(df),
        "valid_samples": len(valid_rows),
        "intent_accuracy": intent_accuracy,
        "entity_metrics": entity_metrics,
        "combined_score": combined_score
    }

def save_outputs(df: pd.DataFrame, logs: List[Dict], failed: List[Dict], outputs_dir: Path, sheet_name: str, worksheet_prefix: str, creds_path: Path):
    """Save outputs to CSV and Google Sheets"""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(outputs_dir / "llm_intent_entity.csv", index=False)
    print(f"Output saved to {outputs_dir / 'llm_intent_entity.csv'}")
    push_to_sheet(df, sheet_name, f"{worksheet_prefix} - full output", creds_path)

    if logs:
        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(outputs_dir / "llm_logs.csv", index=False)
        print(f"LLM logs saved to {outputs_dir / 'llm_logs.csv'}")
        push_to_sheet(logs_df, sheet_name, f"{worksheet_prefix} - llm logs", creds_path)

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(outputs_dir / "llm_failed_requests.csv", index=False)
        print(f"Failed LLM requests saved to {outputs_dir / 'llm_failed_requests.csv'}")

def process_dataset_for_intent_entity_evaluation(
    dataset_path: str, 
    reference_col_name: str, 
    predicted_col_name: str, 
    audio_filepath_col_name: str, 
    creds_path: str,
    language_col_name: str = "language",
    context_col_name: str = "context",
    output_sheet_name: str = "llm-intent-entity-analysis",
    output_worksheet_name: str = "llm-intent-entity-output",
    ignore_cache: bool = False,
    gemini_location: str = "us-central1",
):
    """Main function to process dataset for intent entity evaluation"""
    creds_path_obj = Path(creds_path)
    if not creds_path_obj.exists():
        raise FileNotFoundError(f"Credentials file not found at {creds_path}")

    api = ChatCompletionsAPI(
        model_name="google/gemini-2.5-flash",
        api_key="",
        base_url="",
        gemini=True,
        max_retries=0,
        timeout=None,
        creds_path=creds_path,
        location=gemini_location,
    )
    
    required_cols = {reference_col_name, predicted_col_name, audio_filepath_col_name, language_col_name}
    df = load_and_validate_dataset(dataset_path, required_cols)
    
    # Add context column if it doesn't exist
    if context_col_name not in df.columns:
        df[context_col_name] = ""
    
    df = normalize_texts_in_dataframe(df, reference_col_name, predicted_col_name, language_col_name)
    evaluation_items = prepare_evaluation_items(df)
    
    dataset_name = Path(dataset_path).stem
    successful, failed = query_llm_for_intent_entity_evaluation(
        evaluation_items, dataset_name, api, ignore_cache=ignore_cache
    )
    
    df = process_llm_responses(successful, df)
    metrics = calculate_metrics(df)
    
    print("\n=== INTENT ENTITY EVALUATION RESULTS ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"Entity score (mean): {metrics['entity_metrics']['mean']:.4f}")
    print(f"Entity score (median): {metrics['entity_metrics']['median']:.4f}")
    print(f"Combined score: {metrics['combined_score']:.4f}")
    
    outputs_dir = PROJECT_ROOT / "outputs" / Path(dataset_path).stem
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log records for failed/successful responses
    log_records = []
    for item in successful:
        key = item.get("key", {})
        response = item.get("response", {})
        log_records.append({
            "index": key.get("index", -1),
            "hypothesis": key.get("hypothesis", ""),
            "ground_truth": key.get("ground_truth", ""),
            "context": key.get("context", ""),
            "intent_score": response.get("intent_score", -1),
            "intent_explanation": response.get("intent_explanation", ""),
            "entity_score": response.get("entity_score", -1.0),
            "entity_explanation": response.get("entity_explanation", "")
        })
    
    save_outputs(df, log_records, failed, outputs_dir, output_sheet_name, output_worksheet_name, creds_path_obj)

if __name__ == "__main__":
    process_dataset_for_intent_entity_evaluation(
        dataset_path = "/path/to/dataset_with_predictions.csv", 
        reference_col_name="transcription", 
        predicted_col_name="prediction", 
        audio_filepath_col_name="audio_filepath",
        creds_path="/path/to/creds.json",
        language_col_name="language",
        context_col_name="context",
        output_sheet_name="LLM Intent Entity Analysis",
        output_worksheet_name="<name of worksheet>",
        ignore_cache=True,
    )
