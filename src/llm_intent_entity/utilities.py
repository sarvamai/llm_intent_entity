from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from transformers import WhisperProcessor
from joblib import Parallel, delayed
import re
import string
from pathlib import Path
import json
import gspread
import numpy as np

lang_to_code = {
    'hindi': 'hi',
    'bengali': 'bn',
    'tamil': 'ta',
    'telugu': 'te',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'odia': 'or',
    'oria': 'or',
    'assamese': 'or',
    'punjabi': 'pa',
    'english': 'en'
}

indic_langs = {'hi', 'bn', 'ta', 'te', 'gu', 'kn', 'ml', 'mr', 'or', 'pa'}

class IndicNormalizer:
    def __init__(self):
        self.indic_factory = IndicNormalizerFactory()
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_tokenizer = self.whisper_processor.tokenizer # type: ignore
        
    def normalize_text(self, text: str, lang_code: str) -> str:
        lang_code = lang_to_code.get(lang_code, lang_code)
        if pd.isna(text) or not isinstance(text, str):
            return text
        if not text: return text
        base_lang = lang_code.split('-')[0].lower()
        text = re.sub(r'([,\-\.\(\)\[\]\{\}/\\])\B', r' ', text)
        INDIC_PUNCTUATION = "।॥॰''\"‛‟′″´˝^°¤।॥॰¯'—–‑°¬´ۭ۪\u200b\u200c\u200d\u200e\u200f"
        text = text.translate(str.maketrans('', '', string.punctuation + INDIC_PUNCTUATION)).lower()
        
        if base_lang in indic_langs and base_lang != 'ur': # urdu has special handling
            normalizer = self.indic_factory.get_normalizer(base_lang)
            text = normalizer.normalize(text)
        else:
            text = self.whisper_tokenizer.normalize(text)
        text = re.sub(' +', ' ', text).strip()
        return text
    
    def _normalize_batch(self, text_batch: List[str], lang_batch: List[str]) -> List[str]:
        return [self.normalize_text(text, lang) for text, lang in zip(text_batch, lang_batch)]

    def normalize_texts(self, text_list: List[str], lang_list: List[str], n_jobs: int = -1, batch_size: int = 500) -> List[str]:
        if len(text_list) != len(lang_list):
            raise ValueError("text_list and lang_list must have the same length")

        if not text_list:
            return []

        batches: List[Tuple[List[str], List[str]]] = []
        for i in range(0, len(text_list), batch_size):
            batches.append((text_list[i:i+batch_size], lang_list[i:i+batch_size]))

        processed_batches = Parallel(n_jobs=n_jobs)(
            delayed(self._normalize_batch)(text_batch, lang_batch) for text_batch, lang_batch in tqdm(
                batches, desc="Normalizing text batches"
        ))
        
        if processed_batches:
            return [item for sublist in processed_batches for item in sublist] # type: ignore
        return []

class IndicASRPostProcessor:
    NO_NORMALIZE_LANGS = {'ur', 'kok', 'mai', 'doi', 'sat', 'mni', 'brx', 'ks'}

    def __init__(self):
        self.factory = IndicNormalizerFactory()
        self.translator = self._create_translator()

    def _create_translator(self):
        translator = {
            '॥': ' ', '۔': ' ', '।': ' ', ''': '', '–': ' ', ''': ' ', 'ʼ': '', '°': ' ',
            '¬': ' ', 'ۭ': ' ', '۪': ' ', '‑': ' ', '—': ' ', '\u200b': '', '\u200c': '',
            '\u200d': '', '´': '', ',': '', '\u200e': '', '\u200f': '', '"': '', '"': ''
        }
        translator.update({x: " " for x in (set(string.punctuation) - {',', '<', '>', '|'})})
        return str.maketrans(translator)

    def normalize_text(self, text: str, lang: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return text
        if not text:
            return text
        
        lang_code = lang_to_code.get(lang.lower(), lang.lower())
        
        text = text.translate(self.translator)
        
        if lang_code in indic_langs and lang_code not in self.NO_NORMALIZE_LANGS:
            normalizer = self.factory.get_normalizer(lang_code)
            text = normalizer.normalize(text)
            
        text = re.sub(' +', ' ', text).strip()
        text = re.sub('\t+', ' ', text).strip()
        return text

    def _normalize_batch(self, text_batch: List[str], lang_batch: List[str]) -> List[str]:
        return [self.normalize_text(text, lang) for text, lang in zip(text_batch, lang_batch)]

    def normalize_texts(self, text_list: List[str], lang_list: List[str], n_jobs: int = -1, batch_size: int = 500) -> List[str]:
        if len(text_list) != len(lang_list):
            raise ValueError("text_list and lang_list must have the same length")

        if not text_list:
            return []

        batches: List[Tuple[List[str], List[str]]] = []
        for i in range(0, len(text_list), batch_size):
            batches.append((text_list[i:i+batch_size], lang_list[i:i+batch_size]))

        processed_batches = Parallel(n_jobs=n_jobs)(
            delayed(self._normalize_batch)(text_batch, lang_batch) for text_batch, lang_batch in tqdm(
                batches, desc="Normalizing text batches"
        ))
        
        if processed_batches:
            return [item for sublist in processed_batches for item in sublist] # type: ignore
        return []

# METRICS FOR INTENT AND ENTITY EVALUATION

def calculate_intent_accuracy(intent_scores: List[int]) -> float:
    """Calculate accuracy for intent scores (0 or 1)"""
    if not intent_scores:
        return 0.0
    return sum(intent_scores) / len(intent_scores)

def calculate_entity_metrics(entity_scores: List[float]) -> dict:
    """Calculate metrics for entity scores (0 to 1)"""
    if not entity_scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    
    entity_scores_array = np.array(entity_scores)
    return {
        "mean": float(np.mean(entity_scores_array)),
        "median": float(np.median(entity_scores_array)),
        "std": float(np.std(entity_scores_array))
    }


# SHEETS

def _col_idx_to_excel(idx: int) -> str:
    letters = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters

def push_to_sheet(df, sheet_name, subsheet_name, creds_path, overwrite=False, chunk_size=5000):
    try:
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found at {creds_path}.")
        client = gspread.service_account(filename=str(creds_path))
        try:
            sheet = client.open(sheet_name)
        except gspread.exceptions.SpreadsheetNotFound:
            sheet = client.create(sheet_name)
            # To share with your org, uncomment and replace with your domain:
            # sheet.share("yourdomain.com", perm_type="domain", role="reader")
        final_name = subsheet_name
        try:
            existing_worksheet = sheet.worksheet(subsheet_name)
            if overwrite:
                print(f"Subsheet '{subsheet_name}' already exists. Overwriting.")
                sheet.del_worksheet(existing_worksheet)
            else:
                counter = 1
                while True:
                    final_name = f"{subsheet_name}_{counter}"
                    try:
                        sheet.worksheet(final_name)
                        counter += 1
                    except gspread.exceptions.WorksheetNotFound:
                        break
                print(f"Subsheet '{subsheet_name}' already exists. Creating new subsheet '{final_name}'.")
        except gspread.exceptions.WorksheetNotFound:
            pass
        print(f"Writing to subsheet '{final_name}' in spreadsheet '{sheet_name}'...")
        worksheet = sheet.add_worksheet(title=final_name, rows=1, cols=df.shape[1])
        df = df.replace([np.inf, -np.inf], np.nan)
        df_filled = df.fillna("").applymap(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, tuple, dict)) else x)
        
        print(f"DataFrame has {df.shape[1]} columns and {df.shape[0]} rows. Pushing in chunks of {chunk_size}.")
        data = [df_filled.columns.values.tolist()] + df_filled.values.tolist()
        
        start_row = 1
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            end_row = start_row + len(chunk) - 1
            
            worksheet.resize(rows=end_row) 
            
            last_col_letter = _col_idx_to_excel(df.shape[1])
            range_to_update = f"A{start_row}:{last_col_letter}{end_row}"
            worksheet.update(range_to_update, chunk)
            
            start_row = end_row + 1
            print(f"Uploaded rows {i} to {i+len(chunk)-1}")

        print(f"Successfully pushed data. Sheet URL: {sheet.url}")
    except Exception as e:
        print(f"Error pushing data to sheet: {e}")


def load_from_sheet(subsheet_name, creds_path, sheet_name=None, sheet_id=None):
    if sheet_name is None and sheet_id is None:
        print("Error: You must provide either sheet_name or sheet_id.")
        return None
    try:
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found at {creds_path}.")
        client = gspread.service_account(filename=str(creds_path))

        sheet = None
        if sheet_id:
            try:
                sheet = client.open_by_key(sheet_id)
            except gspread.exceptions.SpreadsheetNotFound:
                print(f"Error: Spreadsheet with ID '{sheet_id}' not found.")
                return None
        elif sheet_name:
            spreadsheets = client.list_spreadsheet_files()
            matching_sheets = [s for s in spreadsheets if s['name'] == sheet_name]

            if not matching_sheets:
                print(f"Error: No spreadsheet found with name '{sheet_name}'.")
                return None
            
            if len(matching_sheets) > 1:
                print(f"Multiple spreadsheets found with the name '{sheet_name}'.")
                print("Please use --sheet-id with one of the following IDs:")
                for s in matching_sheets:
                    print(f"  - Name: {s['name']}, ID: {s['id']}")
                return None
            
            sheet = client.open_by_key(matching_sheets[0]['id'])
        
        worksheet = sheet.worksheet(subsheet_name) # type: ignore
        
        print(f"Loading data from subsheet '{subsheet_name}' in spreadsheet '{sheet.title}'...") # type: ignore
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
        return df

    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Subsheet '{subsheet_name}' not found in spreadsheet '{sheet.title}'.") # type: ignore
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def list_sheets(creds_path):
    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials file not found at {creds_path}.")
    client = gspread.service_account(filename=str(creds_path))
    print("Fetching list of spreadsheets...")
    spreadsheets = client.list_spreadsheet_files()
    for sheet in spreadsheets:
        url = f"https://docs.google.com/spreadsheets/d/{sheet['id']}"
        print(f"  - Name: {sheet['name']}, URL: {url}")


def delete_spreadsheet(sheet_id, creds_path):
    if not creds_path.exists():
        raise FileNotFoundError(f"Credentials file not found at {creds_path}.")
    client = gspread.service_account(filename=str(creds_path))
    try:
        spreadsheet = client.open_by_key(sheet_id)
        sheet_name = spreadsheet.title
        num_subsheets = len(spreadsheet.worksheets())
        confirm = input(f"Confirm deletion of spreadsheet: '{sheet_name}' with all {num_subsheets} subsheets? (y/n) ")
        if confirm.lower() == 'y':
            client.del_spreadsheet(spreadsheet.id)
            print(f"Spreadsheet '{sheet_name}' deleted successfully.")
        else:
            print("Deletion aborted.")
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Spreadsheet with ID '{sheet_id}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
