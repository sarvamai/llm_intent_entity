from pydantic import BaseModel, Field, ValidationError
from .llm_api import ChatCompletionsAPI
import logging
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
import pandas as pd
from .utilities import IndicNormalizer, push_to_sheet, calculate_intent_accuracy, calculate_entity_metrics
from .typed_entities import (
    MatchPolicy,
    TypedEntityCandidate,
    TypedEntityScore,
    entity_aware_text_signature,
    score_typed_entities,
    typed_score_explanation,
)

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
    typed_entities_complete: bool = False
    ground_truth_typed_entities: List[TypedEntityCandidate] = Field(default_factory=list)
    hypothesis_typed_entities: List[TypedEntityCandidate] = Field(default_factory=list)

def build_prompt(item: Dict[str, Any]) -> str:
    """Build prompt for intent entity evaluation"""
    prompt = PROMPT_TEMPLATE + "\n\n**INPUT:**\n"
    json_object = {
        "index": item["index"],
        "hypothesis": item["hypothesis"],
        "ground_truth": item["ground_truth"],
        "context": item.get("context", ""),
        "language": item.get("language", ""),
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

def prepare_evaluation_items(
    df: pd.DataFrame,
    language_col: str = "language",
    context_col: str = "context",
) -> List[Dict[str, Any]]:
    """Prepare items for LLM evaluation"""
    items = []
    for idx, row in df.iterrows():
        item = {
            "index": idx,
            "hypothesis": row["norm_prediction"],
            "ground_truth": row["norm_reference"],
            "context": row.get(context_col, ""),
            "language": row.get(language_col, ""),
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
                        cache_key = (
                            key["hypothesis"],
                            key["ground_truth"],
                            key.get("context", ""),
                            key.get("language", ""),
                        )
                        cached_responses_map[cache_key] = cached_item
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed cache line in {cache_file}: '{line.strip()}'. Error: {e}")

    items_to_query = []
    successful_from_cache = []
    for item in evaluation_items:
        cache_key = (
            item["hypothesis"],
            item["ground_truth"],
            item.get("context", ""),
            item.get("language", ""),
        )
        cached_item = cached_responses_map.get(cache_key)
        if cached_item is None:
            # Cache entries created before typed scoring did not include language.
            legacy_cache_key = (*cache_key[:3], "")
            cached_item = cached_responses_map.get(legacy_cache_key)
        if cached_item is None:
            items_to_query.append(item)
        else:
            successful_from_cache.append(cached_item)

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

def process_llm_responses(
    successful_responses: list,
    df: pd.DataFrame,
    match_policy: MatchPolicy | None = None,
) -> pd.DataFrame:
    """Process LLM responses and add results to dataframe"""
    # Create a mapping from index to response
    response_map = {}
    for item in successful_responses:
        key = item.get("key", {})
        response = item.get("response", {})
        if key and "index" in key:
            response_map[key["index"]] = (response, key)

    # Add response columns to dataframe
    intent_scores = []
    intent_explanations = []
    entity_scores = []
    ground_truth_entities = []
    preserved_entities = []
    missing_entities = []
    entity_explanations = []
    typed_ground_truth_entities = []
    typed_hypothesis_entities = []
    entity_precisions = []
    entity_recalls = []
    entity_f1_scores = []
    named_entity_error_rates = []
    entity_review_required = []
    entity_review_reasons = []
    entity_scoring_methods = []

    for idx in df.index:
        if idx in response_map:
            resp, key = response_map[idx]
            intent_score = resp.get("intent_score", -1)
            intent_explanation = resp.get("intent_explanation", "")
            typed_complete = resp.get("typed_entities_complete", False)
            reference_candidates = resp.get("ground_truth_typed_entities", [])
            hypothesis_candidates = resp.get("hypothesis_typed_entities", [])

            if typed_complete:
                try:
                    typed_score = score_typed_entities(
                        reference_candidates,
                        hypothesis_candidates,
                        language=key.get("language", ""),
                        policy=match_policy,
                    )
                    canonical_references = [match.reference for match in typed_score.matches] + typed_score.missing
                    canonical_hypotheses = [match.hypothesis for match in typed_score.matches] + typed_score.spurious
                except (TypeError, ValueError, ValidationError) as exc:
                    typed_score = TypedEntityScore(
                        score=None,
                        precision=None,
                        recall=None,
                        f1=None,
                        named_entity_error_rate=None,
                        review_required=True,
                        review_reasons=[f"Invalid typed entity payload: {exc}"],
                    )
                    canonical_references = []
                    canonical_hypotheses = []
                reference_payload = [entity.typed_payload() for entity in canonical_references]
                hypothesis_payload = [entity.typed_payload() for entity in canonical_hypotheses]

                if (
                    typed_score.score == 1.0
                    and entity_aware_text_signature(key.get("ground_truth", ""), canonical_references)
                    == entity_aware_text_signature(key.get("hypothesis", ""), canonical_hypotheses)
                ):
                    intent_score = 1
                    deterministic_note = "Deterministic typed entities preserve the utterance after canonicalization."
                    intent_explanation = f"{intent_explanation} {deterministic_note}".strip()

                entity_score = typed_score.score if typed_score.score is not None else -1.0
                gt_entities = ", ".join(
                    f"{entity.entity_type.value}({entity.raw} -> {entity.display_value})"
                    for entity in canonical_references
                ) or "NA"
                preserved = ", ".join(
                    f"{match.entity_type.value}({match.reference.display_value})"
                    for match in typed_score.matches
                ) or "NA"
                missing = ", ".join(
                    f"{entity.entity_type.value}({entity.display_value})" for entity in typed_score.missing
                ) or "NA"
                entity_explanation = typed_score_explanation(typed_score)
                entity_precisions.append(typed_score.precision)
                entity_recalls.append(typed_score.recall)
                entity_f1_scores.append(typed_score.f1)
                named_entity_error_rates.append(typed_score.named_entity_error_rate)
                entity_review_required.append(typed_score.review_required)
                entity_review_reasons.append("; ".join(typed_score.review_reasons))
                entity_scoring_methods.append("typed_deterministic")
            else:
                entity_score = resp.get("entity_score", -1.0)
                gt_entities = resp.get("ground_truth_entities", "")
                preserved = resp.get("preserved_entities", "")
                missing = resp.get("missing_entities", "")
                entity_explanation = resp.get("entity_explanation", "")
                entity_precisions.append(None)
                entity_recalls.append(None)
                entity_f1_scores.append(None)
                named_entity_error_rates.append(None)
                entity_review_required.append(False)
                entity_review_reasons.append("")
                entity_scoring_methods.append("legacy_llm")
                reference_payload = reference_candidates
                hypothesis_payload = hypothesis_candidates

            intent_scores.append(intent_score)
            intent_explanations.append(intent_explanation)
            entity_scores.append(entity_score)
            ground_truth_entities.append(gt_entities)
            preserved_entities.append(preserved)
            missing_entities.append(missing)
            entity_explanations.append(entity_explanation)
            typed_ground_truth_entities.append(json.dumps(reference_payload, ensure_ascii=False))
            typed_hypothesis_entities.append(json.dumps(hypothesis_payload, ensure_ascii=False))
        else:
            # Fill with error values for missing responses
            intent_scores.append(-1)
            intent_explanations.append("ERROR: No response")
            entity_scores.append(-1.0)
            ground_truth_entities.append("ERROR")
            preserved_entities.append("ERROR")
            missing_entities.append("ERROR")
            entity_explanations.append("ERROR: No response")
            typed_ground_truth_entities.append("[]")
            typed_hypothesis_entities.append("[]")
            entity_precisions.append(None)
            entity_recalls.append(None)
            entity_f1_scores.append(None)
            named_entity_error_rates.append(None)
            entity_review_required.append(True)
            entity_review_reasons.append("No response")
            entity_scoring_methods.append("error")

    df["intent_score"] = intent_scores
    df["intent_explanation"] = intent_explanations
    df["entity_score"] = entity_scores
    df["ground_truth_entities"] = ground_truth_entities
    df["preserved_entities"] = preserved_entities
    df["missing_entities"] = missing_entities
    df["entity_explanation"] = entity_explanations
    df["ground_truth_typed_entities"] = typed_ground_truth_entities
    df["hypothesis_typed_entities"] = typed_hypothesis_entities
    df["entity_precision"] = entity_precisions
    df["entity_recall"] = entity_recalls
    df["entity_f1"] = entity_f1_scores
    df["named_entity_error_rate"] = named_entity_error_rates
    df["entity_review_required"] = entity_review_required
    df["entity_review_reasons"] = entity_review_reasons
    df["entity_scoring_method"] = entity_scoring_methods

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
            "typed_entity_metrics": {
                "samples": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "named_entity_error_rate": 0.0,
            },
        }

    intent_accuracy = calculate_intent_accuracy(valid_rows["intent_score"].tolist())
    entity_metrics = calculate_entity_metrics(valid_rows["entity_score"].tolist())
    typed_rows = valid_rows[valid_rows["entity_scoring_method"] == "typed_deterministic"]
    typed_entity_metrics = {
        "samples": len(typed_rows),
        "precision": float(typed_rows["entity_precision"].mean()) if len(typed_rows) else 0.0,
        "recall": float(typed_rows["entity_recall"].mean()) if len(typed_rows) else 0.0,
        "f1": float(typed_rows["entity_f1"].mean()) if len(typed_rows) else 0.0,
        "named_entity_error_rate": float(typed_rows["named_entity_error_rate"].mean()) if len(typed_rows) else 0.0,
    }

    return {
        "total_samples": len(df),
        "valid_samples": len(valid_rows),
        "intent_accuracy": intent_accuracy,
        "entity_metrics": entity_metrics,
        "typed_entity_metrics": typed_entity_metrics,
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
    entity_match_policy: MatchPolicy | None = None,
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
    evaluation_items = prepare_evaluation_items(df, language_col_name, context_col_name)
    
    dataset_name = Path(dataset_path).stem
    successful, failed = query_llm_for_intent_entity_evaluation(
        evaluation_items, dataset_name, api, ignore_cache=ignore_cache
    )
    
    df = process_llm_responses(successful, df, entity_match_policy)
    metrics = calculate_metrics(df)
    
    print("\n=== INTENT ENTITY EVALUATION RESULTS ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Intent accuracy: {metrics['intent_accuracy']:.4f}")
    print(f"Entity score (mean): {metrics['entity_metrics']['mean']:.4f}")
    print(f"Entity score (median): {metrics['entity_metrics']['median']:.4f}")
    if metrics["typed_entity_metrics"]["samples"]:
        print(f"Typed entity precision: {metrics['typed_entity_metrics']['precision']:.4f}")
        print(f"Typed entity recall: {metrics['typed_entity_metrics']['recall']:.4f}")
        print(f"Typed entity F1: {metrics['typed_entity_metrics']['f1']:.4f}")
        print(f"Named Entity Error Rate: {metrics['typed_entity_metrics']['named_entity_error_rate']:.4f}")
    
    outputs_dir = PROJECT_ROOT / "outputs" / Path(dataset_path).stem
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log records for failed/successful responses
    log_records = []
    for item in successful:
        key = item.get("key", {})
        response = item.get("response", {})
        processed_row = df.loc[key["index"]] if key.get("index") in df.index else None
        log_records.append({
            "index": key.get("index", -1),
            "hypothesis": key.get("hypothesis", ""),
            "ground_truth": key.get("ground_truth", ""),
            "context": key.get("context", ""),
            "intent_score": processed_row["intent_score"] if processed_row is not None else response.get("intent_score", -1),
            "intent_explanation": processed_row["intent_explanation"] if processed_row is not None else response.get("intent_explanation", ""),
            "entity_score": processed_row["entity_score"] if processed_row is not None else response.get("entity_score", -1.0),
            "entity_explanation": processed_row["entity_explanation"] if processed_row is not None else response.get("entity_explanation", ""),
            "entity_precision": processed_row["entity_precision"] if processed_row is not None else None,
            "entity_recall": processed_row["entity_recall"] if processed_row is not None else None,
            "entity_f1": processed_row["entity_f1"] if processed_row is not None else None,
            "named_entity_error_rate": processed_row["named_entity_error_rate"] if processed_row is not None else None,
            "entity_review_required": processed_row["entity_review_required"] if processed_row is not None else False,
            "entity_scoring_method": processed_row["entity_scoring_method"] if processed_row is not None else "error",
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
