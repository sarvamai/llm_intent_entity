# LLM Intent Entity Evaluation

This repository provides tools for evaluating intent and entity extraction accuracy in ASR (Automatic Speech Recognition) outputs using Large Language Models (LLMs), specifically Google's Gemini 2.5 Flash via Vertex AI.

## Overview

The tool compares ASR transcriptions against ground truth text and evaluates:
- **Intent Preservation**: Whether the core meaning/intent of the sentence is preserved (scored 0 or 1)
- **Entity Extraction**: How well named entities, dates, numbers, and other key information are preserved (scored 0-1)

## Features

- **Comprehensive Evaluation**: Evaluates both intent and entity preservation
- **Multi-language Support**: Specialized support for Indian languages with proper normalization
- **Caching**: Avoids redundant API calls by caching results
- **Google Sheets Integration**: Automatically pushes results to Google Sheets
- **Detailed Metrics**: Provides accuracy, mean/median scores, and combined metrics
- **Error Handling**: Robust error handling with detailed logging

## Requirements

- Python 3.12+
- Google Cloud credentials for Vertex AI
- Google Sheets API credentials (optional, for sheets integration)

## Installation

1. Clone or create the repository structure
2. Install dependencies:
```bash
pip install -e .
```

## Setup

1. **Google Cloud Credentials**: Ensure you have a service account JSON file with Vertex AI permissions
2. **Google Sheets Credentials** (optional): Service account JSON file with Google Sheets API access

## Usage

### Basic Usage

```python
from main import process_dataset_for_intent_entity_evaluation

process_dataset_for_intent_entity_evaluation(
    dataset_path="/path/to/your_dataset.csv",
    reference_col_name="ground_truth",
    predicted_col_name="asr_output", 
    audio_filepath_col_name="audio_file",
    creds_path="/path/to/vertex_ai_creds.json",
    language_col_name="language",
    context_col_name="context",
    output_sheet_name="Intent Entity Analysis",
    output_worksheet_name="Results",
    ignore_cache=False,
    gemini_location="us-central1"
)
```

### Dataset Format

Your dataset should be a CSV or JSONL file with the following columns:
- `ground_truth`: The reference/ground truth transcription
- `asr_output`: The ASR model's transcription
- `audio_file`: Path to the audio file
- `language`: Language of the audio (e.g., "hindi", "english", "tamil")
- `context`: Optional context information for better evaluation

Example CSV:
```csv
ground_truth,asr_output,audio_file,language,context
"मैं कल दिल्ली जाऊंगा","मैं कल दिल्ली जाउंगा","/path/audio1.wav","hindi","Travel planning"
"Please call John Smith","Please call Jon Smith","/path/audio2.wav","english","Phone conversation"
```

### Output

The tool generates:
1. **Main CSV**: Complete results with intent/entity scores and explanations
2. **LLM Logs**: Detailed logs of all LLM evaluations
3. **Failed Requests**: Any failed API calls for debugging
4. **Google Sheets**: Results automatically uploaded (if configured)

### Metrics

- **Intent Accuracy**: Percentage of samples where intent is preserved
- **Entity Score**: Mean/median/std of entity preservation scores (0-1)
- **Combined Score**: Weighted combination of intent and entity scores

## Configuration

### Supported Languages

The tool supports text normalization for:
- Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, English

### Evaluation Criteria

#### Intent Scoring (0 or 1)
- Strict evaluation of core meaning preservation
- Considers subject/object relationships, pronouns, questions vs statements
- Special handling for acknowledgments, fillers, and language preferences

#### Entity Scoring (0-1)
- Evaluates preservation of people, places, organizations, dates, numbers
- Supports partial scoring for partially preserved entities
- Handles Indic language-specific considerations (pronouns, negation, etc.)

## Files

- `main.py`: Main processing logic and orchestration
- `llm_api.py`: Vertex AI API wrapper with threading and error handling
- `utilities.py`: Text normalization and Google Sheets utilities
- `prompt_template.txt`: LLM evaluation prompt template
- `pyproject.toml`: Project dependencies

## Replication
- Clone/get the project
- cd llm_evaluation/llm_intent_entity
- uv venv --python 3.12
- source .venv/bin/activate
- uv pip sync uv.lock
- uv pip install -e . (Or install in editable mode)