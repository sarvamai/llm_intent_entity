# LLM Intent & Entity Evaluation

This repository provides a sophisticated framework for evaluating the performance of Automatic Speech Recognition (ASR) models by focusing on meaning preservation rather than just textual accuracy. It uses Large Language Models (LLMs) to assess whether the core **intent** of a spoken utterance and its key **entities** are correctly captured in the transcribed text.

## Background and Motivation

Traditional metrics for evaluating ASR systems, such as Word Error Rate (WER) and Character Error Rate (CER), measure performance by comparing the ASR's output text to a human-verified reference, word for word. While useful, these metrics have a significant limitation: they do not measure **comprehensibility**.

As highlighted in Google's research on meaning preservation, ASR models with a high WER can still be incredibly useful if the core meaning of the user's speech is preserved [[https://research.google/blog/assessing-asr-performance-with-meaning-preservation/](https://research.google/blog/assessing-asr-performance-with-meaning-preservation/)]. This is especially true for applications like voice commands, messaging, and conversations where minor grammatical errors are tolerable.

This tool was created to bridge that gap, providing a more nuanced and practical assessment of ASR performance.

## Why Traditional Metrics Are Insufficient

Standard WER and CER penalize all variations from the reference text equally, which is a flawed approach. The impact of a transcription error depends heavily on the context. A minor change in spelling or a `matra` (vowel sign in Indic scripts) can have vastly different consequences in different sentences.

Consider these examples:

- **Benign Error (Low Impact):**
  - **Reference:** `मैं कल दिल्ली जाऊंगा` (I will go to Delhi tomorrow)
  - **ASR Output:** `मैं कल दिल्ली जाउंगा` (A minor, common spelling variation)
  - **Problem:** A traditional WER metric would penalize this, even though the meaning is perfectly preserved and any native speaker would understand it without issue.

- **Severe Error (High Impact):**
  - **Reference:** `मुझे तीन टिकट चाहिए` (I need three tickets)
  - **ASR Output:** `मुझे तीन की जगह चाहिए` (I need a place for three)
  - **Problem:** A single word change completely alters the user's intent, turning a request for tickets into a request for space. A simple WER score might not adequately reflect the severity of this functional failure.

Traditional metrics cannot distinguish between these two scenarios. They lack the semantic understanding to know when an error is trivial versus when it breaks the user's intended communication.

## Methodology: Evaluating Meaning with LLMs

To overcome these limitations, this framework uses an LLM to perform a deeper, semantic evaluation of the ASR output against the ground truth. The evaluation is broken down into two key components:

### 1. Intent Preservation Score

The LLM is asked to determine if the **core meaning or intent** of the sentence is preserved. It is instructed to be extremely strict and focus on whether the fundamental action and subject remain the same.

- **Score 1 (Intent Preserved):** The core message is intact. This includes cases with minor spelling variations, equivalent phrasing ("please repeat" vs. "say that again"), or acceptable colloquialisms.
- **Score 0 (Intent Broken):** The meaning has changed. This includes errors that alter the subject ("I will do it" vs. "You will do it"), reverse the action ("call him" vs. "he will call"), or change a statement to a question.

### 2. Entity Preservation Score

The LLM identifies key **entities** in the ground truth—such as people, places, dates, numbers, or specific objects—and then calculates how well these are preserved in the ASR output.

- The score is a float between 0 and 1, representing the fraction of entities that were correctly transcribed.
- It penalizes for missing entities (e.g., "call John" vs. "call") and incorrectly substituted entities (e.g., "John Smith" vs. "Jon Smith").
- If the ground truth contains no entities (e.g., "hello, how are you?"), the score is 1, as there were no entities to get wrong.

## Conclusion

By focusing on intent and entity preservation, this LLM-based evaluation provides a far more accurate and practical assessment of an ASR model's real-world utility. It moves beyond rigid textual comparisons to measure what truly matters: whether the user's meaning was successfully understood. This approach is essential for developing ASR systems that are robust, reliable, and truly useful, especially in the linguistically diverse context of Indic languages.

## Installation

1. Clone/get the project
2. cd llm_evaluation/llm_intent_entity
3. uv venv --python 3.12
4. source .venv/bin/activate
5. uv pip sync uv.lock
6. uv pip install -e . (Or install in editable mode)

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
    gemini_location=""
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
