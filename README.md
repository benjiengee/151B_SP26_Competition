# CSE 151B Math Reasoning Competition

### Team Medevac: Benjamin Ng, Miguel Santos, Kanishk Hari, & Derek Huang

This repository contains our final inference pipeline for the **CSE 151B Spring 2026 Math Reasoning Competition**.

---

# Quick Start: Reproduce Our Kaggle Submission

This is the intended path for instructional staff verification.

From the repo root, install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Then run:

```bash
python run_inference.py
```

This command calls `run_inference()` using the default inference configuration hardcoded in `run_inference.py`.

These defaults are the same model path, prompt templates, sampling parameters, vLLM settings, and output formatting used for our Kaggle submission. To reproduce our submitted results, do not change the model path or inference hyperparameters.

By default, the script reads:

```text
data/private.jsonl
```

and writes timestamped output files to:

```text
results/
```

Example output files:

```text
results/submission_0531_1934.csv
results/submission_0531_1934.jsonl
```

The `.csv` file is the Kaggle submission file and has exactly:

```csv
id,response
```

No manual post-processing is required.

---

## Notes on Reproducibility

The course instructions state that string-identical outputs are not required because generation can vary. This repository includes the final model path, prompts, inference hyperparameters, and output formatting used for our submission.

The intended reproducibility setup is:

```text
same repo + same Hugging Face model + same hyperparameters + comparable GPU
```

We used an A100 GPU for final inference. Running on smaller GPUs may require lowering memory-related vLLM settings.

The final CSV produced by `run_inference()` is written automatically, so no manual post-processing is required.

*Note: The `0.660` public score on Kaggle was generated through a run on `prompt_engineering.ipynb`. Minimal score fluctuation is expected since the model, hyperparameters, and any other impacting variables remain the same in finalized `run_inference()` pipeline. `run_inference()` is essentially `prompt_engineering.ipynb` refactored.*

---

## Single Entry Point (Overview)
The main entry point is `run_inference()` in `run_inference.py`. Calling this function runs the full pipeline end-to-end:

1. Loads the model.
2. Loads the dataset.
3. Builds prompts.
4. Runs inference with vLLM.
5. Writes the final Kaggle submission CSV.

The final CSV contains the columns:

```csv
id,response
```

The `response` column contains the full model-generated response, including the reasoning trace and final boxed answer.

---

## Single Entry Point (Details)

The required single entry point is:

```python
run_inference()
```

It is defined in:

```text
run_inference.py
```

For command-line verification, run:

```bash
python run_inference.py
```

The bottom of `run_inference.py` contains:

```python
if __name__ == "__main__":
    run_inference()
```

This means running `python run_inference.py` executes the final private-set inference pipeline directly.

For Python or notebook usage, the same function can be called with:

```python
from run_inference import run_inference

run_inference()
```

For TA verification, use the command-line path:

```bash
python run_inference.py
```

---

## Model Weights

Our inference pipeline loads the model from Hugging Face.

Final model path:

```text
Qwen/Qwen3-4B-Thinking-2507
```

Supervised fine-tuned model path:

```text
benjiengee/qwen3-4b-thinking-sft-merged
```

The model path is hardcoded in `run_inference.py` inside the `InferenceConfig` class so that the pipeline matches our Kaggle submission:

```python
model_id: str = "TODO: replace with final Hugging Face model path"
```

During verification, the model is downloaded automatically from Hugging Face when `run_inference()` is called. No manual model download is required as long as the Hugging Face model repository is public.

Base model:

```text
Qwen/Qwen3-4B-Thinking-2507
```

---

## Hardware Used

Our final inference run was performed on:

```text
GPU: NVIDIA A100-SXM4-40GB
Platform: Google Colab Pro
```

Approximate total inference time:

```text
Private set, 16k max tokens: 3 hours
```

The pipeline was tested on an A100 GPU. Smaller GPUs may require reducing memory-related vLLM parameters such as:

```text
max_tokens
max_model_len
max_num_seqs
max_num_batched_tokens
```

---

## Environment Setup

This project was tested in a CUDA 12.8-compatible environment with:

```text
Python 3.12
torch 2.9.0+cu128
transformers 4.56.2
vllm 0.11.1
```

Install dependencies with:

```bash
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

If a conflicting vLLM installation already exists, run this cleanup first:

```bash
pip uninstall -y vllm vllm-flash-attn flashinfer-python flashinfer-cubin humming-kernels tokenspeed-mla tokenspeed-triton quack-kernels
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

We pin `vllm==0.11.1` because newer vLLM versions caused CUDA runtime compatibility issues in our Colab environment.

---

## Expected `requirements.txt`

The repo should include a `requirements.txt` similar to:

```text
transformers==4.56.2
sympy
numpy
tqdm
bitsandbytes
pandas
huggingface_hub
antlr4-python3-runtime==4.11.1
vllm==0.11.1
```

---

## Output Files

`run_inference()` writes two output files.

### 1. Kaggle CSV

The CSV file is the final Kaggle submission file.

It contains exactly:

```csv
id,response
```

Example:

```csv
id,response
0,"The solution is ... \boxed{105950}"
1,"The correct answer is ... \boxed{F}"
```

### 2. JSONL Record File

The JSONL file is for debugging and recordkeeping.

For private data, each row contains:

```json
{
  "id": 0,
  "is_mcq": false,
  "response": "..."
}
```

For public evaluation, each row additionally contains gold answers and correctness:

```json
{
  "id": 0,
  "is_mcq": false,
  "gold": ["325*(1+325)"],
  "response": "...",
  "correct": true
}
```

---

## Final Inference Configuration

The final inference configuration is defined in `run_inference.py`.

This is the configuration used for our Kaggle submission. To reproduce our submitted results, do not change the model path, prompt templates, sampling parameters, or vLLM settings.

```text
max_tokens = 16384
max_model_len = 32768
max_num_seqs = 16
max_num_batched_tokens = 8192
temperature = 0.6
top_p = 0.95
top_k = 20
quantization = bitsandbytes
```

The prompt templates and all sampling parameters are included directly in `run_inference.py`.

When running `python run_inference.py`, the terminal prints a formatted run configuration before inference begins, including:

```text
Run timestamp
Model ID
Data path
Output CSV path
Output JSONL path
Max tokens
Max model length
Max sequences
Max batched tokens
Sampling parameters
Evaluation/scoring mode
```

This is intended to make each run easier to verify and trace.

---

## Optional: Manual Calls for Testing and Experimentation

The following examples are provided for development, debugging, and public-set evaluation. They are not required for reproducing our Kaggle submission.

For TA verification, use:

```bash
python run_inference.py
```

### Quick Local Public Test

For quick testing, temporarily change the bottom of `run_inference.py` from:

```python
if __name__ == "__main__":
    run_inference()
```

to:

```python
if __name__ == "__main__":
    run_inference(
        data_path="data/public.jsonl",
        output_dir="results",
        run_name="public_eval_5_16k",
        eval_n=5,
        score_outputs=True,
    )
```

Then run:

```bash
python run_inference.py
```

This avoids issues with vLLM multiprocessing that can occur when running inline Python through stdin.

Before final submission or TA verification, the bottom of `run_inference.py` should be restored to:

```python
if __name__ == "__main__":
    run_inference()
```

### Explicit Private-Set Call

This is equivalent in purpose to the default command-line run, but allows changing output names during experimentation:

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_dir="results",
    run_name="private_submission_16k",
    eval_n=-1,
    score_outputs=False,
)
```

This produces files like:

```text
results/private_submission_16k_0531_1934.csv
results/private_submission_16k_0531_1934.jsonl
```

### Public Evaluation from Python or Notebook

To test the pipeline on public data with scoring from Python or a notebook:

```python
from run_inference import run_inference

run_inference(
    data_path="data/public.jsonl",
    output_dir="results",
    run_name="public_eval_10_16k",
    eval_n=10,
    score_outputs=True,
)
```

This will:

1. Load the first 10 public examples.
2. Run inference.
3. Score the responses using `judger.py`.
4. Save timestamped JSONL and CSV output files.

For a 100-question public evaluation:

```python
run_inference(
    data_path="data/public.jsonl",
    output_dir="results",
    run_name="public_eval_100_16k",
    eval_n=100,
    score_outputs=True,
)
```

To evaluate the full public dataset:

```python
run_inference(
    data_path="data/public.jsonl",
    output_dir="results",
    run_name="public_eval_full_16k",
    eval_n=-1,
    score_outputs=True,
)
```

---

## Repository Structure

```text
151B_SP26_Competition/
├── README.md
├── run_inference.py
├── requirements.txt
├── judger.py
├── utils.py
├── data/
│   ├── public.jsonl
│   └── private.jsonl
├── results/
├── prompt_engineering.ipynb
└── sft_train.ipynb
```

---

## LLM Disclaimer
We acknowledge the use of AI tools in the development of this project. All final code, experiments, model choices, hyperparameters, and submissions were reviewed and made by the project team. AI tools were used in the following contexts:

- **Environment setup and debugging:** Colab setup, A100 runtime setup, CUDA/vLLM compatibility, package/version conflicts

- **Error debugging:** vLLM engine errors, CUDA library errors, multiprocessing issues, GPU memory issues, Colab disconnections

- **Pipeline design and organization:** `run_inference()` structure, single-entry-point design, public/private evaluation workflow

- **Code generation and refactoring:** helper functions, JSONL/CSV output writing, timestamped filenames, cleaner terminal logging

- **Prompt engineering support:** math reasoning prompts, multiple-choice prompts, answer-format instructions

- **Training and inference guidance:** SFT setup, dataset filtering/subsetting, token limits, batching settings, inference hyperparameters

- **Result formatting and submission preparation:** Kaggle `id,response` CSV format, raw response preservation, output validation

- **Documentation:** README setup instructions, model-weight instructions, reproducibility notes, usage examples

- **Git and workflow support:** branch management, pull/rebase/merge guidance, commit inspection, repository cleanup
