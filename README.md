# CSE 151B Math Reasoning Competition

This repository contains our final inference pipeline for the CSE 151B Spring 2026 Math Reasoning Competition.

The main entry point is run_inference() in run_inference.py. Calling this function runs the full pipeline end-to-end:

1. Loads the model.
2. Loads the dataset.
3. Builds prompts.
4. Runs inference with vLLM.
5. Writes the final Kaggle submission CSV.

The final CSV contains the columns:

csv id,response 

The response column contains the full model-generated response, including the reasoning trace and final boxed answer.

---

## Quick Start: Reproduce Final Inference

From the repo root, install dependencies:

bash pip install -U pip pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 

Then run:

bash python run_inference.py 

By default, this calls:

python run_inference() 

and reads:

text data/private.jsonl 

It writes timestamped output files to:

text results/ 

Example output files:

text results/submission_0531_1934.csv results/submission_0531_1934.jsonl 

The .csv file is the Kaggle submission file.

---

## Single Entry Point

The required single entry point is:

python run_inference() 

It is defined in:

text run_inference.py 

Default usage:

python from run_inference import run_inference  run_inference() 

Equivalent command-line usage:

bash python run_inference.py 

The default function call runs inference on the private dataset and writes the final submission CSV.

---

## Running the Private Set

To explicitly run the private set:

python from run_inference import run_inference  run_inference(     data_path="data/private.jsonl",     output_dir="results",     run_name="private_submission_16k",     eval_n=-1,     score_outputs=False, ) 

This produces files like:

text results/private_submission_16k_0531_1934.csv results/private_submission_16k_0531_1934.jsonl 

The CSV has exactly:

csv id,response 

---

## Quick Public Evaluation

To test the pipeline on public data with scoring:

python from run_inference import run_inference  run_inference(     data_path="data/public.jsonl",     output_dir="results",     run_name="public_eval_10_16k",     eval_n=10,     score_outputs=True, ) 

This will:

1. Load the first 10 public examples.
2. Run inference.
3. Score the responses using judger.py.
4. Save timestamped JSONL and CSV output files.

For a 100-question public evaluation:

python run_inference(     data_path="data/public.jsonl",     output_dir="results",     run_name="public_eval_100_16k",     eval_n=100,     score_outputs=True, ) 

To evaluate the full public dataset:

python run_inference(     data_path="data/public.jsonl",     output_dir="results",     run_name="public_eval_full_16k",     eval_n=-1,     score_outputs=True, ) 

---

## Model Weights

Our inference pipeline loads the model from Hugging Face.

Final model path:

text TODO: replace with final Hugging Face model path 

For example:

text benjiengee/qwen3-4b-thinking-sft-merged 

If using the base model instead of the fine-tuned model, the base model path is:

text Qwen/Qwen3-4B-Thinking-2507 

The model path is set in run_inference.py inside the InferenceConfig class:

python model_id: str = "TODO: replace with final Hugging Face model path" 

During verification, the model is downloaded automatically from Hugging Face when run_inference() is called. No manual model download is required as long as the Hugging Face model repository is public.

---

## Hardware Used

Our final inference run was performed on:

text GPU: NVIDIA A100-SXM4-40GB Platform: Google Colab Pro 

Approximate total inference time:

text Private set, 16k max tokens: TODO: fill in final runtime 

The pipeline was tested on an A100 GPU. Smaller GPUs may require reducing memory-related vLLM parameters such as:

text max_tokens max_model_len max_num_seqs max_num_batched_tokens 

---

## Environment Setup

This project was tested in a CUDA 12.8-compatible environment with:

text Python 3.12 torch 2.9.0+cu128 transformers 4.56.2 vllm 0.11.1 

Install dependencies with:

bash pip install -U pip pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 

If a conflicting vLLM installation already exists, run this cleanup first:

bash pip uninstall -y vllm vllm-flash-attn flashinfer-python flashinfer-cubin humming-kernels tokenspeed-mla tokenspeed-triton quack-kernels pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 

We pin vllm==0.11.1 because newer vLLM versions caused CUDA runtime compatibility issues in our Colab environment.

---

## Expected requirements.txt

The repo should include a requirements.txt similar to:

text transformers==4.56.2 sympy numpy tqdm bitsandbytes pandas huggingface_hub antlr4-python3-runtime==4.11.1 vllm==0.11.1 

---

## Output Files

run_inference() writes two output files.

### 1. Kaggle CSV

The CSV file is the final Kaggle submission file.

It contains exactly:

csv id,response 

Example:

csv id,response 0,"The solution is ... \boxed{105950}" 1,"The correct answer is ... \boxed{F}" 

### 2. JSONL Record File

The JSONL file is for debugging and recordkeeping.

For private data, each row contains:

json {   "id": 0,   "is_mcq": false,   "response": "..." } 

For public evaluation, each row additionally contains gold answers and correctness:

json {   "id": 0,   "is_mcq": false,   "gold": ["325*(1+325)"],   "response": "...",   "correct": true } 

---

## Final Inference Hyperparameters

The final inference configuration is defined in run_inference.py.

text max_tokens = 16384 max_model_len = 32768 max_num_seqs = 16 max_num_batched_tokens = 8192 temperature = 0.6 top_p = 0.95 top_k = 20 quantization = bitsandbytes 

The prompt templates and all sampling parameters are included directly in run_inference.py.

---

## Repository Structure

text 151B_SP26_Competition/ ├── README.md ├── run_inference.py ├── requirements.txt ├── judger.py ├── utils.py ├── data/ │   ├── public.jsonl │   └── private.jsonl ├── results/ ├── prompt_engineering.ipynb └── sft_train.ipynb 

---

## Notes on Reproducibility

The course instructions state that string-identical outputs are not required because generation can vary. This repository includes the final model path, prompts, inference hyperparameters, and output formatting used for our submission.

The intended reproducibility setup is:

text same repo + same Hugging Face model + same hyperparameters + comparable GPU 

We used an A100 GPU for final inference. Running on smaller GPUs may require lowering memory-related vLLM settings.

The final CSV produced by run_inference() is written automatically, so no manual post-processing is required.