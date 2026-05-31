import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    # Hardcode this to the exact model used for the Kaggle submission.
    model_id: str = "Qwen/Qwen3-4B-Thinking-2507"

    # Default final-submission dataset.
    data_path: str = "data/public.jsonl"

    # These are usually overwritten with timestamped names.
    output_csv_path: str = "results/submission.csv"
    output_jsonl_path: str = "results/final_results.jsonl"

    gpu_id: str = "0"

    # Final inference settings used for Kaggle submission.
    max_tokens: int = 16384
    max_model_len: int = 32768
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # Use -1 for all examples, or a positive integer for quick testing.
    eval_n: int = 5

    # Public data has answers; private data does not.
    score_outputs: bool = False


SYSTEM_PROMPT_MATH = (
    "You are a precise mathematical reasoner. Solve the following problem rigorously. "
    "After obtaining an answer, independently check it for errors or contradictions. "
    "Return only the corrected final solution inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}. "
    "Be concise. "
)

SYSTEM_PROMPT_MCQ = (
    "You are a precise mathematical reasoner. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "After obtaining an answer, independently check it for errors or contradictions. "
    "Output ONLY the letter of your final chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "Be concise. "
)


# =============================================================================
# Pretty printing helpers
# =============================================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def print_kv(key: str, value) -> None:
    print(f"{key:<24}: {value}")


# =============================================================================
# Timestamped output paths
# =============================================================================

def make_run_timestamp() -> str:
    """Return a compact filesystem-safe timestamp in Pacific time."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d_%H%M")


def make_output_paths(
    output_dir: str = "results",
    run_name: str = "submission",
    timestamp: Optional[str] = None,
) -> tuple[str, str]:
    """
    Create timestamped JSONL and CSV output paths.

    Example:
      results/submission_0531_1934.jsonl
      results/submission_0531_1934.csv
    """
    if timestamp is None:
        timestamp = make_run_timestamp()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    base = output_dir_path / f"{run_name}_{timestamp}"
    return str(base.with_suffix(".jsonl")), str(base.with_suffix(".csv"))


# =============================================================================
# Data loading
# =============================================================================

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_eval_data(data: list[dict], eval_n: int) -> list[dict]:
    if eval_n > 0:
        return data[:eval_n]
    return data


def get_dataset_stats(data: list[dict]) -> tuple[int, int, int]:
    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options") for d in data)
    return len(data), n_mcq, n_free


def print_dataset_stats(data: list[dict], label: str) -> None:
    total, n_mcq, n_free = get_dataset_stats(data)
    print_kv(label, f"{total} questions ({n_mcq} MCQ, {n_free} free-form)")


# =============================================================================
# Prompt construction
# =============================================================================

def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{label}. {option.strip()}"
            for label, option in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    return SYSTEM_PROMPT_MATH, question


def build_chat_prompts(data: list[dict], tokenizer: AutoTokenizer) -> list[str]:
    prompts = []

    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    return prompts


# =============================================================================
# Model loading and generation
# =============================================================================

def load_model_and_tokenizer(config: InferenceConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    print_subsection("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print_subsection("Loading model with vLLM")
    llm = LLM(
        model=config.model_id,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("\nModel loaded successfully.")
    return tokenizer, llm, sampling_params


def generate_responses(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[str]:
    print_subsection("Generating responses")
    print_kv("Number of prompts", len(prompts))

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]

    print("\nGeneration complete.")
    return responses


# =============================================================================
# Scoring for public set only
# =============================================================================

def extract_letter(text: str) -> str:
    match = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == gold_letter.strip().upper()


def score_responses(data: list[dict], responses: list[str]) -> list[dict]:
    sys.path.insert(0, ".")
    from judger import Judger

    judger = Judger(strict_extract=False)
    results = []

    print_subsection("Scoring responses")

    for item, response in tqdm(zip(data, responses), total=len(data), desc="Scoring"):
        is_mcq = bool(item.get("options"))
        gold = item["answer"]

        if is_mcq:
            correct = score_mcq(response, str(gold))
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except Exception:
                correct = False

        results.append(
            {
                "id": item.get("id"),
                "is_mcq": is_mcq,
                "gold": gold,
                "response": response,
                "correct": correct,
            }
        )

    print("\nScoring complete.")
    return results


def print_summary(results: list[dict]) -> None:
    mcq_res = [r for r in results if r["is_mcq"]]
    free_res = [r for r in results if not r["is_mcq"]]

    def acc(subset: list[dict]) -> float:
        return sum(r["correct"] for r in subset) / len(subset) * 100 if subset else 0.0

    print_section("Evaluation Results")
    print(
        f"{'MCQ':<12}: {sum(r['correct'] for r in mcq_res):4d} / "
        f"{len(mcq_res):4d} ({acc(mcq_res):.2f}%)"
    )
    print(
        f"{'Free-form':<12}: {sum(r['correct'] for r in free_res):4d} / "
        f"{len(free_res):4d} ({acc(free_res):.2f}%)"
    )
    print(
        f"{'Overall':<12}: {sum(r['correct'] for r in results):4d} / "
        f"{len(results):4d} ({acc(results):.2f}%)"
    )


# =============================================================================
# Saving outputs
# =============================================================================

def save_jsonl(path: str, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print_kv("Saved JSONL", output_path)


def save_submission_csv(
    path: str,
    data: list[dict],
    responses: list[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "id": item.get("id"),
            "response": response,
        }
        for item, response in zip(data, responses)
    ]

    rows = sorted(rows, key=lambda row: int(row["id"]))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)

    print_kv("Saved CSV", output_path)


def build_detailed_records(
    data: list[dict],
    responses: list[str],
    scored_results: Optional[list[dict]] = None,
) -> list[dict]:
    if scored_results is not None:
        return scored_results

    return [
        {
            "id": item.get("id"),
            "is_mcq": bool(item.get("options")),
            "response": response,
        }
        for item, response in zip(data, responses)
    ]


# =============================================================================
# Single entry point
# =============================================================================

def run_inference(
    data_path: str = "data/private.jsonl",
    output_csv_path: Optional[str] = None,
    output_jsonl_path: Optional[str] = None,
    output_dir: str = "results",
    run_name: str = "submission",
    eval_n: int = -1,
    score_outputs: bool = False,
) -> None:
    """
    Single entry point for the full inference pipeline.

    Loads data, loads model, generates responses, optionally scores public data,
    and writes the final Kaggle CSV.

    If output paths are not provided, timestamped JSONL and CSV files are created.
    """
    run_timestamp = make_run_timestamp()

    if output_jsonl_path is None or output_csv_path is None:
        auto_jsonl_path, auto_csv_path = make_output_paths(
            output_dir=output_dir,
            run_name=run_name,
            timestamp=run_timestamp,
        )

        if output_jsonl_path is None:
            output_jsonl_path = auto_jsonl_path

        if output_csv_path is None:
            output_csv_path = auto_csv_path

    config = InferenceConfig(
        data_path=data_path,
        output_csv_path=output_csv_path,
        output_jsonl_path=output_jsonl_path,
        eval_n=eval_n,
        score_outputs=score_outputs,
    )

    print_section("Run Configuration")
    print_kv("Run timestamp", run_timestamp)
    print_kv("Model ID", config.model_id)
    print_kv("Data path", config.data_path)
    print_kv("Output CSV", config.output_csv_path)
    print_kv("Output JSONL", config.output_jsonl_path)
    print_kv("Max tokens", config.max_tokens)
    print_kv("Max model length", config.max_model_len)
    print_kv("Max sequences", config.max_num_seqs)
    print_kv("Max batched tokens", config.max_num_batched_tokens)
    print_kv("Temperature", config.temperature)
    print_kv("Top-p", config.top_p)
    print_kv("Top-k", config.top_k)
    print_kv("Eval N", config.eval_n)
    print_kv("Score outputs", config.score_outputs)

    print_section("Loading Dataset")
    data = load_jsonl(config.data_path)
    print_dataset_stats(data, label="Full dataset")

    eval_data = select_eval_data(data, config.eval_n)
    print_dataset_stats(eval_data, label="Inference set")

    tokenizer, llm, sampling_params = load_model_and_tokenizer(config)

    print_section("Building Prompts")
    prompts = build_chat_prompts(eval_data, tokenizer)
    print_kv("Prompts built", len(prompts))

    responses = generate_responses(llm, prompts, sampling_params)

    assert len(responses) == len(eval_data), (
        f"Expected {len(eval_data)} responses, got {len(responses)}"
    )

    scored_results = None
    if config.score_outputs:
        scored_results = score_responses(eval_data, responses)
        print_summary(scored_results)

    detailed_records = build_detailed_records(
        eval_data,
        responses,
        scored_results=scored_results,
    )

    print_section("Saving Outputs")
    save_jsonl(config.output_jsonl_path, detailed_records)
    save_submission_csv(config.output_csv_path, eval_data, responses)

    print_section("Done")
    print_kv("JSONL", config.output_jsonl_path)
    print_kv("CSV", config.output_csv_path)


# =============================================================================
# Command-line entry
# =============================================================================

if __name__ == "__main__":
    # Final private-set submission run.
    # This is the configuration the TA should use for reproducibility.
    run_inference()

    # For quick local testing, temporarily replace the above with:
    #
    # run_inference(
    #     data_path="data/public.jsonl",
    #     output_dir="results",
    #     run_name="public_eval_5_16k",
    #     eval_n=5,
    #     score_outputs=True,
    # )