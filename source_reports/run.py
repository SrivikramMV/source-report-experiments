"""Collect source reasoning and matched/conflict source reports."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from .models import NativeReasoning, PROFILES


INSTRUCTION = (
    "Before solving, separately quote the exact value or equation from the "
    "scratchpad that determines the answer and the corresponding value or "
    "equation from the user prompt. Do not reconstruct the scratchpad from "
    "the prompt. State whether the quoted evidence describes the same "
    "mathematical problem state; mathematically equivalent expressions count "
    "as matching even if their wording differs. Then solve only the user's problem."
)
SOURCE_CAP = 8192
REPORT_CAP = 2048
CONTEXT_CAP = 12288


def digest(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def load_pairs(location: str, split: str, limit: int | None):
    from datasets import load_dataset

    local = Path(location)
    if local.is_dir():
        path = local / "data" / f"{split}.jsonl"
        dataset = load_dataset("json", data_files={split: str(path)}, split=split)
    elif local.is_file():
        dataset = load_dataset("json", data_files={split: str(local)}, split=split)
    else:
        dataset = load_dataset(location, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    pairs = list(dataset)
    required = {"id", "family", "source_question", "matched_question", "conflict_question"}
    for pair in pairs:
        missing = required - pair.keys()
        if missing:
            raise ValueError(f"Pair {pair.get('id')} is missing {sorted(missing)}")
    if len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate pair IDs in the selected dataset")
    return pairs


def append_row(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(row["request_id"], row["status"], flush=True)


def generate(engine, interface, prompt_ids, cap, sampling_type, profile, seed):
    if len(prompt_ids) + cap > CONTEXT_CAP:
        raise ValueError("Prompt and generation budget exceed the context limit")
    params = sampling_type(
        temperature=0.0,
        top_p=profile.top_p,
        top_k=-1,
        repetition_penalty=profile.repetition_penalty,
        presence_penalty=0.0,
        max_tokens=cap,
        seed=seed,
        stop_token_ids=interface.stop_ids,
        skip_special_tokens=False,
    )
    response = engine.generate(
        [{"prompt_token_ids": prompt_ids}], params, use_tqdm=False
    )[0]
    if not response.finished or len(response.outputs) != 1:
        raise RuntimeError("Generation returned an incomplete or ambiguous response")
    output = response.outputs[0]
    ids = list(output.token_ids)
    if not ids or output.finish_reason != "stop":
        raise RuntimeError(f"Generation ended with {output.finish_reason!r}")
    return ids, output.finish_reason


def collect(args, pairs):
    profile = PROFILES[args.model]
    directory = args.output / profile.name / args.split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.jsonl"
    manifest_path = directory / "run.json"
    manifest = {
        "model": profile.model_id,
        "revision": profile.revision,
        "dataset": args.dataset,
        "split": args.split,
        "pair_ids": [pair["id"] for pair in pairs],
        "seed": args.seed,
        "source_cap": SOURCE_CAP,
        "report_cap": REPORT_CAP,
        "context_cap": CONTEXT_CAP,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "instruction": INSTRUCTION,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Run settings differ from {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["request_id"] in completed:
                raise ValueError("Duplicate request in saved results")
            completed[row["request_id"]] = row

    from vllm import LLM, SamplingParams

    interface = NativeReasoning(profile)
    options = dict(
        model=profile.model_id,
        revision=profile.revision,
        tokenizer_revision=profile.revision,
        dtype=profile.dtype,
        max_model_len=CONTEXT_CAP,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=args.seed,
    )
    if profile.language_model_only:
        options["language_model_only"] = True
    engine = LLM(**options)

    for pair in pairs:
        source_id = pair["id"] + ":source"
        source = completed.get(source_id)
        if source is None:
            source = dict(
                request_id=source_id, case_id=pair["id"], family=pair["family"],
                stage="source", status="failed", question=pair["source_question"],
            )
            try:
                prompt = interface.question_prompt(pair["source_question"])
                output_ids, reason = generate(
                    engine, interface, prompt, SOURCE_CAP, SamplingParams, profile, args.seed
                )
                trace_ids, answer_ids = interface.source_parts(output_ids)
                source.update(
                    status="generated", reasoning=interface.decode(trace_ids),
                    reasoning_token_ids=trace_ids, reasoning_sha256=digest(trace_ids),
                    final_answer=interface.decode(answer_ids),
                    generated_token_ids=output_ids, finish_reason=reason,
                    prompt_sha256=digest(prompt),
                )
            except Exception as exc:
                source.update(error_type=type(exc).__name__, error=str(exc))
            append_row(path, source)
            completed[source_id] = source

        for condition in ("matched", "conflict"):
            request_id = pair["id"] + ":" + condition
            if request_id in completed:
                continue
            question = pair[condition + "_question"]
            row = dict(
                request_id=request_id, case_id=pair["id"], family=pair["family"],
                stage="report", condition=condition, question=question,
                source_request_id=source_id, status="failed",
            )
            if source["status"] != "generated":
                row.update(status="source_unavailable")
            else:
                try:
                    trace_ids = source["reasoning_token_ids"]
                    prompt = interface.report_prompt(question, trace_ids, INSTRUCTION)
                    output_ids, reason = generate(
                        engine, interface, prompt, REPORT_CAP, SamplingParams,
                        profile, args.seed,
                    )
                    row.update(
                        status="generated", report=interface.decode(output_ids),
                        generated_token_ids=output_ids, finish_reason=reason,
                        source_reasoning_sha256=digest(trace_ids),
                        prompt_sha256=digest(prompt),
                    )
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc))
            append_row(path, row)
            completed[request_id] = row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROFILES, required=True)
    parser.add_argument("--dataset", default="dataset", help="Local dataset folder or HF dataset ID")
    parser.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    collect(args, load_pairs(args.dataset, args.split, args.limit))


if __name__ == "__main__":
    main()
