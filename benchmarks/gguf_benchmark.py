"""Decision-quality evaluation of GGUF quantizations of the pinned 4B model.

Mirrors the quality suite of mlx_benchmark.py on the llama.cpp CPU backend:
authored144 and perturbations108 in direct mode, compared row by row with the
published Torch BF16 predictions. A quantized model is judged by decision
agreement with its BF16 reference (argmax flips, probability drift), not by
perplexity. Outputs are create-only.

    python benchmarks/gguf_benchmark.py \\
      --model Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \\
      --gguf hilbert-4b-baseline.Q4_K_M.gguf --quant Q4_K_M \\
      --conversion conversion.json --output results/gguf/<date>-q4_k_m
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import evaluate
from mlx_benchmark import choice, compare, read, write, write_rows

SUITES = ("authored144", "perturbations108")


def quality(backend, model, tokenizer, metadata, output, max_tokens):
    datasets, report = {}, {}
    for name in SUITES:
        gold = read(f"benchmarks/data/{name}.jsonl")
        backend.score(model, tokenizer, gold[0], metadata, max_tokens)  # warm-up, not recorded
        started = time.perf_counter()
        predictions = []
        for index, row in enumerate(gold):
            predictions.append(backend.score(model, tokenizer, row, metadata, max_tokens))
            if (index + 1) % 12 == 0:
                print(f"{name}: {index + 1}/{len(gold)} ({time.perf_counter() - started:.0f}s)", flush=True)
        elapsed = time.perf_counter() - started
        write_rows(output / f"{name}.jsonl", predictions)
        baseline = read(f"results/raw/predictions/direct-{name}.jsonl")
        report[name] = {"evaluation": evaluate.evaluate(gold, predictions, comparison=baseline),
                        "vs_published_torch": compare(baseline, predictions),
                        "wall_seconds": elapsed, "seconds_per_decision": elapsed / len(gold)}
        datasets[name] = (gold, predictions)
    gold, base = datasets["authored144"]
    perturb_gold, perturb = datasets["perturbations108"]
    by_id = {row["id"]: row for row in base}
    variants = defaultdict(list)
    for row, prediction in zip(perturb_gold, perturb):
        variants[row["provenance"]["variant"]].append((row, prediction))
    report["perturbation_stability"] = {}
    for name, pairs in variants.items():
        references = [dict(by_id[row["provenance"]["base_id"]], id=row["id"]) for row, _ in pairs]
        comparison = compare(references, [pred for _, pred in pairs])
        comparison.pop("prompt_mismatches")  # wording/order/context intentionally change these prompts
        report["perturbation_stability"][name] = comparison
    missing = [row for row in gold if row["provenance"]["variant"] == "missing"]
    report["missing_evidence_confident_non_insufficient_at_0_8"] = sum(
        choice(by_id[row["id"]]) != "insufficient" and max(by_id[row["id"]]["probabilities"]) >= .8
        for row in missing)
    write(output / "quality.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Reference tokenizer source (HF id or local dir)")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--quant", required=True, help="Quantization label recorded in the manifest, e.g. Q4_K_M")
    parser.add_argument("--conversion", type=Path, help="JSON describing how the GGUF was produced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new")
    from semif_phase1 import llamacpp_backend as backend

    model, tokenizer, metadata = backend.load_model(args.model, args.revision, args.gguf,
                                                    threads=args.threads, context_tokens=args.max_tokens)
    args.output.mkdir(parents=True)
    manifest = {
        "model": metadata,
        "quantization": args.quant,
        "conversion": json.loads(args.conversion.read_text()) if args.conversion else None,
        "reference": "results/raw/predictions/direct-{authored144,perturbations108}.jsonl (published Torch BF16)",
        "fixture_sha256": {f"{name}.jsonl": hashlib.sha256(Path(f"benchmarks/data/{name}.jsonl").read_bytes()).hexdigest()
                           for name in SUITES},
        "host": {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
        "suite": "quality",
    }
    write(args.output / "manifest.json", manifest)
    report = quality(backend, model, tokenizer, metadata, args.output, args.max_tokens)
    summary = {name: {"mean_family_balanced_accuracy": report[name]["evaluation"]["mean_family_balanced_accuracy"],
                      "argmax_flips_vs_bf16": len(report[name]["vs_published_torch"]["argmax_flips"]),
                      "agreement_with_bf16": 1 - len(report[name]["vs_published_torch"]["argmax_flips"]) / report[name]["vs_published_torch"]["rows"],
                      "max_probability_difference": report[name]["vs_published_torch"]["max_probability_difference"],
                      "mean_max_probability_difference": report[name]["vs_published_torch"]["mean_max_probability_difference"],
                      "prompt_mismatches": len(report[name]["vs_published_torch"]["prompt_mismatches"]),
                      "seconds_per_decision": report[name]["seconds_per_decision"]}
               for name in SUITES}
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
