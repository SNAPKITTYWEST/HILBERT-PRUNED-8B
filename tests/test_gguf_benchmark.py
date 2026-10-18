import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, "benchmarks/gguf_benchmark.py", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_existing_output_is_refused_before_loading(tmp_path):
    result = run("--model", "unused", "--revision", "unused", "--gguf", "missing.gguf",
                 "--quant", "Q4_K_M", "--output", str(tmp_path))
    assert result.returncode == 2
    assert "Output directory must be new" in result.stderr


def test_committed_gguf_runs_match_their_checksums():
    for sums in sorted((ROOT / "results/gguf").glob("*/SHA256SUMS")):
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            assert hashlib.sha256((sums.parent / name.strip()).read_bytes()).hexdigest() == digest, name


def test_committed_gguf_summaries_are_consistent():
    for summary_path in sorted((ROOT / "results/gguf").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        quality = json.loads((summary_path.parent / "quality.json").read_text())
        for suite, values in summary.items():
            flips = quality[suite]["vs_published_torch"]["argmax_flips"]
            rows = quality[suite]["vs_published_torch"]["rows"]
            assert values["argmax_flips_vs_bf16"] == len(flips)
            assert abs(values["agreement_with_bf16"] - (1 - len(flips) / rows)) < 1e-12
            assert values["prompt_mismatches"] == 0
