# GGUF evidence: Hilbert 4B baseline on llama.cpp

These runs score GGUF quantizations of the **frozen, unpruned 4B baseline**
(`Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)
through the repository's own llama.cpp backend
([`src/semif_phase1/llamacpp_backend.py`](../../src/semif_phase1/llamacpp_backend.py)).
They are **not** the pruned/distilled Hilbert 4B, which remains PROPOSED.

The model files are published at
[huggingface.co/Snapkitty/hilbert-4b-baseline-GGUF](https://huggingface.co/Snapkitty/hilbert-4b-baseline-GGUF).
No weights are committed here.

## How the files were produced

Both files come from one BF16 conversion with the llama.cpp sources vendored
in `llama-cpp-python==0.3.35` (sdist SHA-256
`1139dbb54509074b70893fab8554e3b079aa9f4d312058ce4018ef0019e3de12`,
build-info commit `4df29be`), text model only, no importance matrix:

```bash
python vendor/llama.cpp/convert_hf_to_gguf.py <Qwen3.5-4B@851bf6e> --outtype bf16 \
  --outfile hilbert-4b-baseline.BF16.gguf
vendor/llama.cpp/build/bin/llama-quantize hilbert-4b-baseline.BF16.gguf hilbert-4b-baseline.Q8_0.gguf Q8_0 4
vendor/llama.cpp/build/bin/llama-quantize hilbert-4b-baseline.BF16.gguf hilbert-4b-baseline.Q4_K_M.gguf Q4_K_M 4
```

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `hilbert-4b-baseline.Q8_0.gguf` | 4,610,580,192 | `4c9aff9532cdfba6e27f0f6c102f6086cb79bca4ef779e1ac17d608dc5a0beab` |
| `hilbert-4b-baseline.Q4_K_M.gguf` | 2,783,446,752 | `e84b5e30c8ac4636e20c11685c1d7d269cbd1b46cc564daa95465d636045dfaa` |

Each run's `manifest.json` records the conversion, the GGUF hash, and the
runtime versions. Its `model.source` is a local snapshot of the pinned
revision (only the tokenizer files are read from it).

## How they were scored

```bash
python benchmarks/gguf_benchmark.py --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --gguf hilbert-4b-baseline.Q4_K_M.gguf --quant Q4_K_M --threads 4 \
  --output results/gguf/<new-dir>
```

Direct mode on `authored144` and `perturbations108`, CPU only (4 threads,
x86-64, no GPU offload), each compared row by row with the published Torch
BF16 predictions in `results/raw/predictions/`. Prompts are built by the
reference tokenizer and every scored prompt is re-tokenized through the GGUF
vocabulary, so `prompt_sha256` and token counts are comparable with BF16.

## Results

| Run | Suite | Agreement with BF16 | Argmax flips | Balanced accuracy | Δ vs BF16 (95% paired bootstrap) | Mean max prob. diff | s/decision |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Q8_0 | authored144 | 99.31% | 1 / 144 | 80.67% | −0.65 pp [−2.22, 0.00] | 0.013 | 1.94 |
| Q8_0 | perturbations108 | 98.15% | 2 / 108 | 76.58% | −1.41 pp [−3.67, 0.00] | 0.018 | 2.11 |
| Q4_K_M | authored144 | 92.36% | 11 / 144 | 84.42% | +3.10 pp [+0.04, +6.60] | 0.075 | 2.28 |
| Q4_K_M | perturbations108 | 91.67% | 9 / 108 | 76.08% | −1.90 pp [−6.30, +2.99] | 0.098 | 2.47 |

BF16 reference: 81.32% (authored144) and 77.99% (perturbations108). Every row
had zero prompt mismatches. On the missing-evidence check, BF16 and Q8_0 each
select one confident (≥ 0.8) non-insufficient answer; Q4_K_M selects none.

Reading the numbers:

- **Q8_0 is effectively lossless for decisions**: 3 flips in 252 rows, and
  neither accuracy interval excludes zero.
- **Q4_K_M changes about 8% of decisions.** Its authored144 accuracy is higher
  than BF16 with an interval that only just excludes zero; with 11 flips this
  is best read as quantization noise moving borderline rows, not as an
  improvement. On perturbations108 the difference is not significant.
- For comparison, MLX affine 4-bit (`results/mlx/2026-09-17-q4-fixed`) scored
  78.92% and 79.89% on the same suites on Apple Silicon.

`SHA256SUMS` in each run directory covers its files.
