# Hilbert

## Pruned & Quantized 4B Decision Model · CUDA Backend · Alloy Governance

**Hilbert** is an experimental pruned 4B semantic-decision model and systems runtime derived from the decision-native architecture demonstrated by SemIf.

The project focuses on a narrow execution path:

```text
state + question + declared options
              │
              ▼
       Hilbert 4B Model
              │
              ▼
       typed option scores
              │
       ┌──────┴──────┐
       ▼             ▼
    CUDA          Governance
   Backend          Layer
       │             │
       ▼             ▼
   Kernels         Alloy
       │          constraints
       └──────┬──────┘
              ▼
        Auditable Result
```

> **Status:** Experimental research implementation.
> Model pruning, distillation, CUDA kernels, and Alloy governance specifications must be independently benchmarked and verified before production claims are made.

---

## 1. What Hilbert Is

Hilbert treats many AI workloads as **typed decisions rather than text-generation problems**.

Instead of asking a model to generate:

```text
"The request should be routed to account-access support because..."
```

Hilbert evaluates declared alternatives directly:

```json
{
  "question": "Which queue should handle this request?",
  "options": [
    {"id": "access", "description": "Account access support."},
    {"id": "billing", "description": "Billing support."}
  ]
}
```

The resulting computation is conceptually:

```text
P(access | state, question, options)
P(billing | state, question, options)
```


# 2. Hilbert 4B

Hilbert targets a **pruned/distilled 4B-class model**.

The intended transformation is:

```text
Reference semantic model
          │
          ▼
   distillation data
          │
          ▼
   structured decisions
          │
          ▼
       pruning
          │
          ▼
    Hilbert 4B
          │
          ▼
    quantization
          │
          ▼
   CUDA execution
```



---

# 3. Quantization

Hilbert is served **quantized**. The pruned and distilled 4B model is the
thing being measured; quantization is how it fits and runs on consumer GPUs
(the reference target is an RTX 3080, sm_86). A quantized model is judged by
**decision agreement with its BF16 reference**, not by perplexity. A format
that flips decisions is a correctness issue (§13), however much memory it
saves.

### What exists in this repository

Status uses the ladder of §19: PROPOSED → IMPLEMENTED → BENCHMARKED →
INDEPENDENTLY VERIFIED.

| Piece | Where | Status | Evidence |
|---|---|---|---|
| GGUF loading, 13 weight formats (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K; IQ formats refused) | [`cleanroom-transformer/src/gguf.c`](cleanroom-transformer/src/gguf.c), [`ggml_quant.h`](cleanroom-transformer/src/ggml_quant.h) | IMPLEMENTED, tested | Every format dequantizes bit-exactly against llama.cpp's reference. Tiny Llama GGUFs match PyTorch and transformers' GGUF loader within 1e-6 relative ([engine README](cleanroom-transformer/README.md#gguf-files)) |
| Quantized weights resident on the GPU | the same engine, `--gpu-weights quantized` (default) | IMPLEMENTED | Weights stay in GGUF block format and are decoded inside the CUDA kernels: an 8B Q4_K_M needs about 5 GB of weights instead of about 17 GB as BF16. `selftest` checks every quantized-weight kernel against the host decoder. It was reported passing on an RTX 3080; no log is committed |
| Tensor inspection | `cleanroom-transformer dequant --tensor NAME` | IMPLEMENTED | Prints any GGUF tensor as float32 rows, for checking a quantized file by hand |
| Quantized models in the browser | [`webgpu-demo/`](webgpu-demo/README.md) | IMPLEMENTED | Qwen3-0.6B Q8_0, MiniCPM5-2B Q4_K_M and Qwen3.5-4B Q4_K_M, all pinned by revision. The scores the demo shows come from the BF16 checkpoints, not from these quantized files |
| 27B at 5.0 bits per weight (exl3) | [`exl3-bridge/`](exl3-bridge/README.md) | BENCHMARKED | Committed row-level results with SHA256SUMS: `authored144` balanced accuracy 0.9579 (pinned 4B BF16: 0.813); `shape777` argmax agreement 0.8443 with the 4B BF16 predictions (121 flips). Model family and quantization both differ, so this is **not** a quantization ablation |
| Quantized Hilbert 4B | — | PROPOSED | The engine's GGUF loader reads Llama-architecture files only. The Qwen3.5 hybrid layers (Gated DeltaNet and gated attention) need loader and kernel support before a quantized Hilbert 4B can run through the CUDA backend |
| Full 8B GGUF run end to end | — | PROPOSED | Tokenizer and prompts of a released Llama 3 8B Instruct Q4_K_M match Hugging Face; the full forward pass on that file has not been run |
| Prune-then-quantize helpers (PyTorch, INT8) | [sovereign-engine-v2](https://github.com/SNAPKITTYWEST/sovereign-engine-v2) (external) | IMPLEMENTED, does not run as written | See below |

### The pruning and quantization tools: sovereign-engine-v2

The prune-and-quantize tooling lives in a separate repository,
[SNAPKITTYWEST/sovereign-engine-v2](https://github.com/SNAPKITTYWEST/sovereign-engine-v2),
read at commit
[`d4bdfd2`](https://github.com/SNAPKITTYWEST/sovereign-engine-v2/commit/d4bdfd2a136ece421489813f309f89b363a38c42).
The relevant code is `ModelPruner` in
[`src/models/checkpoint_manager.py`](https://github.com/SNAPKITTYWEST/sovereign-engine-v2/blob/d4bdfd2a136ece421489813f309f89b363a38c42/src/models/checkpoint_manager.py)
and the save → prune → quantize → upload sequence in
[`src/models/checkpoint_workflow.py`](https://github.com/SNAPKITTYWEST/sovereign-engine-v2/blob/d4bdfd2a136ece421489813f309f89b363a38c42/src/models/checkpoint_workflow.py).

- **Pruning:** L1 unstructured (`torch.nn.utils.prune.l1_unstructured`) on
  every `Linear` and `Conv2d` weight, or structured pruning of whole output
  channels by L2 norm (`ln_structured`, `n=2`, `dim=0`).
- **Quantization:** PyTorch post-training *dynamic* quantization
  (`torch.quantization.quantize_dynamic`, `qint8`).
- **Checkpoints:** each one gets a SHA-256 seal written to `audit.jsonl`. A
  caller that passes the previous seal chains them, and the workflow chains
  the pruned checkpoint to its source.



