# llama.cpp Kepler (K80) — nc-Dequantize for K-Quants

Fixes the MUL_MAT slowdown on Kepler GPUs (Compute Capability < 5.0, no DP4A) when running K-quant models (q2_K … q6_K).

## The problem

On Kepler (e.g. Tesla K80, CC 3.7), non-contiguous K-quants were routed to the slow MMQ path — or offloaded to the CPU — because the non-contiguous dequantize converter ("nc-converter") was missing for K-quants. MMQ on Kepler is ~4× slower than cuBLAS.

## The solution

Three components:

1. **cont-cpy** — block-walk in whole blocks (`ggml_type_size` instead of `ggml_element_size`), so `ggml_cont(q4_K)` works.
2. **nc-Dequantize (fp32)** — a stride-enabled dequantize kernel for q2/q3/q4/q5/q6_K, registered in `ggml_get_to_fp32_nc_cuda`.
3. **nc-Dequantize (fp16/bf16)** — the same kernel additionally registered for fp16/bf16 output.

Plus a routing fix: `ggml_cuda_should_use_mmq` now only routes to MMQ for non-contiguous src0 without an nc-converter; contiguous K-quants stay on cuBLAS.

## Measured results (Tesla K80, cool die, 175 W)

- Prefill: **42.8 → ~140 t/s** (~3.3×) — back to the cuBLAS baseline.
- non-contiguous ≈ cuBLAS (1.01–1.05×); GPU ~2× faster than CPU at every batch size.
- Correctness: NMSE exactly **0** for all five K-quants (fp16 + bf16).
- Regression: K80 (CUDA2) 19208/19208 · RTX 3090 (CUDA0) 19268/19268 — no DP4A regression.

## Origin

Developed through careful measurement by one individual and two agents. At no point were any technical objections raised. However, the original upstream PR ([#30338](https://github.com/ggml-org/llama.cpp/pull/30338)) was closed by the maintainer due to procedural concerns — further evidence-based adjustments, which served to improve the patch's functionality, were submitted after the initial approval of the original changes. This repository keeps the patch publicly available.

## Files

- `patch-llama-cuda-kepler-cont-cpy.patch`
- `patch-llama-cuda-kepler-nc-dequant-kquants.patch`
