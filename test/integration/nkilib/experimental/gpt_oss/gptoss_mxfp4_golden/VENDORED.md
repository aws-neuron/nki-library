# Vendored: GPT-OSS MXFP4 decode golden

These files are a **verbatim copy** of the standalone pure-CPU PyTorch golden from
the sibling package `private-vllm-neuron/gptoss_mxfp4_golden`. They are
vendored here so the `gpt_oss_mxfp4_decode_layer` kernel test is **self-contained**
— it needs no cross-package `sys.path` bridge and no `private-vllm-neuron` checkout.

The golden constructs a `GptOssDecoderLayer` and runs its `forward_decode`, so it
is the reference oracle the NKI kernel is diffed against. It is pure `torch` — no
`nki`, `vllm`, or `transformers` imports on the decode-layer path.

## What was copied (and what was not)

Copied (torch-only closure reached by `layer_entry.gptoss_mxfp4_decode_layer`):
`__init__.py`, `config.py`, `loader.py`, `paged_kv.py`, `kernels.py`, `model.py`,
`layer_entry.py`, `golden_entry.py`, `README.md`.

**Not** copied:
- `hf_reference.py` — pulls in `transformers` (only used by the golden's own
  cross-check tests, not by the kernel oracle).
- `test_golden.py` — the golden's own pytest suite; excluded so pytest does not
  collect it here and so we don't drag in `transformers` / a real checkpoint.

`safetensors` and `transformers` remain lazy (function-local) imports inside
`loader.py` / `golden_entry.py` and are never reached by the single-layer oracle.

## Re-syncing

Upstream source of truth:
`private-vllm-neuron/gptoss_mxfp4_golden/`

To refresh after the golden changes upstream, re-copy the files listed above
verbatim (keep them byte-identical to upstream to make diffs trivial). Do not
hand-edit the vendored copies.
