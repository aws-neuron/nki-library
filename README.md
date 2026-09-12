# NKI Library

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI.
These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKL kernels into your models.
More details can be found in the [NKI Library Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/index.html)

## Kernel Reference

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_cte.py)                         | The kernel implements attention with support for multiple variants and optimizations.                        |
| [Attention KV Parallel Segmented CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_kv_parallel_segmented_cte.py) | The kernel implements KV-parallel segmented prefill attention with online softmax merging for context parallelism. |
| [Attention TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_tkg.py)                         | The kernel implements attention specifically optimized for token generation use cases.                       |
| [MLP Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/mlp/mlp.py)                                                   | The kernel implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations.           |
| [MoE CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_cte/)                       | The kernel implements Mixture of Experts optimized for Context Encoding use cases.              |
| [MoE TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg.py)                                   | The kernel implements Mixture of Experts optimized for Token Generation use cases.                           |
| [Output Projection CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_cte.py) | The kernel computes the output projection operation optimized for Context Encoding use cases.           |
| [Output Projection TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_tkg.py) | The kernel computes the output projection operation optimized for Token Generation use cases.           |
| [QKV Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/qkv/qkv.py)                                                   | The kernel performs Query-Key-Value projection with optional normalization fusion.                                     |
| [RMSNorm-Quant Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_quant.py)                           | The kernel performs optional RMS normalization followed by quantization to `fp8`.                            |
| [RMSNorm MX Prefill Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_mx_prefill.py)                 | The kernel fuses RMSNorm with MX quantization and optional router top-K in token-major `[T, H]` layout for prefill. |
| [RoPE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/embeddings/rope.py)                                          | The kernel applies Rotary Position Embedding to input embeddings with optional LNC sharding.                 |
| [Router Top-K Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/router_topk/router_topk.py)                          | The kernel computes router logits and top-K selection for Mixture of Experts models.                         |
| [Cumsum Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/cumsum/cumsum.py)                                          | The kernel computes cumulative sum along the last dimension.                                                 |

### Experimental Kernels

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention Block TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/attention_block_tkg.py)  | The kernel implements fused attention block for TKG with RMSNorm, QKV, RoPE, and output projection.          |
| [Cross Entropy Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/loss/cross_entropy.py)                      | The kernel implements memory-efficient cross entropy loss forward and backward passes for large vocabularies. |
| [Depthwise Conv1D Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/depthwise_conv1d.py)  | The kernel implements depthwise 1D convolution using implicit GEMM.          |
| [Blockwise MM Backward Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/bwd/blockwise_mm_backward.py) | The kernel implements blockwise matrix multiplication backward pass for dropless Mixture of Experts. |
| [Conv1D Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv1d.py)  | The kernel implements 1D convolution using a filter replication strategy. |
| [Conv3D Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv3d.py)  | The kernel implements 3D convolution using a filter replication strategy. |
| [Conv3D Transpose Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv3d_transpose.py)  | The kernel implements 3D transposed convolution by performing a 3D convolution with input dilation. |
| [Conv3D Temporal Unroll Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/conv/conv3d_temporal_unroll.py) | The kernel implements 3D convolution with temporal unrolling and column tiling for small C_out configurations. |
| [Dynamic Shape Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/dynamic_shapes/) | The kernels dynamic input shapes with dynamic loop tiling on dynamic dimension. |
| [Fine-Grained AllGather Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/fg_allgather.py) | The kernel implements fine-grained ring-based all-gather. |
| [FGCC Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/fgcc.py) | The kernel implements fused all-gather and matrix multiplication (Fine-Grained Gather Collective Compute). |
| [Ring Attention Backward Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/ring_attention_bwd.py) | The kernel implements the backward pass for ring attention using collective permute operations. |
| [RNG Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/rng/rng.py) | The kernels provide GPSIMD engine RNG state management and random number generation. |
| [Transformer TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/transformer_tkg.py) | The kernel implements a transformer forward pass megakernel optimized for token generation (TKG). |
| [Ring Attention Forward Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/ring_attention_fwd.py) | The kernel implements ring attention forward using attention_cte with HBM I/O and online softmax reduction for context parallelism. |
| [Fused Adam Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/optimizer/fused_adam.py) | The kernel implements a fused Adam/AdamW/AMSGrad optimizer step with SPMD tiling and Scalar Engine fusion. |
| [MXFP8 Matmul Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/matmul_mxfp8/matmul_mxfp8_generic_kernel.py) | The kernel implements MXFP8 matrix multiplication with configurable tiling and quantization. |
| [MXFP8 MLP Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/mlp_mxfp8/) | The kernels implement MXFP8 MLP forward and backward passes with recompute support. |
| [MXFP8 MoE Backward Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_mxfp8/bwd/blockwise_mm_backward_mxfp8.py) | The kernel implements blockwise matrix multiplication backward pass for dropless Mixture of Experts using MXFP8 quantized matmuls. |
| [MXFP8 Quantize Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/quantize_mxfp8/quantize_mxfp8.py) | The kernel implements block-wise MXFP8 quantization with scale packing. |
| [Foreach Norm Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/foreach/foreach_norm.py) | The kernels implement L1, L2, and Linf norm computation with SPMD tiling and fused activation-reduce. |
| [Foreach Elementwise Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/foreach/foreach_elementwise.py) | The kernels implement elementwise add, sub, mul, div, addcdiv, addcmul and sqrt operations with SPMD tiling for scalar and tensor operands. |
| [Linear Scan Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/linear_scan.py) | The kernel computes first-order linear recurrence along the last dimension using tensor_tensor_scan. |
| [Selective Scan Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/selective_scan.py) | The kernel implements fused Mamba-style discretization, recurrence, and output projection. |
| [SSD Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/ssd.py) | The kernel implements Mamba-2 chunk-wise parallel SSD computation with TensorE matmuls and VectorE scans, with optional LNC sharding across heads. |
| [Gather Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/misc/gather.py) | The kernel gathers rows from a 2D input tensor based on a 1D index tensor using indirect DMA load. |
| [Scatter-Add Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/misc/scatter_add.py) | The kernel scatter-adds values from a source tensor into a destination tensor using a gather-accumulate-scatter pattern. |
| [NeuroTile](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/neurotile/) | Tile-iterator library that abstracts HBM/SBUF/PSUM tiling, sharding, and access-pattern construction for kernel authors; tutorials and example kernels live under `src/nkilib_src/nkilib/experimental/neurotile/examples/`. |
| [MSDeformableAttention Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deformable_attention/ms_deformable_attention.py)  | The kernel implements multi-scale deformable attention with an indirect DMA transpose strategy.|
| [MSDeformableAttentionBwd Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deformable_attention/ms_deformable_attention.py)  | The kernel implements multi-scale deformable attention backward with an indirect DMA transpose and combined bilinear corner scatter-add strategy.|
| [GpSIMD Top-K Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/topk/gpsimd_topk.py) | The kernel computes top-k over the last dimension using the GpSIMD nisa.topk instruction (bfloat16, gen3+). |
| [MXFP8 Attention TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention_mxfp8/attention_mxfp8_tkg.py) | The kernel implements MXFP8 flash decode attention for token generation. |
| [Sparse Attention Indexer Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/sparse_attention_indexer/sparse_attention_indexer_mx_bf16score.py) | The kernel implements the DeepSeek sparse attention indexer: MX-quantized Q/K/W projections, a BF16 score matmul, and hardware top-K selection of the most relevant KV positions per query. |
| [DeepSeek V3.2 MX MLP Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deepseekv32_mlp/mlp_deepseek_mx.py) | The kernel implements the DeepSeek V3.2 MLP for shared-experts and first dense layers with MX-prequantized packed block-scale input, auto-selecting hoisted or tiled weights with token or intermediate LNC sharding. |
| [DeepSeek V4 CSA Decode Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deepseek_v4_csa/csa_decode_attention.py) | The kernels implement DeepSeek V4 Compressed Sparse Attention for decode, headlined by a megakernel that fuses the lightning-indexer scoring, GpSimd top-K selection and O(window + k) gathered sparse attention into one 2-core launch, so the attention cost is independent of the context length. |
| [DeepSeek V4 CSA Prefill Kernels](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deepseek_v4_csa/csa_prefill_attention.py) | The kernels implement DeepSeek V4 Compressed Sparse Attention for prefill: a fused RMSNorm+RoPE projection tail, the gated-pooling KV compressor, the indexer's bisection top-K selection mask, and mask-predicated sparse attention with a compile-time per-tile causal bound. |
| [DeepSeek V4 CSA Attention Block](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/deepseek_v4_csa/csa_block.py) | The module composes the CSA kernels into complete prefill and decode attention blocks, head-parallel across tensor-parallel ranks with the cross-rank output all-reduce merged into the traced block. |

## Integration with the Neuron Compiler

The Neuron compiler includes a bundled version of this package within `neuronx-cc`, accessible under the `nkilib` Python namespace (for example, `import nkilib`). This bundled version is referred to as "bundled nkilib" throughout this guide. Bundled nkilib has been validated to work with that particular compiler version and can be used out of the box.

If you want to contribute a kernel change or use the latest kernels, you can integrate with this package directly.

> **Note:** Unlike bundled nkilib, **kernels from this package are not guaranteed to be compatible with the latest release of the Neuron compiler**. To start from a known good commit compatible with your compiler version, find the branch corresponding to your compiler version in this repository.

### Installation
1. Install `neuronx-cc` as usual (most likely already done). For more information, see the [Neuron Quick Start
Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/quick-start/index.html).
2. Install this package into the same virtual environment as the rest of your project:
   ```bash
   pip install nki-library
   ```
3. Import and use kernels as usual. This package automatically replaces bundled nkilib kernels with the content of this package. No code changes are required.

### Uninstalling
To uninstall, run the following command:
```bash
pip uninstall nki-library
```

After uninstalling, the compiler falls back to the bundled nkilib.

### Controlling which package gets loaded
To _temporarily_ revert to the bundled version of nkilib, set the `NKILIB_FORCE_BUNDLED_LIBRARY` environment variable to a truthy value:
```bash
export NKILIB_FORCE_BUNDLED_LIBRARY=true
```

On the next execution of neuronx-cc, it will use the bundled version of nkilib. To go back to the kernels from this package, unset `NKILIB_FORCE_BUNDLED_LIBRARY`

```bash
unset NKILIB_FORCE_BUNDLED_LIBRARY
```
