# NKI Library

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI.
These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKL kernels into your models.
More details can be found in the [NKI Library Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/index.html)

## Kernel Reference

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_cte.py)                         | The kernel implements attention with support for multiple variants and optimizations.                        |
| [Attention TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_tkg.py)                         | The kernel implements attention specifically optimized for token generation use cases.                       |
| [MLP Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/mlp/mlp.py)                                                   | The kernel implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations.           |
| [MoE CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_cte/)                       | The kernel implements Mixture of Experts optimized for Context Encoding use cases.              |
| [MoE TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg.py)                                   | The kernel implements Mixture of Experts optimized for Token Generation use cases.                           |
| [Output Projection CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_cte.py) | The kernel computes the output projection operation optimized for Context Encoding use cases.           |
| [Output Projection TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/output_projection/output_projection_tkg.py) | The kernel computes the output projection operation optimized for Token Generation use cases.           |
| [QKV Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/qkv/qkv.py)                                                   | The kernel performs Query-Key-Value projection with optional normalization fusion.                                     |
| [RMSNorm-Quant Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_quant.py)                           | The kernel performs optional RMS normalization followed by quantization to `fp8`.                            |
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
