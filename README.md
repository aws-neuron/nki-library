# NKI Library

The NKI Library provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI.
These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKL kernels into your models.
More details can be found in the [NKI Library Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/index.html)

## NOTE

The kernels in this repo require [the Neuron 2.27 release](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.27.0/index.html).

## Kernel Reference

| Kernel API                                                                                                                                                   | Description                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Attention CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/attention/attention_cte.py)                         | The kernel implements attention with support for multiple variants and optimizations.                        |
| [Attention TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/attention/attention_tkg.py)                         | The kernel implements attention specifically optimized for token generation scenarios.                       |
| [MLP Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/mlp/mlp.py)                                                   | The kernel implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations.           |
| [Output Projection CTE Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/output_projection/output_projection_cte.py) | The kernel computes the output projection operation optimized for Context Encoding use cases.           |
| [Output Projection TKG Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/output_projection/output_projection_tkg.py) | The kernel computes the output projection operation optimized for Token Generation use cases.           |                  
| [QKV Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/qkv/qkv.py)                                                   | The kernel performs Query-Key-Value projection with optional normalization fusion.                                     |
| [RMSNorm-Quant Kernel](https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_standalone/nkilib/core/rmsnorm/rmsnorm_quant.py)                           | The kernel performs optional RMS normalization followed by quantization to `fp8`.                            |


