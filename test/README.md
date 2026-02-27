# NKI Library Test Framework

Documentation and tools for testing NKI Library kernels.
## Set up 

```bash
# 1. Initialize Python virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install nki-library + all dependencies
make install

# 4. Download the Neuron compiler, Neuron Kernel Interface (NKI) wheels into `wheelhouse` directory

# 5. Install the compiler and NKI wheels
make install_wheelhouse

# 6. Now you can run regular make targets
make test 
```

## Directory Structure

```
test/
├── integration/nkilib/core/        # Core kernel tests 
│   ├── attention/                  # Attention kernel tests
│   ├── mlp/                        # MLP kernel tests
│   ├── moe/                        # Mixture of Experts tests
│   ├── qkv/                        # QKV projection tests
│   ├── rmsnorm/                    # RMSNorm-Quant tests
│   ├── output_projection/          # Output projection tests
│   ├── router_topk/                # Router Top-K tests
│   ├── cumsum/                     # Cumsum tests
│   ├── embeddings/                 # RoPE embedding tests
│   └── ...
├── unit/                           # Unit tests for framework utilities
├── utils/                          # Test framework utilities
└── docs/                           # Test documentation
```

**Test location convention:** Tests mirror source structure.
- Source: `src/nkilib_src/nkilib/core/mlp/`
- Tests: `test/integration/nkilib/core/mlp/`

## Command Line Interface

### Running Tests

```bash
# Basic test execution
make test ARGS="-k 'test_name' --platform-target trn2 --output-directory <directory_path> --target-host <hostname>"

# Run specific test file
make test ARGS="test/integration/nkilib/core/mlp/test_mlp_tkg.py --platform-target trn2 --output-directory <directory_path>"
```

### Test Mode Options

| Mode | Description | Use Case |
|------|-------------|----------|
| `compile-only` | Compile kernel without running on hardware | Quick syntax/structure validation |
| `compile-and-infer` | Compile and run inference on hardware | Full validation (default) |

```bash
# Compile-only mode (no hardware required)
make test ARGS="test/integration --test-mode compile-only -k 'test_name'"

# Compile and run inference
# no test-mode flag needed, default mode
make test ARGS="test/integration -k 'test_name' --target-host <hostname>"
```

### CLI Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--test-mode` | Test execution mode: `compile-only`, `compile-and-infer` | `compile-and-infer` |
| `--output-directory` | Base directory for test artifacts | `neuron_test_output` |
| `--debug-kernels` | Dump additional debug output for kernel debugging | `False` |
| `--force-local-cleanup` | Auto-cleanup test output directory after tests | `False` |
| `--metric-output` | Enable metrics collection: `file`, `stdout`, `stderr` | `file` |
| `--coverage` | Default coverage strategy: `singles`, `pairs`, `full` | `singles` |
| `--validation-histograms` | Dump validation reports with histograms | `False` |
| `--platform-target` | Target platform: `trn2`, `trn3`, etc. | `trn2` |
| `-k "pattern"` | Run tests matching pattern | All tests |
| `-m "marker"` | Run tests with specific marker (e.g., `-m fast`) | All tests |

### Remote Execution Options

For running tests on remote Trainium/Inferentia hosts:

| Option | Description |
|--------|-------------|
| `--target-host <hostname>` | Remote host(s) for test execution |
| `--target-host-file <path>` | JSON file with host definitions |
| `--ssh-config-path <path>` | SSH config file path (default: `~/.ssh/config`) |
| `--skip-remote-cleanup` | Skip cleanup of remote directories after test execution (useful for debugging) |

### S3 Artifact Options

For large file transfers via S3:

> **Note:** You must have valid AWS credentials configured. Either export `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or use `--aws-profile` to specify a configured profile in `~/.aws/credentials`.

| Option | Description |
|--------|-------------|
| `--artifact-upload-s3-bucket` | S3 bucket for artifact transfer |
| `--artifact-upload-s3-prefix` | S3 prefix for artifacts (default: `artifacts_tmp`) |
| `--aws-profile` | AWS profile for S3 authentication |
| `--upload-test-outcomes` | Upload test outcomes to S3: `all`, `passed`, `failed` (required for test output upload) |
| `--test-output-s3-bucket` | S3 bucket for test output upload |
| `--test-output-s3-prefix` | S3 prefix for test outputs |

## Writing Tests

### Key APIs

| API | Description |
|-----|-------------|
| `@pytest.mark.coverage_parametrize` | Intelligent test case generation with coverage strategies |
| `KernelArgs` | Main test config: `kernel_func`, `compiler_input`, `kernel_input`, `validation_args`, `inference_args` |
| `CompilerArgs` | Compiler config: `logical_nc_config`, `platform_target`, `additional_cmd_args` |
| `InferenceArgs` | Inference config: `collective_ranks`, `enable_determinism_check`, `num_runs`, `profile_all_runs` |
| `ValidationArgs` | Golden validation: `golden_output`, `relative_accuracy`, `absolute_accuracy` |
| `LazyGoldenGenerator` | Defers golden computation: `output_ndarray` (shape/dtype), `lazy_golden_generator` (callable) |
| `PerRankLazyInputGenerator` | Per-rank input generation for collectives: `generator_func(rank_id) -> dict` |
| `PerRankLazyGoldenGenerator` | Per-rank golden generation for collectives: `generator_func(rank_id) -> dict` |
| `Orchestrator` | Test executor, provided via `test_manager` fixture |

### Coverage Parametrize

The `@pytest.mark.coverage_parametrize` decorator provides intelligent test case generation:

```python
@pytest.mark.coverage_parametrize(
    batch_size=[1, 8, 32],
    hidden_size=[256, 768, 1024],
    dtype=[np.float32, ml_dtypes.bfloat16],
    filter=filter_invalid_combinations,  # Optional: skip invalid combos
    coverage="pairs"  # Coverage strategy
)
def test_my_kernel(test_manager, batch_size, hidden_size, dtype, is_negative_test_case):
    ...
```

**Coverage strategies:**
| Strategy | Description | Test Count |
|----------|-------------|------------|
| `"singles"` | Each parameter value appears at least once | Minimal |
| `"pairs"` | All 2-way parameter combinations covered | Moderate |
| `"full"` | Complete cartesian product | Maximum |

For detailed documentation, see [`docs/coverage_parametrize.md`](docs/coverage_parametrize.md).

### Filter Functions

Filter functions classify parameter combinations:

```python
from test.utils.coverage_parametrized_tests import FilterResult

def filter_invalid_combinations(batch_size, seq_len, dtype=None):
    # Invalid: exceeds memory limits
    if batch_size * seq_len > 65536:
        return FilterResult.INVALID

    # Redundant: valid but not worth testing
    if batch_size == 1 and seq_len == 1:
        return FilterResult.REDUNDANT

    return FilterResult.VALID
```

| FilterResult | Description | Positive Tests | Negative Tests |
|--------------|-------------|----------------|----------------|
| `VALID` | Valid combination | ✓ Included | Not included |
| `INVALID` | Invalid (kernel should reject) | Not included | ✓ Included |
| `REDUNDANT` | Valid but redundant | Not included | Not included |

### Input Generation

Use tensor generators from `test.integration.nkilib.utils.tensor_generators` for consistent input creation. **Always set random seed** (`np.random.seed(42)`) for deterministic tests when using random generation.

### Minimal Test Example

```python
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.coverage_parametrized_tests import assert_negative_test_case
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from typing import final
import numpy as np
import pytest

@pytest_test_metadata(name="MyKernel", pytest_marks=["my_kernel"])
@final
class TestMyKernel:
    @pytest.mark.fast  # Mark for fast test selection
    @pytest.mark.coverage_parametrize(
        batch=[1, 64, 128],
        hidden=[256, 1024, 2048],
        dtype=[np.float32],
        coverage="singles",
    )
    def test_my_kernel_fast(self, test_manager: Orchestrator, batch, hidden, dtype, is_negative_test_case):
        inputs = {"x": gaussian_tensor_generator()(name="x", shape=(batch, hidden), dtype=dtype)}
        with assert_negative_test_case(is_negative_test_case):
            test_manager.execute(KernelArgs(
                kernel_func=my_kernel,
                compiler_input=CompilerArgs(),
                kernel_input=inputs,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        output_ndarray={"y": np.zeros((batch,), dtype=dtype)},
                        lazy_golden_generator=lambda: {"y": np.sum(inputs["x"], axis=-1)},
                    ),
                    absolute_accuracy=1e-4,
                ),
            ))
```

For complete examples, see [`test/integration/nkilib/core/cumsum/test_cumsum.py`](integration/nkilib/core/cumsum/test_cumsum.py).

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.fast` | Fast tests with curated test vectors (for compile-only validation) |
| `@pytest.mark.skip_compilation` | Skip compilation phase |

## Performance Metrics Collection

Collect performance metrics with `--metric-output`:

```bash
# Collect metrics to files
make test ARGS="--metric-output file -k test_name"

# View collected metrics (stored in test artifact directories)
find neuron_test_output -name "*.json" -path "*/metrics/*" | head -5

# View collected QoR data (CSV format)
cat neuron_test_output/qor_data_*.csv

# Extract specific QoR values from JSON
for f in $(find . -path '*/metrics/*.json'); do
  jq -r '[.TestName, .TpbSgCyclesSum] | @tsv' "$f"
done
```

The CSV contains: `TestName`, `TpbSgCyclesSum` (cycles), `MbuEstimatedPercent` (memory bandwidth utilization), `ProfilerMFU`, `InferenceTime`.

**Note:** Do not use `--force-local-cleanup` when collecting metrics, as metrics are stored in the local artifact directory.

### Tips

- **Always rebuild before testing** - Tests run against built artifacts
- **Redirect output for analysis:** `brazil-build integration-test ... > /tmp/test.txt 2>&1`
- **Use parallelism** (`-n auto --dist worksteal`) - Always use unless <20 test configs
- **Use timeouts** for long runs: `timeout 300 brazil-build integration-test ...`
