# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random input generation for MXFP8 matmul tests using various distributions."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from test.utils.rng import NKITestsRNG

_rng = NKITestsRNG()


def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """

    np.random.seed(seed)
    random.seed(seed)
    _rng.reset(seed)


class Distribution(ABC):
    """Base class for all initialization distributions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this distribution."""
        pass

    @abstractmethod
    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        """Initialize the tensor."""
        pass

    @abstractmethod
    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        """Generate random parameters, optionally with edge cases."""
        pass


class KaimingNormalDistribution(Distribution):
    name = "kaiming_normal"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.kaiming_normal_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        return {
            "mode": random.choice(["fan_in", "fan_out"]),
            "nonlinearity": random.choice(["relu", "leaky_relu", "tanh", "sigmoid"]),
        }


class KaimingUniformDistribution(Distribution):
    name = "kaiming_uniform"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.kaiming_uniform_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        return {
            "mode": random.choice(["fan_in", "fan_out"]),
            "nonlinearity": random.choice(["relu", "leaky_relu", "tanh", "sigmoid"]),
        }


class XavierNormalDistribution(Distribution):
    name = "xavier_normal"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.xavier_normal_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        gain = random.choice([1e-4, 1e-2, 100.0, 1000.0]) if edge_case else random.uniform(0.5, 2.0)
        return {"gain": gain}


class XavierUniformDistribution(Distribution):
    name = "xavier_uniform"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.xavier_uniform_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        gain = random.choice([1e-4, 1e-2, 100.0, 1000.0]) if edge_case else random.uniform(0.5, 2.0)
        return {"gain": gain}


class NormalDistribution(Distribution):
    name = "normal"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.normal_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        if edge_case:
            mean = random.choice([0.0, -1e3, 1e3, -1e-6, 1e-6])
            std = random.choice([1e-6, 1e-4, 100.0, 1000.0])
        else:
            mean = random.uniform(-1.0, 1.0)
            std = random.uniform(0.01, 2.0)
        return {"mean": mean, "std": std}


class UniformDistribution(Distribution):
    name = "uniform"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.uniform_(tensor, **params)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        if edge_case:
            a, b = random.choice([(-1e3, 1e3), (-1e-6, 1e-6), (0.0, 1e3), (-1e3, 0.0), (1e2, 1e3)])
        else:
            a, b = random.uniform(-2.0, 0.0), random.uniform(0.0, 2.0)
            if a > b:
                a, b = b, a
        return {"a": a, "b": b}


class AllOnesDistribution(Distribution):
    name = "all_ones"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return torch.nn.init.ones_(tensor)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        return {}


class ExponentialDistribution(Distribution):
    name = "exponential"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.exponential_sample(params["rate"], tensor.shape)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        rate = random.choice([1e-4, 1e-2, 100.0, 1000.0]) if edge_case else random.uniform(0.1, 5.0)
        return {"rate": rate}


class LogNormalDistribution(Distribution):
    name = "log_normal"

    def init_fn(self, tensor: torch.Tensor, **params) -> torch.Tensor:
        return _rng.lognormal_sample(params["loc"], params["scale"], tensor.shape)

    def random_params(self, edge_case: bool = False) -> Dict[str, Any]:
        if edge_case:
            loc = random.choice([-10.0, -5.0, 5.0, 10.0])
            scale = random.choice([1e-3, 0.01, 3.0, 5.0])
        else:
            loc = random.uniform(-1.0, 1.0)
            scale = random.uniform(0.1, 2.0)
        return {"loc": loc, "scale": scale}


class DistributionRegistry:
    """Registry for all available distributions."""

    _distributions: Dict[str, Distribution] = {}

    @classmethod
    def register(cls, dist: Distribution) -> None:
        cls._distributions[dist.name] = dist

    @classmethod
    def get(cls, name: str) -> Distribution:
        if name not in cls._distributions:
            raise KeyError(f"Unknown distribution: {name}")
        return cls._distributions[name]

    @classmethod
    def all_names(cls) -> List[str]:
        return list(cls._distributions.keys())

    @classmethod
    def all_distributions(cls) -> List[Distribution]:
        return list(cls._distributions.values())


# Register all distributions
DistributionRegistry.register(KaimingNormalDistribution())
DistributionRegistry.register(KaimingUniformDistribution())
DistributionRegistry.register(XavierNormalDistribution())
DistributionRegistry.register(XavierUniformDistribution())
DistributionRegistry.register(NormalDistribution())
DistributionRegistry.register(UniformDistribution())
DistributionRegistry.register(AllOnesDistribution())
DistributionRegistry.register(ExponentialDistribution())
DistributionRegistry.register(LogNormalDistribution())


def get_random_distributions(
    num: int, dist_names: Optional[List[str]] = None, edge_rate: float = 0.3, seed: Optional[int] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns a list of (dist_name, params) tuples for testing matmul kernels.

    Distributions are picked uniformly at random from the registry, and parameters
    are generated randomly while ensuring edge cases are covered.

    Args:
        num: Number of distribution tuples to return
        dist_names: List of dists to select from
        edge_rage: Rate of edge cases in generated distributions
        seed: Optional random seed for reproducibility

    Returns:
        List of (dist_name, params) tuples where dist_name is a registered distribution
        and params is a dict that can be passed as **kwargs to the init function
    """
    if seed is not None:
        set_seed(seed)

    if dist_names is None:
        dist_names = DistributionRegistry.all_names()
    distributions: List[Tuple[str, Dict[str, Any]]] = []

    for _ in range(num):
        dist_name = random.choice(dist_names)
        dist = DistributionRegistry.get(dist_name)
        edge_case = random.random() < edge_rate
        params = dist.random_params(edge_case=edge_case)
        distributions.append((dist_name, params))

    return distributions


def get_random_inputs(config: Dict[str, Any], seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random LHS, RHS given a generation config.

    Args:
        config: Each config has five expected fields:
            - shapes: a 3-tuple (m, k, n) specifying the shapes of the matrices.
              lhs: (m, k), rhs: (k, n)
            - dists: a list of two strings specifying the initialization distribution
              of the two matrices. These strings correspond to registered distribution names.
            - params: a list of two dicts specifying the initialization parameters.
              The parameters are passed into the distributions to initialize the matrices.
              If the initialization method requires no parameters, two empty dicts should be passed in.
            - stride: the interleave_load stride used in the test. Important for GPU threshold generation
            - elem_dtype: the element type used in mxfp8 quantization.
              Could be "float8_e4m3fn" or "float8_e5m2"
        seed: a random seed

    Returns:
        LHS, RHS: torch tensors in FP32
    """
    m, k, n = config["shapes"]
    dists = config["dists"]
    params = config["params"]

    set_seed(seed)

    lhs_dist = DistributionRegistry.get(dists[0])
    rhs_dist = DistributionRegistry.get(dists[1])

    lhs = lhs_dist.init_fn(torch.empty((m, k)), **params[0])
    rhs = rhs_dist.init_fn(torch.empty((k, n)), **params[1])

    return (lhs, rhs)
