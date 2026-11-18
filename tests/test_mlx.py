"""The MLX sparse layers and depth refiner against the PyTorch ones (which tests/test_sparse.py checks against
dense references). The refiner test needs the checkpoint: PI4_WEIGHTS=/path/to/model.safetensors."""

import os

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from pi4.mlx.sparse import NeighborMap, PixelNeighborMap, SubmanifoldConv3d, sparse_pool  # noqa: E402
from pi4.models import sparse  # noqa: E402


def voxels(seed=0, size=(2, 9, 7, 11), share=0.3):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(size, generator=generator) < share).nonzero().int()


def test_submanifold_conv_matches_pytorch():
    coordinates = voxels()
    features = torch.randn(coordinates.shape[0], 5)
    reference = sparse.SubmanifoldConv3d(5, 4, 3)
    torch.nn.init.normal_(reference.weight)
    torch.nn.init.normal_(reference.bias)
    expected, _ = reference(features, coordinates, None)
    conv = SubmanifoldConv3d(5, 4, 3)
    conv.weight, conv.bias = mx.array(reference.weight.detach().numpy()), mx.array(reference.bias.detach().numpy())
    output, _ = conv(mx.array(features.numpy()), mx.array(coordinates.numpy()))
    np.testing.assert_allclose(np.array(output), expected.detach().numpy(), rtol=1e-4, atol=1e-4)


def test_pixel_neighbor_map_equals_the_searched_one():
    rows, columns = np.meshgrid(np.arange(12), np.arange(20), indexing="ij")
    bins = (np.random.default_rng(0).integers(0, 3, (12, 20)) + rows // 2).astype(np.int32)
    coordinates = np.stack((0 * rows, rows, columns, bins), -1).reshape(-1, 4).astype(np.int32)
    searched = NeighborMap(mx.array(coordinates), 3).rows
    dense = PixelNeighborMap(mx.array(coordinates), mx.array(bins)).rows
    assert np.array_equal(np.array(searched), np.array(dense))


def test_pool_matches_pytorch():
    coordinates = voxels(1)
    features = torch.randn(coordinates.shape[0], 3)
    expected, parents, _, _ = sparse.SparsePool3d()(features, coordinates, None)
    pooled, mlx_parents, _ = sparse_pool(mx.array(features.numpy()), mx.array(coordinates.numpy()))
    order = np.lexsort(np.array(mlx_parents).T[::-1])
    reference_order = np.lexsort(parents.numpy().T[::-1])
    np.testing.assert_array_equal(np.array(mlx_parents)[order], parents.numpy()[reference_order])
    np.testing.assert_allclose(np.array(pooled)[order], expected.numpy()[reference_order], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not os.environ.get("PI4_WEIGHTS"), reason="PI4_WEIGHTS not set")
def test_depth_refiner_matches_pytorch():
    from safetensors.torch import load_file

    from pi4.mlx import Pi4
    from pi4.models.refiner import DepthRefiner

    weights = os.environ["PI4_WEIGHTS"]
    state = {
        key[len("depth_refiner."):]: value
        for key, value in load_file(weights).items()
        if key.startswith("depth_refiner.")
    }
    reference = DepthRefiner()
    reference.load_state_dict(state)
    refiner = Pi4.from_pretrained(weights, dtype=mx.float32).depth_refiner
    torch.manual_seed(0)
    height, width = 96, 160
    rays = torch.randn(1, height, width, 2) * 0.3
    log_depth = torch.log(1 + 3 * torch.linspace(0, 1, height)[:, None].expand(height, width))[None]
    log_depth = log_depth + 0.05 * torch.randn(1, height, width)
    feature = torch.randn(1, height // 16, width // 16, 1026)
    with torch.no_grad():
        expected = reference(rays, log_depth, feature.permute(0, 3, 1, 2)).numpy()
    output = refiner(mx.array(rays.numpy()), mx.array(log_depth.numpy()), mx.array(feature.numpy()))
    np.testing.assert_allclose(np.array(output), expected, rtol=1e-4)
