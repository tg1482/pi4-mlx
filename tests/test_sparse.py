"""The portable sparse layers against dense 3-D references on a random voxel set."""

import torch
from torch.nn import functional as F

from pi4.models.sparse import SparsePool3d, SparseUpsample3d, SubmanifoldConv3d


def voxels(seed=0, size=(2, 9, 7, 11), share=0.3):
    generator = torch.Generator().manual_seed(seed)
    occupied = torch.rand(size, generator=generator) < share
    coordinates = occupied.nonzero().int()
    return occupied, coordinates


def dense(features, coordinates, size):
    grid = features.new_zeros(*size, features.shape[1])
    grid[tuple(coordinates.long().T)] = features
    return grid


def test_submanifold_conv_matches_a_dense_masked_conv():
    occupied, coordinates = voxels()
    features = torch.randn(coordinates.shape[0], 5, dtype=torch.float64)
    conv = SubmanifoldConv3d(5, 4, 3).double()
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)
    output, _ = conv(features, coordinates, None)
    grid = dense(features, coordinates, occupied.shape).permute(0, 4, 1, 2, 3)
    weight = conv.weight.permute(0, 4, 1, 2, 3)  # (Co, Ci, K, K, K)
    reference = F.conv3d(grid, weight, conv.bias, padding=1).permute(0, 2, 3, 4, 1)
    torch.testing.assert_close(output, reference[tuple(coordinates.long().T)])


def test_pool_means_children_and_upsample_copies_parents():
    occupied, coordinates = voxels(1)
    features = torch.randn(coordinates.shape[0], 3, dtype=torch.float64)
    pooled, parents, _, cache = SparsePool3d()(features, coordinates, None)
    size = (occupied.shape[0], *((n + 1) // 2 for n in occupied.shape[1:]))
    total = torch.zeros(*size, 3, dtype=torch.float64)
    count = torch.zeros(size, dtype=torch.float64)
    key = (coordinates[:, 0].long(), *(coordinates[:, 1:].long() // 2).T)
    total.index_put_(key, features, accumulate=True)
    count.index_put_(key, torch.ones(len(features), dtype=torch.float64), accumulate=True)
    assert (count > 0).sum() == len(parents)
    torch.testing.assert_close(pooled, (total / count[..., None])[tuple(parents.long().T)])
    for neighbor_cache in (cache.T, None):
        up, _, _, _ = SparseUpsample3d()(pooled, parents, None, coordinates, None, neighbor_cache)
        torch.testing.assert_close(up, (total / count[..., None])[key])
