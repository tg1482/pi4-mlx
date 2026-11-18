"""Portable stand-ins for the three FlexGEMM sparse layers the depth refiner uses.

FlexGEMM's kernels need CUDA. These modules compute the same functions with plain
tensor operations (sorted coordinate keys and binary search), so Pi4 also runs on
CPU and Apple-silicon (MPS) devices. They keep FlexGEMM's parameter names, weight
layout and call signatures, so a Pi4 checkpoint loads into either implementation.

Conventions matched (FlexGEMM ``b2fadb2``):

* ``SubmanifoldConv3d``: ``out[c] = bias + sum_v weight[v] @ in[c + delta_v]`` over
  the coordinates present, with ``weight`` shaped ``(Co, K, K, K, Ci)`` and the
  taps ``delta_v`` in row-major order over the three spatial coordinate columns
  (the first column is the batch index).
* ``SparsePool3d(kernel_size=2, stride=2, reduce="mean")``: every voxel maps to its
  parent ``c // 2``; the parent's features are the mean over its children.
* ``SparseUpsample3d(scale_factor=2, mode="nearest")`` to given output coordinates:
  every output voxel copies its parent at ``c // 2``.
"""

import torch
from torch import nn


def _keys(coordinates, extent):
    """Unique int64 keys for ``(batch, x, y, z)`` rows within ``extent``."""
    coordinates = coordinates.long()
    key = coordinates[:, 0]
    for column in range(1, 4):
        key = key * extent[column] + coordinates[:, column]
    return key


def _extent(*coordinates):
    stacked = torch.cat([c.long() for c in coordinates])
    return (stacked.amax(dim=0) + 3).tolist()


def _lookup(table_keys, table_order, query_keys):
    """Row of each query key in the table, or -1 when absent."""
    position = torch.searchsorted(table_keys, query_keys).clamp_max(
        table_keys.numel() - 1
    )
    found = table_keys[position] == query_keys
    return torch.where(found, table_order[position], torch.full_like(position, -1))


class NeighborMap:
    """For each voxel, the row of each of its kernel neighbors (-1 when absent)."""

    def __init__(self, coordinates, kernel_size):
        self.coordinates = coordinates
        radius = kernel_size // 2
        extent = _extent(coordinates + radius, coordinates)
        shifted = coordinates.long() + radius
        keys = _keys(shifted, extent)
        sorted_keys, order = torch.sort(keys)
        span = torch.arange(-radius, radius + 1, device=coordinates.device)
        deltas = torch.cartesian_prod(span, span, span)
        query = shifted[:, None, :].repeat(1, deltas.shape[0], 1)
        query[:, :, 1:] += deltas[None]
        self.rows = _lookup(
            sorted_keys, order, _keys(query.reshape(-1, 4), extent)
        ).reshape(coordinates.shape[0], deltas.shape[0])


class SubmanifoldConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, algorithm=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, features, coordinates, shape, neighbor_cache=None):
        if neighbor_cache is None or neighbor_cache.coordinates is not coordinates:
            neighbor_cache = NeighborMap(coordinates, self.kernel_size)
        rows = neighbor_cache.rows
        padded = torch.cat((features, features.new_zeros(1, features.shape[1])))
        weight = self.weight.flatten(1, 3).to(features.dtype)  # (Co, taps, Ci)
        output = self.bias.to(features.dtype).expand(features.shape[0], -1)
        for tap in range(weight.shape[1]):  # row -1 reads the zero row
            output = output + padded[rows[:, tap]] @ weight[:, tap].T
        return output, neighbor_cache


class PoolMap:
    """Children (rows at the fine level) grouped under their parents."""

    def __init__(self, coordinates):
        parents = coordinates.clone()
        parents[:, 1:] = torch.div(parents[:, 1:], 2, rounding_mode="floor")
        self.output_coordinates, self.parent = torch.unique(
            parents, dim=0, return_inverse=True
        )
        self.count = torch.bincount(
            self.parent, minlength=self.output_coordinates.shape[0]
        )

    @property
    def T(self):
        return self


class SparsePool3d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, reduce="mean"):
        super().__init__()
        assert kernel_size == stride == 2 and reduce == "mean"

    def forward(self, features, coordinates, shape, output_shape=None):
        mapping = PoolMap(coordinates)
        pooled = features.new_zeros(mapping.output_coordinates.shape[0], features.shape[1])
        pooled.index_add_(0, mapping.parent, features)
        pooled = pooled / mapping.count[:, None].to(features.dtype)
        coordinates = mapping.output_coordinates.to(coordinates.dtype)
        return pooled, coordinates, output_shape, mapping


class SparseUpsample3d(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        assert scale_factor == 2 and mode == "nearest"

    def forward(
        self,
        features,
        coordinates,
        shape,
        output_coords,
        output_shape,
        neighbor_cache=None,
    ):
        if neighbor_cache is not None:
            parent = neighbor_cache.parent  # the pool that made ``coordinates``
        else:
            parents = output_coords.clone()
            parents[:, 1:] = torch.div(parents[:, 1:], 2, rounding_mode="floor")
            extent = _extent(coordinates, parents)
            sorted_keys, order = torch.sort(_keys(coordinates, extent))
            parent = _lookup(sorted_keys, order, _keys(parents, extent))
        return features[parent], output_coords, output_shape, None
