"""MLX versions of the sparse 3-D layers the depth refiner uses (see ``pi4.models.sparse``).

Voxels are rows of ``(batch, x, y, z)`` int coordinates. Neighbors are found by
binary search over sorted int64 coordinate keys, so everything runs on the GPU.
"""

import math

import mlx.core as mx
from mlx import nn


def _keys(coordinates, extent):
    coordinates = coordinates.astype(mx.int64)
    key = coordinates[:, 0]
    for column in range(1, 4):
        key = key * extent[column] + coordinates[:, column]
    return key


def _extent(*coordinates):
    stacked = mx.concatenate([c.astype(mx.int64) for c in coordinates])
    return (stacked.max(axis=0) + 3).tolist()


def _lower_bound(sorted_keys, queries):
    """Position of the first key >= each query (binary search, vectorized)."""
    count = sorted_keys.shape[0]
    low = mx.zeros(queries.shape, mx.int32)
    high = mx.full(queries.shape, count, mx.int32)
    for _ in range(math.ceil(math.log2(count + 1))):
        middle = (low + high) // 2
        below = sorted_keys[mx.minimum(middle, count - 1)] < queries
        low, high = mx.where(below, middle + 1, low), mx.where(below, high, middle)
    return low


class NeighborMap:
    """Row of each voxel's neighbor at every 3x3x3 tap (row-major taps), or -1 when absent."""

    def __init__(self, coordinates, kernel_size):
        self.coordinates = coordinates
        radius = kernel_size // 2
        shifted = coordinates.astype(mx.int64) + radius
        extent = _extent(shifted + radius)
        keys = _keys(shifted, extent)
        order = mx.argsort(keys).astype(mx.int32)
        sorted_keys = keys[order]
        padded_keys = mx.concatenate((sorted_keys, mx.full((kernel_size,), -1, mx.int64)))
        # The last coordinate is the key's lowest digit, so the taps along it are
        # consecutive keys: one search per (x, y) offset finds where they start,
        # and tap ``step`` is within ``step`` places of that.
        span = range(-radius, radius + 1)
        deltas = mx.array([(0, x, y, -radius) for x in span for y in span], mx.int64)
        query = _keys((shifted[:, None, :] + deltas[None]).reshape(-1, 4), extent)
        position = _lower_bound(sorted_keys, query)
        last = order.shape[0] - 1
        taps = []
        for step in range(kernel_size):
            row = mx.full(query.shape, -1, mx.int32)
            for offset in range(step + 1):
                found = padded_keys[position + offset] == query + step
                row = mx.where(found, order[mx.minimum(position + offset, last)], row)
            taps.append(row)
        self.rows = mx.stack(taps, axis=-1).reshape(coordinates.shape[0], kernel_size**3)


class PixelNeighborMap(NeighborMap):
    """The same map when each pixel holds exactly one voxel ``(0, row, column, bins[row, column])``,
    as at the refiner's full resolution: a neighbor exists where the neighboring pixel's bin
    differs by the tap's bin offset, so no search is needed."""

    def __init__(self, coordinates, bins):
        self.coordinates = coordinates
        height, width = bins.shape
        index = mx.arange(height * width, dtype=mx.int32).reshape(height, width)
        pad = [(1, 1), (1, 1)]
        padded_bins = mx.pad(bins, pad, constant_values=-(2**30))
        padded_index = mx.pad(index, pad, constant_values=-1)
        taps = []
        for row in range(3):
            for column in range(3):
                neighbor_bin = padded_bins[row : row + height, column : column + width]
                neighbor = padded_index[row : row + height, column : column + width]
                for step in (-1, 0, 1):
                    taps.append(mx.where(neighbor_bin - bins == step, neighbor, -1))
        self.rows = mx.stack(taps, axis=-1).reshape(height * width, 27)


class SubmanifoldConv3d(nn.Module):
    def __init__(self, inputs, outputs, kernel_size):
        super().__init__()
        self.weight = mx.zeros((outputs, kernel_size, kernel_size, kernel_size, inputs))
        self.bias = mx.zeros((outputs,))

    def __call__(self, features, coordinates, cache=None):
        if cache is None or cache.coordinates is not coordinates:
            cache = NeighborMap(coordinates, self.weight.shape[1])
        padded = mx.concatenate((features, mx.zeros((1, features.shape[1]), features.dtype)))
        rows = mx.where(cache.rows < 0, features.shape[0], cache.rows)  # absent: the zero row
        weight = self.weight.reshape(self.weight.shape[0], -1).astype(features.dtype)
        gathered = padded[rows].reshape(rows.shape[0], -1)  # (N, taps * Ci)
        output = mx.addmm(self.bias.astype(features.dtype), gathered, weight.T)
        return output, cache


class PoolMap:
    """Each voxel's parent at half resolution, and the parents' coordinates and child counts."""

    def __init__(self, coordinates):
        parents = mx.concatenate((coordinates[:, :1], coordinates[:, 1:] // 2), axis=1)
        keys = _keys(parents, _extent(parents))
        order = mx.argsort(keys)
        sorted_keys = keys[order]
        new = mx.concatenate((mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]))
        group = mx.cumsum(new.astype(mx.int32)) - 1
        count = int(group[-1].item()) + 1
        self.parent = mx.zeros(keys.shape, mx.int32).at[order].add(group)
        self.coordinates = mx.zeros((count, 4), coordinates.dtype).at[group].maximum(parents[order])
        self.count = mx.zeros((count,), mx.float32).at[self.parent].add(1)


def sparse_pool(features, coordinates):
    mapping = PoolMap(coordinates)
    pooled = mx.zeros((mapping.count.shape[0], features.shape[1]), features.dtype)
    pooled = pooled.at[mapping.parent].add(features) / mapping.count[:, None].astype(features.dtype)
    return pooled, mapping.coordinates, mapping
