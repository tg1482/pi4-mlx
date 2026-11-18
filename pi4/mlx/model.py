"""Pi4 geometry (points, confidence, cameras, metric scale) in MLX.

A port of ``pi4.models.Pi4`` for Apple silicon. It loads the same checkpoint and
returns the same geometry outputs; tracking and motion masks (``query_frame``) are
not ported.
"""

import math
from pathlib import Path

import mlx.core as mx
from mlx import nn

from .layers import (
    Block,
    Conv2d,
    ConvPyramid,
    ConvResidual,
    CrossBlock,
    feature_pyramid,
    patch_positions,
    sequence,
    upsample_block,
)
from .sparse import PixelNeighborMap, SubmanifoldConv3d, sparse_pool


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = Conv2d(3, 1024, 16, stride=16)
        self.prefix_tokens = mx.zeros((1, 5, 1024))
        self.blocks = [Block(1024, 16, epsilon=1e-5, scaled=True) for _ in range(18)]
        self.norm = nn.LayerNorm(1024, eps=1e-5)
        self.projection = nn.Linear(2048, 1024)
        self.periods = 100 ** (mx.arange(16).astype(mx.float32) / 16)

    def __call__(self, images):
        features = self.patch_embed(images)
        batch_size, height, width, _ = features.shape
        features = mx.concatenate(
            (
                mx.broadcast_to(self.prefix_tokens, (batch_size, 5, 1024)).astype(features.dtype),
                features.reshape(batch_size, height * width, -1),
            ),
            axis=1,
        )
        rows, columns = mx.meshgrid(
            mx.arange(height).astype(mx.float32) + 0.5,
            mx.arange(width).astype(mx.float32) + 0.5,
            indexing="ij",
        )
        coordinates = mx.stack((rows, columns), axis=-1).reshape(-1, 2) / max(height, width) * 2 - 1
        angles = (2 * math.pi * coordinates[:, :, None] / self.periods.astype(mx.float32)).reshape(
            coordinates.shape[0], -1
        )
        angles = mx.concatenate((angles, angles), axis=-1)
        rotary = mx.sin(angles), mx.cos(angles)
        intermediate = []
        for index, block in enumerate(self.blocks):
            features = block(features, rotary=rotary)
            if index in (11, 17):
                intermediate.append(self.norm(features)[:, 5:])
        return self.projection(mx.concatenate(intermediate, axis=-1))


class TimeGates(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.encoding = mx.zeros((steps, 128))
        self.proj = [nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3072)]

    def __call__(self, step, batch_size):
        gates = sequence(self.proj, self.encoding[step][None]) + 1
        gates = mx.broadcast_to(gates, (batch_size, 3072))
        return gates.reshape(batch_size, 3, 1024).transpose(1, 0, 2)[:, :, None]


class GeometryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_tokens = mx.zeros((1, 1, 5, 1024))
        self.frame_block = Block(1024, 16, normalize=True, scaled=True)
        self.global_block = Block(1024, 16, normalize=True, scaled=True)
        self.readout = Block(1024, 16, normalize=True, scaled=True)
        self.frame_gates = TimeGates(16)
        self.global_gates = TimeGates(16)

    def __call__(self, tokens, positions, batch_size, frames):
        registers = mx.broadcast_to(self.register_tokens, (batch_size, frames, 5, 1024))
        features = mx.concatenate(
            (registers.reshape(batch_size * frames, 5, 1024).astype(tokens.dtype), tokens), axis=1
        )
        token_count = features.shape[1]
        for index in range(16):
            features = self.frame_block(
                features.reshape(batch_size * frames, token_count, 1024),
                positions,
                gates=self.frame_gates(index, batch_size * frames),
            )
            features = self.global_block(
                features.reshape(batch_size, frames * token_count, 1024),
                gates=self.global_gates(index, batch_size),
            )
        return self.readout(features.reshape(batch_size * frames, token_count, 1024), positions)


class ConfidenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        dimensions = (256, 128, 64, 32)
        self.input_blocks = [Conv2d(2, channels, 1) for channels in dimensions]
        self.res_blocks = [[ConvResidual(channels)] for channels in dimensions]
        self.resamplers = [
            upsample_block(first, second) for first, second in zip((512, *dimensions[:-1]), dimensions)
        ]
        self.output = Conv2d(32, 1, 1)

    def __call__(self, prefix, pyramid):
        features = prefix
        for index in range(4):
            features = sequence(self.resamplers[index], features) + self.input_blocks[index](
                pyramid[index + 1]
            )
            features = self.res_blocks[index][0](features)
        return self.output(features)


class PointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = ConvPyramid((512, 256, 128, 64, 32), 3)
        self.confidence = ConfidenceHead()

    def __call__(self, tokens, height, width):
        pyramid = feature_pyramid(tokens.astype(mx.float32), height, width, 5)
        coordinates, prefix = self.points(pyramid)
        logits = self.confidence(prefix, pyramid)[..., 0]
        return coordinates[..., :2], coordinates[..., 2], logits, pyramid[0]


class CameraResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_conv1 = nn.Linear(512, 512)
        self.res_conv2 = nn.Linear(512, 512)
        self.res_conv3 = nn.Linear(512, 512)

    def __call__(self, features):
        residual = nn.relu(self.res_conv2(nn.relu(self.res_conv1(features))))
        return features + nn.relu(self.res_conv3(residual))


def _determinant(matrices):
    a, b, c = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    d, e, f = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    g, h, i = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class CameraHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projects = nn.Linear(1024, 768)
        self.blocks = [Block(768, 12) for _ in range(2)]
        self.linear_out = nn.Linear(768, 512)
        self.res_conv = [CameraResidual() for _ in range(2)]
        self.more_mlps = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()]
        self.pose = nn.Linear(512, 12)

    def __call__(self, tokens, positions):
        features = self.projects(tokens.astype(mx.float32))
        for block in self.blocks:
            features = block(features, positions)
        features = self.linear_out(features)[:, 5:]
        for block in self.res_conv:
            features = block(features)
        raw_pose = self.pose(sequence(self.more_mlps, features.mean(axis=1)))
        rotation = raw_pose[:, 3:].reshape(-1, 3, 3)
        rotation = rotation / mx.maximum(mx.linalg.norm(rotation, axis=-1, keepdims=True), 1e-12)
        # torch.svd(A) gives A = U S V^T with ``left = U`` and ``right = V``.
        left, _, right_t = mx.linalg.svd(rotation.transpose(0, 2, 1), stream=mx.cpu)
        right = right_t.transpose(0, 2, 1)
        determinant = _determinant(right @ left.transpose(0, 2, 1))
        right = mx.concatenate((right[:, :, :-1], right[:, :, -1:] * determinant[:, None, None]), axis=-1)
        rotation = right @ left.transpose(0, 2, 1)
        top = mx.concatenate((rotation, raw_pose[:, :3, None]), axis=-1)
        bottom = mx.broadcast_to(mx.array([0.0, 0.0, 0.0, 1.0]), (top.shape[0], 1, 4))
        return mx.concatenate((top, bottom), axis=1)


class MetricHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = mx.zeros((1, 1, 1024))
        self.projects_x = nn.Linear(1024, 768)
        self.projects_y = nn.Linear(1024, 768)
        self.blocks = [CrossBlock() for _ in range(2)]
        self.output = nn.Linear(768, 1)

    def __call__(self, tokens, positions, batch_size):
        context = self.projects_y(tokens.reshape(batch_size, -1, 1024).astype(mx.float32))
        positions = positions.reshape(batch_size, -1, 2)
        features = self.projects_x(mx.broadcast_to(self.token, (batch_size, 1, 1024)))
        for block in self.blocks:
            features = block(features, context, positions)
        return mx.exp(self.output(features).reshape(batch_size))


class SparseResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SubmanifoldConv3d(channels, channels, 3)
        self.conv2 = SubmanifoldConv3d(channels, channels, 3)

    def __call__(self, features, coordinates, cache=None):
        residual, cache = self.conv1(nn.silu(self.norm1(features)), coordinates, cache)
        residual, cache = self.conv2(nn.silu(residual), coordinates, cache)
        return features + residual, cache


class SparseDown(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)

    def __call__(self, features, coordinates):
        features, coordinates, mapping = sparse_pool(features, coordinates)
        return self.linear(features), coordinates, mapping


class SparseUp(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)

    def __call__(self, features, mapping):
        return self.linear(features)[mapping.parent]


class SparseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 256, 512)
        self.input_proj = nn.Linear(3, 32)
        self.down_stages = [[SparseResidual(dimension)] for dimension in channels]
        self.downsample_blocks = [
            SparseDown(first, second) for first, second in zip(channels[:-1], channels[1:])
        ]
        self.fuse_proj = [nn.Linear(1024, 512), nn.SiLU(), nn.Linear(512, 512)]
        self.bottleneck_stage = [SparseResidual(512)]
        self.upsample_blocks = [
            SparseUp(first, second) for first, second in zip(channels[:0:-1], channels[-2::-1])
        ]
        self.up_stages = [[SparseResidual(dimension)] for dimension in channels[-2::-1]]
        self.out_proj = nn.Linear(32, 1)

    def __call__(self, features, coordinates, condition, cache=None):
        features = self.input_proj(features)
        caches, mappings, skips = [cache, None, None, None, None], [], []
        for index, stage in enumerate(self.down_stages):
            features, caches[index] = stage[0](features, coordinates, caches[index])
            if index < 4:
                skips.append((features, coordinates))
                features, coordinates, mapping = self.downsample_blocks[index](features, coordinates)
                mappings.append(mapping)
        sampled = condition[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
        features = sequence(self.fuse_proj, mx.concatenate((features, sampled), axis=-1))
        features, caches[4] = self.bottleneck_stage[0](features, coordinates, caches[4])
        for index, (upsample, stage) in enumerate(zip(self.upsample_blocks, self.up_stages)):
            level = 3 - index
            skip_features, coordinates = skips[level]
            features = upsample(features, mappings[level])
            features, caches[level] = stage[0](features + skip_features, coordinates, caches[level])
        return self.out_proj(features)


class DepthRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projector = [Conv2d(1026, 1024, 1), nn.GELU(), Conv2d(1024, 512, 1)]
        self.network = SparseUNet()

    def __call__(self, rays, log_depth, feature):
        condition = sequence(self.feature_projector, feature.astype(mx.float32))
        batch_size, height, width = log_depth.shape
        batch, rows, columns = mx.meshgrid(
            mx.arange(batch_size), mx.arange(height), mx.arange(width), indexing="ij"
        )
        features_rays = rays.reshape(-1, 2)
        for _ in range(3):
            bins = mx.round(log_depth * 256).astype(mx.int32)
            bins = bins - bins.min(axis=(1, 2), keepdims=True)
            coordinates = mx.stack((batch, rows, columns, bins), axis=-1).reshape(-1, 4).astype(mx.int32)
            features = mx.concatenate((features_rays, log_depth.reshape(-1, 1)), axis=-1)
            dtype = self.network.input_proj.weight.dtype
            cache = PixelNeighborMap(coordinates, bins[0]) if batch_size == 1 else None
            residual = self.network(
                features.astype(dtype), coordinates, condition.astype(dtype), cache
            )
            log_depth = log_depth + residual.reshape(log_depth.shape).astype(mx.float32)
            mx.eval(log_depth)
        depth = mx.exp(log_depth)[..., None]
        return mx.concatenate((rays * depth, depth), axis=-1)


MEAN = mx.array((0.485, 0.456, 0.406))
DEVIATION = mx.array((0.229, 0.224, 0.225))


class Pi4(nn.Module):
    """Geometry from RGB images ``[B, T, H, W, 3]`` in [0, 1] (channels last)."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.geometry_decoder = GeometryDecoder()
        self.point_head = PointHead()
        self.camera_head = CameraHead()
        self.metric_head = MetricHead()
        self.depth_refiner = DepthRefiner()

    @classmethod
    def from_pretrained(
        cls, model_id="yyfz233/Pi4", *, revision=None, dtype=mx.bfloat16, refiner_dtype=mx.float32
    ):
        """Load the PyTorch checkpoint. ``dtype`` is used for the encoder and decoder
        (the parts PyTorch runs under bfloat16 autocast) and ``refiner_dtype`` for the
        sparse depth refiner; the other heads stay float32."""
        location = Path(model_id)
        if location.is_dir():
            location = location / "model.safetensors"
        if not location.is_file():
            from huggingface_hub import hf_hub_download

            location = hf_hub_download(repo_id=str(model_id), filename="model.safetensors", revision=revision)
        model = cls()
        convolutions = {
            name for name, module in model.named_modules() if isinstance(module, Conv2d)
        }
        weights = []
        for name, value in mx.load(str(location)).items():
            module, _, leaf = name.rpartition(".")
            if module.split(".")[0] not in model:
                continue  # tracking and motion-mask weights
            if module in convolutions and leaf == "weight":
                value = value.transpose(0, 2, 3, 1)
            weights.append((name, value))
        model.load_weights(weights, strict=True)
        model.encoder.set_dtype(dtype)
        model.geometry_decoder.set_dtype(dtype)
        model.depth_refiner.network.set_dtype(refiner_dtype)
        mx.eval(model.parameters())
        return model.eval()

    def __call__(self, images):
        batch_size, frames, height, width, _ = images.shape
        if height % 16 or width % 16:
            raise ValueError("image dimensions must be positive multiples of 16")
        dtype = self.encoder.projection.weight.dtype
        normalized = (images.astype(mx.float32) - MEAN) / DEVIATION
        positions = patch_positions(batch_size * frames, height // 16, width // 16)
        tokens = self.encoder(normalized.reshape(-1, height, width, 3).astype(dtype))
        geometry = self.geometry_decoder(tokens, positions, batch_size, frames).astype(mx.float32)
        mx.eval(geometry)
        points, confidences = [], []
        for frame_tokens in mx.split(geometry[:, 5:], geometry.shape[0]):
            rays, log_depth, confidence, feature = self.point_head(frame_tokens, height, width)
            coarse_depth = mx.log(mx.maximum(mx.exp(log_depth), 1e-6))
            points.append(self.depth_refiner(rays, coarse_depth, feature))
            confidences.append(confidence)
        cameras = self.camera_head(geometry, positions).reshape(batch_size, frames, 4, 4)
        metric_scale = self.metric_head(geometry[:, 5:], positions[:, 5:], batch_size)
        return dict(
            local_points=mx.concatenate(points).reshape(batch_size, frames, height, width, 3),
            camera_poses=cameras,
            metric_scale=metric_scale,
            confidence=mx.sigmoid(mx.concatenate(confidences).reshape(batch_size, frames, height, width)),
        )
