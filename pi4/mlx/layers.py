"""MLX versions of the shared Pi4 layers (``pi4.models.layers``).

Images and feature maps are channels-last (NHWC) here; parameter names match the
PyTorch modules, so the same checkpoint loads into both.
"""

import mlx.core as mx
from mlx import nn


def sequence(layers, features):
    for layer in layers:
        features = layer(features)
    return features


class Conv2d(nn.Module):
    """``torch.nn.Conv2d`` on NHWC input, including ``padding_mode="replicate"``."""

    def __init__(self, inputs, outputs, kernel, stride=1, padding=0, replicate=False, bias=True):
        super().__init__()
        self.weight = mx.zeros((outputs, kernel, kernel, inputs))
        if bias:
            self.bias = mx.zeros((outputs,))
        self._stride, self._padding, self._replicate = stride, padding, replicate

    def __call__(self, features):
        padding = self._padding
        if padding and self._replicate:
            pad = [(0, 0), (padding, padding), (padding, padding), (0, 0)]
            features, padding = mx.pad(features, pad, mode="edge"), 0
        output = mx.conv2d(features, self.weight, stride=self._stride, padding=padding)
        return output + self.bias if "bias" in self else output


class ConvTranspose2d(nn.Module):
    """``torch.nn.ConvTranspose2d`` with ``kernel == stride`` (no overlap), on NHWC."""

    def __init__(self, inputs, outputs, kernel, stride):
        super().__init__()
        assert kernel == stride
        self.weight = mx.zeros((outputs, kernel, kernel, inputs))
        self.bias = mx.zeros((outputs,))

    def __call__(self, features):
        return mx.conv_transpose2d(features, self.weight, stride=self.weight.shape[1]) + self.bias


def pixel_shuffle(features, factor=2):
    batch_size, height, width, channels = features.shape
    channels //= factor * factor
    features = features.reshape(batch_size, height, width, channels, factor, factor)
    return features.transpose(0, 1, 4, 2, 5, 3).reshape(
        batch_size, height * factor, width * factor, channels
    )


class PixelShuffle(nn.Module):
    def __call__(self, features):
        return pixel_shuffle(features)


class MLP(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 4 * dimension)
        self.fc2 = nn.Linear(4 * dimension, dimension)

    def __call__(self, features):
        return self.fc2(nn.gelu(self.fc1(features)))


def rotate_half(features):
    first, second = mx.split(features, 2, axis=-1)
    return mx.concatenate((-second, first), axis=-1)


def grid_rope(features, positions):
    dimension = features.shape[-1] // 2
    frequencies = 1.0 / (100.0 ** (mx.arange(0, dimension, 2).astype(mx.float32) / dimension))
    indices = mx.arange(positions.shape[1] + 1).astype(mx.float32)
    angles = (indices[:, None] * frequencies[None]).astype(features.dtype)
    angles = mx.concatenate((angles, angles), axis=-1)
    sine, cosine = mx.sin(angles), mx.cos(angles)
    output = []
    for axis, tokens in enumerate(mx.split(features, 2, axis=-1)):
        index = positions[..., axis]
        output.append(tokens * cosine[index][:, None] + rotate_half(tokens) * sine[index][:, None])
    return mx.concatenate(output, axis=-1)


def patch_positions(batch_size, height, width):
    rows, columns = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
    positions = mx.stack((rows, columns), axis=-1).reshape(1, -1, 2) + 1
    prefix = mx.zeros((1, 5, 2), dtype=positions.dtype)
    positions = mx.concatenate((prefix, positions), axis=1)
    return mx.broadcast_to(positions, (batch_size, *positions.shape[1:]))


def attention(query, key, value):
    return mx.fast.scaled_dot_product_attention(
        query, key, value, scale=query.shape[-1] ** -0.5
    )


class Attention(nn.Module):
    def __init__(self, dimension, heads, normalize=False):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.proj = nn.Linear(dimension, dimension)
        if normalize:
            self.q_norm = nn.LayerNorm(dimension // heads)
            self.k_norm = nn.LayerNorm(dimension // heads)

    def __call__(self, features, positions=None, rotary=None):
        batch_size, token_count, dimension = features.shape
        projected = self.qkv(features).reshape(
            batch_size, token_count, 3, self.heads, dimension // self.heads
        )
        query, key, value = projected.transpose(2, 0, 3, 1, 4)
        if "q_norm" in self:
            query, key = self.q_norm(query), self.k_norm(key)
        if positions is not None:
            query, key = grid_rope(query, positions), grid_rope(key, positions)
        if rotary is not None:
            sine, cosine = rotary
            prefix = token_count - sine.shape[0]

            def apply_rotary(tokens):
                spatial = tokens[:, :, prefix:].astype(mx.float32)
                spatial = spatial * cosine + rotate_half(spatial) * sine
                return mx.concatenate((tokens[:, :, :prefix], spatial.astype(tokens.dtype)), axis=2)

            query, key = apply_rotary(query), apply_rotary(key)
        attended = attention(query, key, value)
        return self.proj(attended.transpose(0, 2, 1, 3).reshape(batch_size, token_count, dimension))


class Block(nn.Module):
    def __init__(self, dimension, heads, normalize=False, epsilon=1e-6, scaled=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension, eps=epsilon)
        self.attn = Attention(dimension, heads, normalize)
        self.norm2 = nn.LayerNorm(dimension, eps=epsilon)
        self.mlp = MLP(dimension)
        if scaled:
            self.residual_scales = mx.ones((2, dimension))

    def __call__(self, features, positions=None, rotary=None, gates=None, modulation=None):
        attention_scale = feedforward_scale = output_scale = 1
        if gates is not None:
            attention_scale, feedforward_scale, output_scale = gates.astype(features.dtype)
        attention_input = self.norm1(features)
        if modulation is not None:
            attention_input = (1 + modulation[0]) * attention_input + modulation[1]
        attention_output = self.attn(attention_input, positions, rotary)
        if "residual_scales" in self:
            attention_output = attention_output * self.residual_scales[0]
        features = features + attention_scale * attention_output
        feedforward_input = self.norm2(features)
        if modulation is not None:
            feedforward_input = (1 + modulation[2]) * feedforward_input + modulation[3]
        feedforward_output = self.mlp(feedforward_input)
        if "residual_scales" in self:
            feedforward_output = feedforward_output * self.residual_scales[1]
        return output_scale * (features + feedforward_scale * feedforward_output)


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.proj = nn.Linear(768, 768)

    def __call__(self, query_features, context, query_positions, context_positions):
        batch_size, query_count, _ = query_features.shape

        def heads(features):
            return features.reshape(batch_size, -1, 12, 64).transpose(0, 2, 1, 3)

        query = grid_rope(heads(self.q_proj(query_features)), query_positions)
        key = grid_rope(heads(self.k_proj(context)), context_positions)
        attended = attention(query, key, heads(self.v_proj(context)))
        return self.proj(attended.transpose(0, 2, 1, 3).reshape(batch_size, query_count, 768))


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.norm_y = nn.LayerNorm(768, eps=1e-6)
        self.cross_attn = CrossAttention()
        self.norm3 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = MLP(768)

    def __call__(self, features, context, positions):
        features = features + self.cross_attn(
            self.norm2(features), self.norm_y(context), positions[:, :1], positions
        )
        return features + self.mlp(self.norm3(features))


class ConvResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = [
            nn.GroupNorm(1, channels, pytorch_compatible=True),
            nn.ReLU(),
            Conv2d(channels, channels, 3, padding=1, replicate=True),
            nn.GroupNorm(channels // 32, channels, pytorch_compatible=True),
            nn.ReLU(),
            Conv2d(channels, channels, 3, padding=1, replicate=True),
        ]

    def __call__(self, features):
        return features + sequence(self.layers, features)


def upsample_block(inputs, outputs):
    return [
        Conv2d(inputs, outputs * 4, 3, padding=1, replicate=True),
        PixelShuffle(),
        Conv2d(outputs, outputs, 3, padding=1, replicate=True),
    ]


def feature_pyramid(tokens, height, width, stages):
    """The token grid plus normalized pixel coordinates at each of ``stages`` scales (NHWC)."""
    batch_size, _, channels = tokens.shape
    base = tokens.reshape(batch_size, height // 16, width // 16, channels)
    aspect_ratio = width / height
    span_horizontal = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_vertical = 1 / (1 + aspect_ratio**2) ** 0.5
    features = []
    for level in range(stages):
        level_height, level_width = height // 16 * 2**level, width // 16 * 2**level
        horizontal = mx.linspace(
            -span_horizontal * (level_width - 1) / level_width,
            span_horizontal * (level_width - 1) / level_width,
            level_width,
        )
        vertical = mx.linspace(
            -span_vertical * (level_height - 1) / level_height,
            span_vertical * (level_height - 1) / level_height,
            level_height,
        )
        vertical, horizontal = mx.meshgrid(vertical, horizontal, indexing="ij")
        coordinates = mx.broadcast_to(
            mx.stack((horizontal, vertical), axis=-1)[None].astype(tokens.dtype),
            (batch_size, level_height, level_width, 2),
        )
        features.append(mx.concatenate((base, coordinates), axis=-1) if level == 0 else coordinates)
    return features


class ConvPyramid(nn.Module):
    def __init__(self, dimensions, output_channels):
        super().__init__()
        self.input_blocks = [
            Conv2d(1026 if index == 0 else 2, channels, 1) for index, channels in enumerate(dimensions)
        ]
        self.res_blocks = [[ConvResidual(channels)] for channels in dimensions]
        self.resamplers = [
            upsample_block(first, second) for first, second in zip(dimensions[:-1], dimensions[1:])
        ]
        self.output = Conv2d(dimensions[-1], output_channels, 1)

    def __call__(self, pyramid):
        pyramid = [level.astype(mx.float32) for level in pyramid]
        features = self.res_blocks[0][0](self.input_blocks[0](pyramid[0]))
        prefix = features
        for index in range(1, len(self.res_blocks)):
            features = sequence(self.resamplers[index - 1], features) + self.input_blocks[index](
                pyramid[index]
            )
            features = self.res_blocks[index][0](features)
        return self.output(features), prefix
