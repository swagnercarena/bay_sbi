# coding=utf-8

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Embedding models."""


from functools import partial
from typing import Any, Callable, List, Sequence, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any


class MLP(nn.Module):
    """MLP"""
    output_dim: int
    hidden_layers: List[int]
    activation: str

    @nn.compact
    def __call__(self, inputs):

        x = inputs
        for dim in self.hidden_layers:
            x = nn.Dense(dim)(x)
            x = getattr(jax.nn, self.activation)(x)

        return nn.Dense(self.output_dim)(x)

        # providing a name is optional though!
        # the default autonames would be "Dense_0", "Dense_1", ...
        return x


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    skip_traditional: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if self.skip_traditional:
            # Use the traditional ResNet skip connection
            if residual.shape != y.shape:
                residual = self.conv(self.filters * 4, (1, 1),
                                    self.strides, name='conv_proj')(residual)
                residual = self.norm(name='norm_proj')(residual)
        else:
            # Use the ResNetD skip connection.
            if self.strides != (1,1):
                residual = nn.avg_pool(residual, (2,2), strides=(2,2),
                                       padding=((0, 0), (0,0)))
            if residual.shape != y.shape:
                residual = self.conv(self.filters * 4, (1, 1), (1, 1),
                                     name='conv_proj')(residual)
                residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetDBlock(BottleneckResNetBlock):
    "Bottleneck ResNetD block."
    skip_traditional: bool = False


class ResNet(nn.Module):
    """ResNet Class"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_outputs: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm, use_running_average=not train,
            momentum=0.9, epsilon=1e-5, dtype=self.dtype
        )

        x = conv(self.num_filters, (7, 7), (2, 2),
                         padding=[(3, 3), (3, 3)],
                         name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i, strides=strides,
                    conv=conv, norm=norm, act=self.act
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_outputs, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


class ResNetD(nn.Module):
    """ResNet Class"""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_outputs: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm, use_running_average=not train,
            momentum=0.9, epsilon=1e-5, dtype=self.dtype
        )

        # First stem.
        x = conv(self.num_filters // 2, (3, 3), (2, 2),
                 padding=[(1, 1), (1, 1)], name='conv_init_1')(x)
        x = norm(name='bn_init_1')(x)
        x = nn.relu(x)

        # Second stem.
        x = conv(self.num_filters // 2, (3, 3), (1, 1),
                 padding=[(1, 1), (1, 1)], name='conv_init_2')(x)
        x = norm(name='bn_init_2')(x)
        x = nn.relu(x)

        # Third stem.
        x = conv(self.num_filters, (3, 3), (1, 1),
                 padding=[(1, 1), (1, 1)], name='conv_init_3')(x)
        x = norm(name='bn_init_3')(x)
        x = nn.relu(x)

        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i, strides=strides,
                    conv=conv, norm=norm, act=self.act
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_outputs, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18VerySmall = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                            block_cls=ResNetBlock, num_filters=8)
ResNet18Small = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, num_filters=16)
ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)

ResNetD50 = partial(ResNetD, stage_sizes=[3, 4, 6, 3],
                    block_cls=BottleneckResNetDBlock)

ResNet18Local = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)
