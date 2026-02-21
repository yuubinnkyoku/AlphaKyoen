import jax.numpy as jnp
import flax.linen as nn

class ResBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = x + residual
        x = nn.relu(x)
        return x

class ResNet(nn.Module):
    num_blocks: int = 4
    filters: int = 64

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Input x is (B, 3, 9, 9) in NCHW format.
        # Flax Conv expects NHWC by default, so we transpose it.
        x = jnp.transpose(x, (0, 2, 3, 1)) # (B, 9, 9, 3)

        x = nn.Conv(features=self.filters, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResBlock(filters=self.filters)(x, train=train)

        # Policy Head
        p = nn.Conv(features=2, kernel_size=(1, 1), padding='SAME', use_bias=False)(x)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1)) # Flatten
        p = nn.Dense(features=81)(p)
        p = nn.log_softmax(p)

        # Value Head
        v = nn.Conv(features=1, kernel_size=(1, 1), padding='SAME', use_bias=False)(x)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1)) # Flatten
        v = nn.Dense(features=64)(v)
        v = nn.relu(v)
        v = nn.Dense(features=1)(v)
        v = jnp.tanh(v)

        return p, v

class SmallResNet(ResNet):
    """Backward-compatible alias for existing training scripts."""
    def __init__(self, num_blocks=4, filters=64, **kwargs):
        super().__init__(num_blocks=num_blocks, filters=filters, **kwargs)
