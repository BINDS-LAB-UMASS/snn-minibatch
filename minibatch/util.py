import torch
import bindsnet
from matplotlib import cm


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].

    Source: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b#gistcomment-2398882
    """
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)
    return value


def max_without_indices(inputs, dim=0):
    return torch.max(inputs, dim=dim)[0]


class PoissonNodes(bindsnet.network.nodes.Nodes):
    def __init__(
        self,
        rates=None,
        n=None,
        shape=None,
        traces=False,
        traces_additive=False,
        tc_trace=20.0,
        trace_scale=1.0,
        sum_input=False,
        **kwargs,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        if rates is None:
            self.register_buffer("rates", torch.Tensor())
        else:
            self.register_buffer("rates", rates)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the outputs of the population equal to the inputs.

        :param x: Inputs to the layer (ignored, if any).
        """
        self.s = torch.random.rand(self.batch_size, *self.shape) < self.rates * (
            self.dt / 1000
        )

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.rates = torch.zeros(batch_size, *self.shape, device=self.rates.device)
