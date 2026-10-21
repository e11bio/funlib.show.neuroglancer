from .scale_pyramid import ScalePyramid
from .shaders import create_shader_code
import neuroglancer
import numpy as np
from funlib.persistence import Array

# dtypes neuroglancer can't show, and what to show them as instead
DTYPE_CONVERSIONS = {
    np.dtype(bool): np.uint8,
    np.dtype(np.int64): np.uint64,
    np.dtype(np.float64): np.float32,
    np.dtype(np.float16): np.float32,
}


def is_channel_axis(array: Array) -> list[bool]:
    """For each axis of ``array``, whether it is a channel (non-spatial) axis."""
    types = getattr(array, "types", None)
    if types is None:
        return ["^" in name for name in array.axis_names]
    return [t not in ("space", "time") for t in types]


def channel_shape(array: Array) -> list[int]:
    """The sizes of the channel axes of ``array``."""
    return [
        size
        for size, is_channel in zip(array.shape, is_channel_axis(array))
        if is_channel
    ]


def create_coordinate_space(
    array: Array, default_units: str = "nm"
) -> tuple[neuroglancer.CoordinateSpace, list[int]]:
    assert array.spatial_dims > 0

    is_channel = is_channel_axis(array)

    def interleave(list, fill_value):
        return [fill_value if c else list.pop(0) for c in is_channel]

    # neuroglancer shows axes with a trailing "^" as channels
    names = [
        name if not c or name.endswith("^") else f"{name}^"
        for name, c in zip(array.axis_names, is_channel)
    ]
    units = interleave([unit or default_units for unit in array.units], "")
    scales = interleave(list(array.voxel_size), 1)
    offset = interleave(list(array.offset / array.voxel_size), 0)

    return (
        neuroglancer.CoordinateSpace(names=names, units=units, scales=scales),
        offset,
    )


def guess_volume_type(array: Array) -> str:
    """Guess whether ``array`` should be shown as an image or a segmentation.

    32 and 64 bit integer arrays without channels are segmentations, all
    others (floats, 8 and 16 bit integers, multi-channel arrays) are images.
    This differs from neuroglancer's own guess, which shows ``uint16`` raw data
    as a segmentation.
    """
    dtype = np.dtype(array.dtype)
    if (
        not channel_shape(array)
        and np.issubdtype(dtype, np.integer)
        and dtype.itemsize >= 4
    ):
        return "segmentation"
    return "image"


def guess_shader(array: Array) -> str | None:
    """Guess a shader for ``array``, ``None`` for neuroglancer's default.

    Images with a single channel axis of 3 or more channels (affinities,
    LSDs, ...) get ``"rgb"`` on the first three channels, 2 channels get
    ``"add"``. Everything else (grayscale images, segmentations, multiple
    channel axes) uses the default shader.
    """
    if guess_volume_type(array) == "segmentation":
        return None
    channels = channel_shape(array)
    if len(channels) != 1:
        return None
    if channels[0] >= 3:
        return "rgb"
    if channels[0] == 2:
        return "add"
    return None


def add_layer(
    context,
    array: Array | list[Array],
    name: str,
    opacity: float | None = None,
    shader: str | None = None,
    rgb_channels=None,
    color=None,
    channel: int | None = None,
    visible=True,
    value_scale_factor=1.0,
    volume_type: str | None = None,
    default_units: str = "nm",
):
    """Add a layer to a neuroglancer context.

    Args:

        context:

            The neuroglancer context to add a layer to, as obtained by
            ``viewer.txn()``.

        array:

            A ``funlib.persistence`` ``Array``. If a list of arrays is given,
            a ``ScalePyramid`` layer is generated.

        name:

            The name of the layer.

        opacity:

            A float to define the layer opacity between 0 and 1.

        shader:

            A string to be used as the shader. Possible values are:

                None     :  neuroglancer's default shader
                'rgb'    :  Three channels as RGB, with a brightness slider.
                            See argument ``rgb_channels``.
                'gray'   :  One channel as grayscale, with a contrast slider.
                            See argument ``channel``.
                'color'  :  One channel as a constant color. See arguments
                            ``color`` and ``channel``.
                'binary' :  Shows a binary image as black/white.
                'heatmap':  Shows an intensity image as a jet color map.
                'add'    :  All channels summed into RGB (channel ``i`` to
                            ``i % 3``), one weight slider per channel.
                'cadd'   :  All channels summed into RGB in groups of three,
                            one bias slider.
                'shuffle':  Channels as repeating R/G/B, with per-channel
                            on/off and display ranges and a seed to shuffle
                            the colors.

            Any string containing ``main(`` is used as GLSL shader code as is.

        rgb_channels:

            Which channels to use for RGB (default is ``[0, 1, 2]``).

        color:

            A list of floats (or a ``#rrggbb`` string) for the color shader.
            Defaults to a random color.

        channel:

            The channel to show with the ``gray``, ``color``, ``binary`` and
            ``heatmap`` shaders (default 0).

        visible:

            A bool which defines the initial layer visibility.

        value_scale_factor:

            A float to scale array values with for visualization (default
            brightness of the ``rgb`` shader).

        volume_type:

            ``"image"`` or ``"segmentation"``. By default, arrays shown with
            a shader are images, otherwise see ``guess_volume_type``.

        default_units:

            Units for spatial axes the array has no units for.
    """

    is_multiscale = isinstance(array, list)
    arrays = array if is_multiscale else [array]
    array = arrays[0]

    if volume_type is None:
        volume_type = "image" if shader is not None else guess_volume_type(array)

    def local_volume(a):
        dimensions, voxel_offset = create_coordinate_space(a, default_units)
        data = a.data
        if np.dtype(data.dtype) in DTYPE_CONVERSIONS:
            data = data.astype(DTYPE_CONVERSIONS[np.dtype(data.dtype)])
        return neuroglancer.LocalVolume(
            data=data,
            voxel_offset=voxel_offset,
            dimensions=dimensions,
            volume_type=volume_type,
        )

    if is_multiscale:
        layer = ScalePyramid([local_volume(a) for a in arrays])
    else:
        layer = local_volume(array)

    layer_kwargs = {"name": name, "layer": layer, "visible": visible}
    if opacity is not None:
        layer_kwargs["opacity"] = opacity

    if shader is not None:
        channels = channel_shape(array)
        assert len(channels) <= 1, (
            f"Shaders support at most one channel axis, {name} has "
            f"{len(channels)} ({array.axis_names})"
        )
        layer_kwargs["shader"] = create_shader_code(
            shader,
            num_channels=channels[0] if channels else None,
            rgb_channels=rgb_channels,
            color=color,
            channel=channel,
            scale_factor=value_scale_factor,
        )

    context.layers.append(**layer_kwargs)
