"""GLSL shader generators for neuroglancer image layers.

Every generator returns a complete shader string. ``create_shader_code``
dispatches on a shader name; see ``SHADERS`` for the available names.
"""

import random


def _channel_arg(channel):
    """Argument for ``getDataValue``: empty for arrays without a channel axis."""
    return "" if channel is None else str(int(channel))


def _invlerp(name, channel):
    """An ``invlerp`` UI control (contrast slider), optionally bound to a channel."""
    if channel is None:
        return f"#uicontrol invlerp {name}"
    return f"#uicontrol invlerp {name}(channel=[{int(channel)}])"


def _to_hex(color):
    """``[r, g, b]`` floats in [0, 1] (or a ``#rrggbb`` string) to ``#rrggbb``."""
    if isinstance(color, str):
        return color
    return "#" + "".join(
        f"{int(round(255 * min(max(c, 0.0), 1.0))):02x}" for c in color
    )


def random_color():
    """A random bright color as ``#rrggbb`` (components >= 0.5 to avoid dim colors)."""
    return _to_hex([random.uniform(0.5, 1.0) for _ in range(3)])


def rgb_shader(rgb_channels=(0, 1, 2), scale_factor=1.0):
    """Three channels as red, green and blue, with a brightness slider."""
    r, g, b = (int(c) for c in rgb_channels)
    return f"""#uicontrol float brightness slider(default={scale_factor}, min=0, max={max(4.0, 2 * scale_factor)}, step=0.01)
void main() {{
    emitRGB(
        brightness * vec3(
            toNormalized(getDataValue({r})),
            toNormalized(getDataValue({g})),
            toNormalized(getDataValue({b}))));
}}"""


def gray_shader(channel=None):
    """A single channel as grayscale, with a contrast (invlerp) slider."""
    return f"""{_invlerp("normalized", channel)}
void main() {{
    emitGrayscale(normalized());
}}"""


def color_shader(color=None, channel=None):
    """A single channel as a constant color whose alpha is the intensity.

    ``color`` defaults to a random bright color and can be changed in the UI.
    """
    color = random_color() if color is None else _to_hex(color)
    return f"""#uicontrol vec3 color color(default="{color}")
{_invlerp("normalized", channel)}
void main() {{
    emitRGBA(vec4(color, normalized()));
}}"""


def binary_shader(channel=None):
    """A binary image as black/white."""
    return f"""void main() {{
    emitGrayscale(255.0 * toNormalized(getDataValue({_channel_arg(channel)})));
}}"""


def heatmap_shader(channel=None):
    """An intensity image as a jet color map; zero is transparent."""
    return f"""void main() {{
    float v = toNormalized(getDataValue({_channel_arg(channel)}));
    vec4 rgba = vec4(0, 0, 0, 0);
    if (v != 0.0) {{
        rgba = vec4(colormapJet(v), 1.0);
    }}
    emitRGBA(rgba);
}}"""


def additive_shader(
    num_channels,
    shuffle=False,
    total_default=1.0,
    total_min=0.5,
    total_max=2.5,
    channel_default=1.0,
    channel_min=0.5,
    channel_max=2.5,
    saved_defaults=None,
):
    """Sum all channels into RGB (channel ``i`` goes to ``i % 3``), with one
    weight slider per channel and a total brightness slider.

    Args:

        shuffle:

            Shuffle the channel order and randomize the channel weights.

        saved_defaults:

            Optional dict of slider defaults, keys ``total`` and
            ``channel_<i>``. Slider ranges are widened to include them.
    """
    saved_defaults = saved_defaults or {}

    total_default = saved_defaults.get("total", total_default)
    total_min = min(total_min, total_default)
    total_max = max(total_max, total_default)

    controls = [
        f"#uicontrol float total slider(default={total_default}, "
        f"min={total_min}, max={total_max}, step=0.1)"
    ]
    terms = [[], [], []]

    channel_indices = list(range(num_channels))
    if shuffle:
        random.shuffle(channel_indices)

    for i in channel_indices:
        default = saved_defaults.get(f"channel_{i}", channel_default)
        low = min(channel_min, default)
        high = max(channel_max, default)
        if shuffle:
            default = random.uniform(low, high)

        controls.append(
            f"#uicontrol float channel_{i} slider(default={default}, "
            f"min={low}, max={high}, step=0.1)"
        )
        terms[i % 3].append(f"channel_{i} * toNormalized(getDataValue({i}))")

    r, g, b = (" + ".join(t) if t else "0.0" for t in terms)

    return "\n".join(controls) + f"""
void main() {{
    emitRGB(
        total * vec3(
            {r},
            {g},
            {b}));
}}"""


def combined_additive_shader(num_channels, scale=0.5):
    """Sum all channels into RGB in groups of three (channels 0-2 red, 3-5
    green, 6-8 blue, the rest gray), with a single bias slider."""
    pos_bias = (num_channels + 1.0) / num_channels
    num_groups = max(3, (num_channels + 2) // 3)

    return f"""#define NUM_CHANNELS {num_channels}

#uicontrol float pos_bias slider(default={pos_bias}, min=0, max=5, step=0.1)

void main() {{
    vec3 transform[{num_groups}];
    for (int i = 0; i < {num_groups}; ++i) {{
        transform[i] = vec3(0.5, 0.5, 0.5);
    }}
    transform[0] = vec3(1.0, 0.0, 0.0);
    transform[1] = vec3(0.0, 1.0, 0.0);
    transform[2] = vec3(0.0, 0.0, 1.0);

    vec3 color = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < NUM_CHANNELS; ++i) {{
        float value = pos_bias * (toNormalized(getDataValue(i)) - {scale});
        color += value * transform[i / 3];
    }}

    emitRGB(color);
}}"""


def shuffle_shader(
    num_channels,
    slider_default_min=0.0,
    slider_default_max=0.03,
    saved_defaults=None,
):
    """Map channels to repeating red/green/blue and add them up. Each channel
    gets an on/off checkbox and a display range; a ``seed`` slider shuffles
    the channel colors (0 keeps the original order). Also has contrast,
    brightness, per-color scale and invert controls.

    Args:

        saved_defaults:

            Optional dict of control defaults, keys ``chan<i>_active``,
            ``chan<i>_min``, ``chan<i>_max``, ``contrast``, ``brightness``,
            ``scaleR``, ``scaleG``, ``scaleB`` and ``seed``. Slider ranges
            are widened to include them.
    """
    assert (
        num_channels >= 3
    ), f"The shuffle shader needs at least 3 channels, got {num_channels}"

    saved_defaults = saved_defaults or {}
    n = num_channels
    shader = []

    for i in range(n):
        active = saved_defaults.get(f"chan{i}_active", True)
        low = saved_defaults.get(f"chan{i}_min", slider_default_min)
        high = saved_defaults.get(f"chan{i}_max", slider_default_max)
        shader += [
            f"#uicontrol bool chan{i}_active checkbox(default={str(active).lower()})",
            f"#uicontrol float chan{i}_min slider(min={min(0, low)}, max=1, "
            f"step=0.001, default={low})",
            f"#uicontrol float chan{i}_max slider(min=0, max={max(1, high)}, "
            f"step=0.001, default={high})",
        ]

    # (name, default, min, max, step)
    sliders = [
        ("contrast", 0, -3, 3, 0.01),
        ("brightness", 0, -3, 3, 0.01),
        ("scaleR", 1, -3, 3, 0.01),
        ("scaleG", 1, -3, 3, 0.01),
        ("scaleB", 1, -3, 3, 0.01),
        ("seed", 1, 0, 20, 1),
    ]
    for name, default, low, high, step in sliders:
        default = saved_defaults.get(name, default)
        shader.append(
            f"#uicontrol float {name} slider(min={min(low, default)}, "
            f"max={max(high, default)}, step={step}, default={default})"
        )
    shader.append("#uicontrol bool invert checkbox(default=false)")

    def array(type_, values):
        return f"{type_}[]({', '.join(values)})"

    colors = (["vec3(1, 0, 0)", "vec3(0, 1, 0)", "vec3(0, 0, 1)"] * n)[:n]
    channels = range(n)

    shader += [
        "float disp_range(float im, float min, float max) {",
        "    return clamp((im - min) / (max - min), 0.0, 1.0);",
        "}",
        "float rand(float x) {",
        "    return fract(sin(dot(vec2(x, seed), vec2(12.9898, 78.233))) * 43758.5453);",
        "}",
        f"vec3[{n}] shuffleColors(vec3 arr[{n}]) {{",
        f"    vec3[{n}] output_arr = arr;",
        "    if (seed == 0.) {",
        "        return output_arr;",
        "    }",
        f"    for (int i = {n} - 1; i > 0; i--) {{",
        "        int j = int(rand(float(i)) * float(i + 1));",
        "        vec3 temp = output_arr[i];",
        "        output_arr[i] = output_arr[j];",
        "        output_arr[j] = temp;",
        "    }",
        "    return output_arr;",
        "}",
        "void main() {",
        "    vec3 colorSum = vec3(0.0);",
        f"    float channelData[{n}] = "
        + array("float", [f"toNormalized(getDataValue({i}))" for i in channels])
        + ";",
        f"    float minVals[{n}] = "
        + array("float", [f"chan{i}_min" for i in channels])
        + ";",
        f"    float maxVals[{n}] = "
        + array("float", [f"chan{i}_max" for i in channels])
        + ";",
        f"    bool active[{n}] = "
        + array("bool", [f"chan{i}_active" for i in channels])
        + ";",
        f"    vec3 colors[{n}] = shuffleColors({array('vec3', colors)});",
        f"    for (int i = 0; i < {n}; i++) {{",
        "        if (active[i]) {",
        "            colorSum += colors[i] * disp_range(channelData[i], minVals[i], maxVals[i]);",
        "        }",
        "    }",
        "    vec3 im = clamp(colorSum * vec3(scaleR, scaleG, scaleB), 0.0, 1.0);",
        "    im = (im - 0.5) * pow(2.0, contrast) + 0.5 + brightness;",
        "    emitRGB(invert ? 1.0 - im : im);",
        "}",
    ]

    return "\n".join(shader)


# shader name -> one-line description
SHADERS = {
    "rgb": "three channels as RGB (see rgb_channels)",
    "gray": "one channel as grayscale with a contrast slider (see channel)",
    "color": "one channel as a constant color (see color, random if not given)",
    "binary": "a binary image as black/white",
    "heatmap": "an intensity image as a jet color map",
    "add": "all channels summed into RGB, one weight slider per channel",
    "cadd": "all channels summed into RGB in groups of three, one bias slider",
    "shuffle": "channels as repeating R/G/B with shufflable colors and ranges",
}


def create_shader_code(
    shader,
    num_channels=None,
    rgb_channels=None,
    color=None,
    channel=None,
    scale_factor=1.0,
):
    """Create shader code for one of the shaders in ``SHADERS``.

    Args:

        shader:

            The name of the shader (see ``SHADERS``), or GLSL code (anything
            containing ``main(``), which is returned unchanged.

        num_channels:

            The number of channels along the array's channel axis, ``None``
            if it has none.

        rgb_channels:

            The channels to show as RGB (``rgb`` only, default ``[0, 1, 2]``).

        color:

            ``[r, g, b]`` floats or a ``#rrggbb`` string (``color`` only).

        channel:

            The channel to show (``gray``, ``color``, ``binary``, ``heatmap``),
            defaults to 0 for arrays with a channel axis.

        scale_factor:

            Default brightness (``rgb`` only).
    """
    if "main(" in shader:
        return shader

    if shader not in SHADERS:
        raise ValueError(f"Unknown shader {shader!r}, choose from {list(SHADERS)}")

    if channel is None and num_channels is not None:
        channel = 0
    if channel is not None:
        assert num_channels is not None, f"Shader {shader!r}: array has no channels"
        assert channel < num_channels, (
            f"Shader {shader!r}: channel {channel} out of range for "
            f"{num_channels} channels"
        )

    if shader in ("rgb", "add", "cadd", "shuffle"):
        assert (
            num_channels is not None
        ), f"Shader {shader!r} needs an array with a channel axis"

    if shader == "rgb":
        rgb_channels = [0, 1, 2] if rgb_channels is None else list(rgb_channels)
        assert len(rgb_channels) == 3, f"rgb needs 3 channels, got {rgb_channels}"
        assert (
            max(rgb_channels) < num_channels
        ), f"rgb channels {rgb_channels} out of range for {num_channels} channels"
        return rgb_shader(rgb_channels, scale_factor)
    if shader == "gray":
        return gray_shader(channel)
    if shader == "color":
        return color_shader(color, channel)
    if shader == "binary":
        return binary_shader(channel)
    if shader == "heatmap":
        return heatmap_shader(channel)
    if shader == "add":
        return additive_shader(num_channels)
    if shader == "cadd":
        return combined_additive_shader(num_channels)
    if shader == "shuffle":
        return shuffle_shader(num_channels)
