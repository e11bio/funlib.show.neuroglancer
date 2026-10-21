#!/usr/bin/env python

from funlib.show.neuroglancer import add_layer
from funlib.show.neuroglancer.add_layer import guess_shader
from funlib.show.neuroglancer.shaders import SHADERS
from funlib.persistence import open_ds
from funlib.persistence.arrays.metadata import get_default_metadata_format
from funlib.geometry import Coordinate
import argparse
import glob
import neuroglancer
import os
import re
import webbrowser
import zarr
from pathlib import Path
import numpy as np  # noqa: F401 This import is used in the eval statement below

# attributes that may hold the voxel size, tried if the configured
# funlib.persistence ``voxel_size_attr`` is missing (gunpowder snapshots write
# ``voxel_size``, older daisy pipelines ``resolution``)
VOXEL_SIZE_ATTRS = ("voxel_size", "resolution")

# names of the scale levels in a multiscale group (``s0``, ``s1``, ... or OME
# style ``0``, ``1``, ...)
SCALE_LEVEL = re.compile(r"^s?(\d+)$")


def parse_ds_name(ds):
    """Split a dataset path into the path (glob) and an optional slice.

    ``raw[0]`` and ``raw:[0]`` slice ``raw``, ``raw[0]:`` (trailing ``:``) is
    the glob ``raw[0]`` without slicing.
    """
    if ds.endswith(":"):
        return ds[:-1], None
    match = re.fullmatch(r"(.*?):?\[([^\[\]]*)\]", ds)
    if match is None:
        return ds, None
    ds, slices = match.groups()
    return ds, eval(f"np.s_[{slices or ':'}]")


def open_ds_any(path, voxel_size=None):
    """``open_ds``, but if the dataset lacks the configured voxel size
    attribute, take the voxel size from any of ``VOXEL_SIZE_ATTRS``. Offsets
    are in world units, so a missed voxel size misplaces the array."""
    if voxel_size is None:
        attrs = zarr.open(str(path), mode="r").attrs
        configured = get_default_metadata_format().voxel_size_attr
        if configured.split("/")[0] not in attrs:
            voxel_size = next((attrs[k] for k in VOXEL_SIZE_ATTRS if k in attrs), None)
    return open_ds(str(path), voxel_size=voxel_size)


def open_array(path, voxel_size=None):
    """Open an array, or the levels of a multiscale group as a list of arrays.

    ``voxel_size`` overrides the voxel size (of the first level, the others are
    scaled accordingly).
    """
    node = zarr.open(str(path), mode="r")
    if not isinstance(node, zarr.Group):
        return open_ds_any(path, voxel_size)

    levels = sorted(
        (int(match.group(1)), key)
        for key in node.array_keys()
        if (match := SCALE_LEVEL.match(key))
    )
    if not levels:
        raise ValueError(
            f"{path} is a group but not a multiscale array (no s0, s1, ... or "
            f"0, 1, ... arrays). Use '{path}/*' to show the arrays it contains."
        )
    print(f"Found scales {[key for _, key in levels]}")

    arrays = [open_ds_any(Path(path, key)) for _, key in levels]
    if voxel_size is not None:
        base = arrays[0].voxel_size
        arrays = [
            open_ds_any(
                Path(path, key),
                Coordinate(voxel_size) * a.voxel_size / base,
            )
            for (_, key), a in zip(levels, arrays)
        ]
    return arrays


def parse_shader_spec(spec):
    """Turn a shader spec (``<shader>[:<params>][@<channel>]``, see the parser
    epilog) into ``add_layer`` kwargs."""
    if spec in ("default", "none"):
        return {}

    spec, _, channel = spec.partition("@")
    name, _, params = spec.partition(":")
    if name not in SHADERS:
        raise ValueError(f"unknown shader {name!r}, choose from {list(SHADERS)}")

    kwargs = {"shader": name}
    if name in ("gray", "heatmap", "binary") and params:
        # gray:3 is short for gray@3
        channel, params = params, ""
    if channel:
        if name not in ("gray", "color", "heatmap", "binary"):
            raise ValueError(f"shader {name!r} takes no channel")
        kwargs["channel"] = int(channel)

    if not params:
        return kwargs
    if name == "rgb":
        kwargs["rgb_channels"] = [int(c) for c in params.split(",")]
        if len(kwargs["rgb_channels"]) != 3:
            raise ValueError("rgb needs 3 channels")
    elif name == "color" and params.startswith("#"):
        kwargs["color"] = params
    elif name == "color":
        kwargs["color"] = [float(c) for c in params.split(",")]
        if len(kwargs["color"]) != 3:
            raise ValueError("color needs 3 values")
    else:
        raise ValueError(f"shader {name!r} takes no parameters")

    return kwargs


def shader_spec(spec):
    """argparse type for shader specs: validate early, keep the string."""
    try:
        parse_shader_spec(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid shader {spec!r}: {e}")
    return spec


VOLUME_TYPES = {
    "image": "image",
    "img": "image",
    "segmentation": "segmentation",
    "seg": "segmentation",
    "auto": None,
}


class DatasetAction(argparse.Action):
    """Start a new dataset group; the options after it apply to its datasets."""

    def __call__(self, parser, namespace, values, option_string=None):
        groups = getattr(namespace, self.dest) or []
        groups.append({"paths": values})
        setattr(namespace, self.dest, groups)


class GroupOptionAction(argparse.Action):
    """Attach an option to the preceding --dataset/-d group."""

    def __call__(self, parser, namespace, values, option_string=None):
        groups = getattr(namespace, self.dest)
        if not groups:
            parser.error(f"{option_string} has to follow a --dataset/-d argument")
        group = groups[-1]
        key = self.metavar.lower().replace("-", "_")
        if key in group:
            parser.error(f"Only one {option_string} allowed per --dataset/-d")
        if isinstance(values, list) and len(values) not in (1, len(group["paths"])):
            parser.error(
                f"{option_string} takes one value for all datasets of the "
                f"preceding --dataset/-d, or one per dataset "
                f"({len(group['paths'])}), got {len(values)}: {values}"
            )
        group[key] = values


parser = argparse.ArgumentParser(
    description="Show zarr arrays in neuroglancer.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""shader specs (-c):
  default         neuroglancer's default shader
  rgb[:r,g,b]     channels r,g,b as RGB (default 0,1,2)
  gray[:c]        channel c as grayscale, with a contrast slider
  color[:COLOR][@c]
                  channel c (default 0) in a constant color, COLOR is r,g,b
                  floats in [0, 1] or #rrggbb (default: random)
  heatmap[:c]     channel c as a jet color map
  binary[:c]      channel c as black/white
  add             all channels summed into RGB, a weight slider per channel
  cadd            all channels summed into RGB in groups of three
  shuffle         channels as R/G/B with shufflable colors and ranges

  Without -c: rgb for images with >= 3 channels, add for 2, default otherwise.

examples:
  neuroglancer -d data.zarr/raw data.zarr/labels
  neuroglancer -d data.zarr/affs -c rgb:3,4,5 -d data.zarr/raw -c gray
  neuroglancer -d 'data.zarr/crop_*/raw[0]'
  neuroglancer -d data.zarr/raw_u16_labels -t seg -vs 40,8,8
""",
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    nargs="+",
    action=DatasetAction,
    metavar="PATH",
    help="The paths to the datasets to show (globs allowed). A slice can be "
    "appended, e.g. 'raw[0]' or 'raw:[0, 100:200]'; append ':' to a glob "
    "with brackets to not slice it, e.g. 'raw_[1345]:'. The options below "
    "apply to the datasets of the preceding --dataset/-d",
    dest="groups",
)
parser.add_argument(
    "--shader",
    "-c",
    type=shader_spec,
    nargs="+",
    action=GroupOptionAction,
    metavar="SHADER",
    help="Shader specs, one for all datasets or one per dataset (see below)",
    dest="groups",
)
parser.add_argument(
    "--type",
    "-t",
    type=str,
    nargs="+",
    choices=list(VOLUME_TYPES),
    action=GroupOptionAction,
    metavar="TYPE",
    help="Volume types (image/img, segmentation/seg, auto), one for all "
    "datasets or one per dataset. Default: auto, i.e., segmentation for 32/64 "
    "bit integers without channels and no shader, image otherwise",
    dest="groups",
)
parser.add_argument(
    "--voxel-size",
    "-vs",
    type=str,
    action=GroupOptionAction,
    metavar="VOXEL_SIZE",
    help="Override the voxel size, e.g. 40,8,8",
    dest="groups",
)
parser.add_argument(
    "--slices",
    "-s",
    type=str,
    action=GroupOptionAction,
    metavar="SLICES",
    help="A slice operation to apply to the datasets, e.g. '0' or '0, 100:200'",
    dest="groups",
)
parser.add_argument(
    "--no-browser",
    "-n",
    type=bool,
    nargs="?",
    const=True,
    default=False,
    help="If set, do not open a browser, just print a URL",
)
parser.add_argument(
    "--bind-address",
    "-b",
    type=str,
    default="0.0.0.0",
    help="Bind address",
)
parser.add_argument("--port", "-p", type=int, default=0, help="The port to bind to.")


def per_dataset(values, i):
    """The i-th value of a group option (one value for all, or one per dataset)."""
    if values is None:
        return None
    return values[0] if len(values) == 1 else values[i]


def add_dataset(viewer, ds_path, spec, volume_type, slices, voxel_size, layer_names):
    print(f"Adding {ds_path}" + (f" with slices {slices}" if slices else ""))
    array = open_array(ds_path, voxel_size)
    levels = array if isinstance(array, list) else [array]
    if slices is not None:
        for level in levels:
            level.lazy_op(slices)

    if spec is None:
        spec = guess_shader(levels[0]) or "default"
    print(
        f"  shape {levels[0].shape}, axes {levels[0].axis_names}, "
        f"voxel size {levels[0].voxel_size}, {levels[0].dtype}, shader {spec}"
    )

    # layer names have to be unique, prefix parents (then number) until they are
    ds_path = Path(ds_path)
    candidates = [ds_path.name]
    for parent in reversed(ds_path.parent.parts):
        candidates.append(f"{parent}/{candidates[-1]}")
    name = next((n for n in candidates if n not in layer_names), None)
    i = 2
    while name is None or name in layer_names:
        name, i = f"{ds_path.name} ({i})", i + 1

    with viewer.txn() as s:
        add_layer(s, array, name, volume_type=volume_type, **parse_shader_spec(spec))
    layer_names.add(name)


def main():
    args = parser.parse_args()
    if not args.groups:
        parser.error("Nothing to show, pass datasets with --dataset/-d")

    neuroglancer.set_server_bind_address(args.bind_address, bind_port=args.port)
    viewer = neuroglancer.Viewer()
    layer_names = set()

    for group in args.groups:
        group_slices = group.get("slices")
        if group_slices is not None:
            group_slices = eval(f"np.s_[{group_slices}]")
        voxel_size = group.get("voxel_size")
        if voxel_size is not None:
            voxel_size = Coordinate(int(v) for v in voxel_size.split(","))

        for i, path in enumerate(group["paths"]):
            glob_path, slices = parse_ds_name(path)
            if slices is None:
                slices = group_slices
            spec = per_dataset(group.get("shader"), i)
            volume_type = VOLUME_TYPES[per_dataset(group.get("type"), i) or "auto"]

            ds_paths = sorted(glob.glob(glob_path))
            if not ds_paths:
                print(f"No datasets match {glob_path}, skipping")

            for ds_path in ds_paths:
                try:
                    add_dataset(
                        viewer,
                        ds_path,
                        spec,
                        volume_type,
                        slices,
                        voxel_size,
                        layer_names,
                    )
                except Exception as e:
                    print(f"Couldn't add {ds_path}, skipping: {type(e).__name__}: {e}")

    url = str(viewer)
    print(url)
    if os.environ.get("DISPLAY") and not args.no_browser:
        webbrowser.open_new(url)

    print("Press ENTER to quit")
    input()
