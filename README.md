# funlib.show.neuroglancer

Show zarr arrays in [neuroglancer](https://github.com/google/neuroglancer),
from the command line or from python.

## Install

```bash
pip install git+https://github.com/e11bio/funlib.show.neuroglancer
```

## Usage

```bash
neuroglancer -d <datasets> [options for these datasets] [-d ...]
```

Options after a `-d` apply to its datasets. `-c` and `-t` take one value for
all datasets, or one per dataset.

| Flag                   | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `-d, --dataset`        | paths to arrays, globs allowed                 |
| `-c, --shader`         | shader, see below (default: automatic)         |
| `-t, --type`           | `image`, `seg` or `auto` (default)             |
| `-s, --slices`         | slice to apply, e.g. `0` or `0, 100:200`       |
| `-vs, --voxel-size`    | override the voxel size, e.g. `40,8,8`         |
| `-n, --no-browser`     | only print the URL                             |
| `-b, --bind-address`   | bind address (default `0.0.0.0`)               |
| `-p, --port`           | port (default: random)                         |

Defaults:

- 32/64 bit integer arrays without channels are segmentations, everything else
  is an image.
- Images with 3+ channels get `rgb` on channels 0, 1, 2. Images with 2 channels
  get `add`.
- Groups of scale levels (`s0`, `s1`, ...) are shown as one multiscale layer.
- The voxel size is read from the `voxel_size` or `resolution` attribute.

## Shaders

| Shader            | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| `default`         | neuroglancer's default                                              |
| `rgb[:r,g,b]`     | channels `r,g,b` as RGB (default `0,1,2`)                           |
| `gray[:c]`        | channel `c` as grayscale, with a contrast slider                    |
| `color[:C][@c]`   | channel `c` in color `C` (`r,g,b` in [0, 1] or `#rrggbb`, default random) |
| `heatmap[:c]`     | channel `c` as a jet color map                                      |
| `binary[:c]`      | channel `c` as black/white                                          |
| `add`             | all channels summed into RGB, a slider per channel                  |
| `cadd`            | all channels summed into RGB in groups of three                     |
| `shuffle`         | channels as R/G/B, with per-channel toggles and ranges and shufflable colors |

## Examples

```bash
# raw and labels
neuroglancer -d /path/to/data.zarr/{raw,labels}

# every array in a container
neuroglancer -d '/path/to/data.zarr/*'

# channels 3, 4, 5 as RGB, raw as grayscale
neuroglancer -d /path/to/data.zarr/lsds -c rgb:3,4,5 -d /path/to/data.zarr/raw -c gray

# one shader per dataset
neuroglancer -d /path/to/data.zarr/{affs,raw} -c shuffle gray

# channel 2 in red
neuroglancer -d /path/to/data.zarr/raw -c 'color:#ff0000@2'

# 16 bit labels as segmentation, with a voxel size the metadata lacks
neuroglancer -d /path/to/data.zarr/labels -t seg -vs 40,8,8

# first channel of raw in every crop
neuroglancer -d '/path/to/data.zarr/crop_*/raw[0]'

# the same, with -s
neuroglancer -d '/path/to/data.zarr/crop_*/raw' -s 0
```

Slices can be appended to a path (`raw[0]`). A path like `raw_[1345]` is read
as a slice. To use it as a glob instead (`raw_1`, `raw_3`, ...), end it with
`:`, i.e., `raw_[1345]:`.

## Python

### add_layer

```python
import neuroglancer
from funlib.persistence import open_ds
from funlib.show.neuroglancer import add_layer

viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    add_layer(s, open_ds("/path/to/data.zarr/raw"), "raw")
    add_layer(s, open_ds("/path/to/data.zarr/lsds"), "lsds", shader="rgb", rgb_channels=[3, 4, 5])
print(viewer)
```

### ScalePyramid

Passing a list of arrays (one per scale level) to `add_layer` creates a
`ScalePyramid` layer:

```python
scales = [open_ds(f"/path/to/data.zarr/raw/s{i}") for i in range(3)]
with viewer.txn() as s:
    add_layer(s, scales, "raw")
```

`ScalePyramid` can also be used directly, with one `neuroglancer.LocalVolume`
per scale level:

```python
from funlib.show.neuroglancer import ScalePyramid

layer = ScalePyramid([volume_s0, volume_s1, volume_s2])
with viewer.txn() as s:
    s.layers.append(name="raw", layer=layer)
```

### Video tool

Records a video as a script of viewer states, then renders the script to PNG
frames. Both take a function that creates a viewer with your layers:

```python
from funlib.show.neuroglancer import RenderArgs, run_edit, run_render

def create_viewer():
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        add_layer(s, open_ds("/path/to/data.zarr/raw"), "raw")
    return viewer

args = RenderArgs()
args.script = "/path/to/script.txt"
args.output_directory = "/path/to/frames"

# edit: k adds a keypoint, [ and ] step between keypoints, enter plays, q quits
run_edit(create_viewer, args)

# render the script to /path/to/frames/*.png
run_render(create_viewer, args)
```

`RenderArgs` also sets the frame size (`width`, `height`), `fps` and the
default transition `duration` between keypoints.
