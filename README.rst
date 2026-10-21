funlib.show.neuroglancer
========================

Scripts and helpers for the
`neuroglancer <https://github.com/google/neuroglancer>`_ WebGL viewer.

Currently contains:

- a layer to show multi-scale volumes from python

- a modified ``video_tool.py`` to work with ``LocalVolume``

- a convenience method to add ``funlib.persistence`` arrays to a neuroglancer
  context (``add_layer``), with a set of shaders for multi-channel data

- a ``neuroglancer`` command line tool to show zarr arrays

Usage
-----

Pass the arrays to show with ``-d``. Terminal glob syntax selects multiple
arrays::

    neuroglancer -d path/to/your/data.zarr/and/volumes/*

Groups of scale levels (``s0``, ``s1``, ... or ``0``, ``1``, ...) are shown as
a single multiscale layer.

Layer types and shaders are chosen automatically: 32 and 64 bit integer arrays
without channels are shown as segmentations, everything else as images. Images
with 3 or more channels get an RGB shader on the first three channels, images
with 2 channels an additive shader.

The options ``-c`` (shader), ``-t`` (volume type), ``-vs`` (voxel size) and
``-s`` (slices) apply to the datasets of the preceding ``-d``. ``-c`` and
``-t`` take one value for all of them, or one per dataset::

    # channels 3, 4, 5 of the LSDs as RGB, raw as grayscale with a contrast slider
    neuroglancer -d data.zarr/lsds -c rgb:3,4,5 -d data.zarr/raw -c gray

    # one shader per dataset
    neuroglancer -d data.zarr/affs data.zarr/raw -c shuffle gray

    # 16 bit labels as segmentation, with a voxel size the metadata lacks
    neuroglancer -d data.zarr/labels -t seg -vs 40,8,8

Shader specs:

==================  ==========================================================
``default``         neuroglancer's default shader
``rgb[:r,g,b]``     channels ``r,g,b`` as RGB (default ``0,1,2``)
``gray[:c]``        channel ``c`` as grayscale, with a contrast slider
``color[:C][@c]``   channel ``c`` in a constant color ``C`` (``r,g,b`` floats
                    in [0, 1] or ``#rrggbb``, default random)
``heatmap[:c]``     channel ``c`` as a jet color map
``binary[:c]``      channel ``c`` as black/white
``add``             all channels summed into RGB, a weight slider per channel
``cadd``            all channels summed into RGB in groups of three
``shuffle``         channels as R/G/B, with per-channel on/off and display
                    ranges and a seed to shuffle the colors
==================  ==========================================================

The voxel size is read from the attribute configured for
``funlib.persistence`` (``voxel_size`` by default). Arrays that lack it but
have a ``voxel_size`` or ``resolution`` attribute use that instead.

Extra optional arguments are:

- ``--no-browser``
- ``--bind-address``
- ``--port``

Slicing
^^^^^^^

We also have slicing support (This command will select only the first channel
of raw from every crop)::

    neuroglancer -d data.zarr/crop_*/raw[0]

The same slice can be given for all datasets of a ``-d`` with ``-s``::

    neuroglancer -d data.zarr/crop_*/raw -s 0

This can lead to a conflict if you want to use glob expansion to select
multiple arrays. Here it is unclear if you want to visualize the arrays
``raw_1``, ``raw_3``, ``raw_4`` and ``raw_5``, or if you want to visualize
channel ``1345`` of the array ``raw_``::

    neuroglancer -d data.zarr/raw_[1345]

A slice is assumed. To select the arrays ``raw_1``, ``raw_3``, ``raw_4`` and
``raw_5`` instead, end the glob with ``:``::

    neuroglancer -d data.zarr/raw_[1345]:

To be explicit about slicing, separate it from the glob with ``:``::

    neuroglancer -d data.zarr/raw_*:[0]
