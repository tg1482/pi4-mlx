# Pi4 on Apple silicon (MLX)

A fork of [yyfz/Pi4](https://github.com/yyfz/Pi4) that runs Pi4's geometry on a Mac.

**What changed**
- `pi4/mlx/`: Pi4 geometry (points, confidence, cameras, metric scale) ported to MLX. It loads the original checkpoint.
- `pi4/models/sparse.py`: plain-PyTorch stand-ins for the three CUDA-only FlexGEMM layers, so the PyTorch model also
  runs without CUDA.
- Tracking and motion masks are not ported.

**Checked**: against the PyTorch model in fp32, camera poses match to 1e-6, metric scale to 1e-5, depth to a median of
0.03 % (`tests/`). On an M2 Pro: ~1.2 s per frame at 512x384.

```python
import mlx.core as mx
from pi4.mlx import Pi4

model = Pi4.from_pretrained("yyfz233/Pi4", dtype=mx.float32, refiner_dtype=mx.float16)
out = model(images)  # images: [B, T, H, W, 3] in [0, 1], H and W multiples of 16
```

Install: `pip install -e ".[mlx]"`. Licence unchanged (CC BY-NC 4.0).

---

<div align="center">

<h2>🌌 π⁴: A Foundation Model for 4D Scene Geometry and Motion</h2>

<p>Physical Intelligence Team, Shanghai Artificial Intelligence Laboratory</p>

<p>
  <a href="https://yyfz.github.io/pi4"><img src="docs/assets/project-page-badge.svg" alt="Project Page" height="36"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2611.23144"><img src="docs/assets/paper-badge.svg" alt="Paper" height="36"></a>
  &nbsp;
  <a href="https://weiguangzhao.github.io/GameRealm/"><img src="docs/assets/dataset-badge.svg" alt="GameRealm Dataset" height="36"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/yyfz233/Pi4"><img src="docs/assets/huggingface-demo-badge.svg" alt="Hugging Face Demo" height="36"></a>
</p>

<p>
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="#python-api">Python API</a> &nbsp;·&nbsp;
  <a href="docs/inference.md">Inference Guide</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/yyfz233/Pi4">Weights</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

<p>
  <img src="docs/assets/figure_skating.gif" alt="Figure Skating: joint geometry and motion with synchronized global and RGB insets." width="49%">
  <img src="docs/assets/waterfall.gif" alt="Waterfall: joint geometry and motion with synchronized global and RGB insets." width="49%">
</p>

<p><sub>Geometry and motion in a single network forward pass, with synchronized global and RGB views.</sub></p>

</div>

## Overview

**From pixels to a scene in motion.** Pi4 brings geometric reconstruction and
motion estimation into a single model. It accepts single images, unordered image
collections, and videos without supplied camera calibration, building a shared
3D scene representation in one forward pass.

- **One compact model.** Approximately **0.5B parameters** for joint geometry
  and motion.
- **Strong across tasks.** The best overall average rank among the compared
  methods across the paper's **16 geometry evaluation settings**, with competitive
  results in single- and multiview 2D/3D tracking and moving-object segmentation.
- **Thousands of images, one forward pass.** Joint reconstruction and tracking
  from **thousands of input images**, despite training with at most **64 views**.

## Quickstart

### 1. Install

Use **Linux x86-64**, **Python 3.11**, an **NVIDIA GPU with BF16 support**, and a
CUDA 12.6-compatible NVIDIA driver. Run the commands from the repository root:

```bash
conda create -n pi4 python=3.11 -y
conda activate pi4
python -m pip install --upgrade pip
python -m pip install -r requirements-cu126.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print(torch.__version__, torch.version.cuda)"
```

This installs **PyTorch 2.12.1 + CUDA 12.6**, **Triton 3.7.1**, the pinned
FlexGEMM backend, and Pi4 with its image/video dependencies. Git and a C compiler
(such as GCC) are required for the backend and Triton's runtime compilation.
The CUDA runtime is supplied by the PyTorch dependencies; a separate toolkit
installation is not needed. See the
[installation notes](docs/inference.md#install) for checks and other CUDA builds.

**Weights:** the example automatically downloads and caches the checkpoint from
[yyfz233/Pi4](https://huggingface.co/yyfz233/Pi4). If the automatic download is
slow, [download model.safetensors manually](https://huggingface.co/yyfz233/Pi4/resolve/main/model.safetensors)
and pass its local path with `--weights /path/to/model.safetensors`.

### 2. Reconstruct and Track

**Figure Skating — stride 1, first 150 frames:**

```bash
python example.py examples/figure_skating.mp4 \
  --output outputs/quickstart \
  --query-frame 0 --stride 1 --max-frames 150 --viewer
```

This processes **150 frames at 512 × 288** together in one forward pass,
covering the first approximately **12 seconds** of the 60-second source.
The full video is long, so this quickstart sets `--max-frames 150` to reduce
runtime and GPU memory use. This is an optional limit, not a model requirement;
omitting it processes every frame remaining after `--stride` sampling.
The GIFs above are short previews of the paper demos, not the quickstart output.
Omit `--query-frame` for geometry-only inference; use a new output directory
for each run.

The 150-frame configuration measured **12.03 GiB peak GPU memory** on an
A800 80 GB. For smaller GPUs, reduce the frame count; the measured 25-frame
configuration uses `--stride 1 --max-frames 25`. These are reference
measurements, not hardware guarantees. See the
[memory and timing table](docs/inference.md#quickstart).

**Kart — full clip, stride 2:**

```bash
python example.py examples/kart.mp4 \
  --output outputs/kart \
  --query-frame 0 --stride 2 --viewer
```

**Cycling — full clip, stride 2:**

```bash
python example.py examples/cycling.mp4 \
  --output outputs/cycling \
  --query-frame 0 --stride 2 --viewer
```

All three clips are stored at their inference resolution: **512 × 288** for
Figure Skating and Kart, **288 × 512** for Cycling. The videos retain every
source frame; `--stride` samples frames at loading time. Each command runs its sampled sequence
in one forward pass; the Cycling example has more frames than the 150-frame
memory measurement above.

**Reproduce the full Figure Skating demo from the paper and project page:**
keep stride 1 and omit `--max-frames` to process all **750 frames** together.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python example.py examples/figure_skating.mp4 \
  --output outputs/figure_skating_full \
  --query-frame 0 --stride 1 --viewer
```

The full run needs substantially more GPU memory; see the
[full-sequence measurements](docs/inference.md#reproduce-the-project-page-demo).

### 3. Explore in Your Browser

```bash
python viewer/serve.py outputs/quickstart --port 8080
```

Open the address printed by the server. Orbit the reconstruction, inspect
camera motion and point tracks, or play the sequence alongside the RGB input.
Viewing exported results requires **no GPU or model weights**; the server uses
the Python standard library and bundled Three.js, with no Node build or CDN.
Forward port 8080 if inference runs on a remote machine.

Replace `outputs/quickstart` with `outputs/kart`, `outputs/cycling`, or
`outputs/figure_skating_full` to view another run. See
[example videos](examples/README.md) for clip details.

## Python API

```python
import torch
from pi4 import Pi4
from pi4.io import load_images

model = Pi4.from_pretrained("yyfz233/Pi4").cuda().eval()
frames, metadata = load_images(
    "examples/figure_skating.mp4", stride=1, max_frames=25, max_size=512
)
images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)
images = images.float().div(255).unsqueeze(0).cuda()

prediction = model(images, query_frame=0)
```

Inputs are floating-point RGB in `[0, 1]`, shaped `[B, T, 3, H, W]`, with
height and width divisible by 16. Normalization, inference precision, and
gradient disabling are handled internally. Tracking requires at least two
images; omit `query_frame` to return geometry only.

| Output | Keys |
| :--- | :--- |
| Scene geometry | `local_points`, `camera_poses`, `metric_scale`, `confidence` |
| Dense correspondences | `tracks_2d`, `tracks_3d`, `query_frame` |
| Motion and reliability | `dynamic_mask`, `visibility`, `tracking_confidence` |

`local_points` and camera translations use the model's native scale;
`tracks_3d` are already in predicted metric world coordinates. See the
[output contract](docs/inference.md#outputs) for tensor shapes, coordinate
conventions, and visibility semantics.

`from_pretrained` accepts a local Safetensors file, a directory containing
`model.safetensors`, or a Hugging Face repository ID with optional `revision=`.
Calling it without arguments loads `yyfz233/Pi4`. Loading is strict; no
additional pretrained models or configuration files are required.

## Go Further

- [Inference guide](docs/inference.md) — full API, export formats, track loading, and viewer controls.
- [Full-length Figure Skating demo](docs/inference.md#reproduce-the-project-page-demo) — all 750 frames in one forward pass, with measured memory requirements.
- [Example videos](examples/README.md) — figure skating, indoor karting, and cycling.

## Citation

If you use Pi4 in your research, please cite:

```bibtex
@misc{wang2026pi4,
  title         = {{$\pi^4$}: A Foundation Model for {4D} Scene Geometry and Motion},
  author        = {Wang, Yifan and Zhao, Weiguang and Yang, Ni and Guo, Haoyu and Dong, Junting and Zhang, Weinan and Shen, Chunhua},
  year          = {2026},
  eprint        = {2611.23144},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2611.23144}
}
```

## License

| Component | License | Commercial Use |
| :--- | :--- | :--- |
| Original Pi4 code | [CC BY-NC 4.0](LICENSE) | Not permitted |
| Pi4 model weights | [CC BY-NC 4.0](LICENSE_MODEL) | Not permitted |

Third-party components retain their upstream licenses. See [NOTICE](NOTICE) and [upstream licenses](licenses/).
