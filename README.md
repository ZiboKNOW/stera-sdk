<h1 align="center">
  <img alt="" src="assets/logo_light.png" height="32" align="center" style="vertical-align: middle;">
   Stera SDK
</h1>
<p align="center">by <a href="https://fpvlabs.ai">FPV Labs</a> (<a href="https://fpvlabs.ai">fpvlabs.ai</a>)</p>

<p align="center">
  <a href="https://pypi.org/project/stera-sdk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/stera-sdk?color=4c8bf5"></a>
  <a href="https://pypi.org/project/stera-sdk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/stera-sdk?color=4c8bf5"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

<a href="https://fpvlabs.ai/app"><img src="assets/app-qr.svg" alt="Scan to get the iOS app" width="96" align="right"></a>

`stera-sdk` is the Python SDK behind *Stera* ([website](https://fpvlabs.ai/stera) · [iOS app](https://fpvlabs.ai/app)).

Load MCAP recordings, run swappable hand, face, and skeleton models behind one schema,
and visualize or export clean episodes, ready for embodied AI, VLAs, and world models.

### Quick Start

```python
import stera
from stera.data import MCAPReader
from stera.models import HandTracker, FaceBlurrer
from stera.viz import Visualizer
from stera.processing import MeshRefiner
from stera.eval import Evaluate

session = MCAPReader("recording.mcap")
hands   = HandTracker(model="mediapipe")
blur    = FaceBlurrer(model="mediapipe")
viz     = Visualizer(session, map_3d="auto")

for frame in session.frames():
    blurred = blur.blur(frame)
    poses   = hands.detect_hands(frame)
    session.add_rgb_frame(frame.index, blurred)
    session.add_hand_pose(frame.index, poses)
    viz.log_frame(frame, hands=poses)

mesh = MeshRefiner(session).refine()           # clean + densify + colorize
session.export("episodes/run_01", visualizer=viz, mesh=mesh)
Evaluate(session).show()                       # interactive HTML QC report
```

<p align="center">
  <img src="assets/hero.png" alt="Stera SDK">
</p>

## Getting started

```bash
pip install "stera-sdk[all]"
```

## HaWoR support in this fork

This fork adds `HandTracker(model="hawor")` and a public
`HandTracker.detect_sequence(...)` adapter. The upstream SDK at `df6a5ec`
supports WiLoR, MediaPipe, and HaMeR, but does not include the HaWoR backend.
The adapter also exposes WiLoR's existing sequence method.

Install this checkout into an environment already configured for
[HaWoR](https://github.com/ThunderVVV/HaWoR), including its PyTorch/CUDA,
Ultralytics, and MANO dependencies:

```bash
python -m pip install --no-deps -e /path/to/stera-sdk
```

For a new environment, install the SDK's core dependencies and HaWoR's
dependencies first; `--no-deps` only preserves an existing environment.
Model source and weights are external and are not bundled or downloaded by
the SDK. The default `model_path` layout is:

```text
HaWoR/
  hawor/                                 # upstream Python package
  lib/                                   # upstream model and dataset code
  weights/hawor/checkpoints/hawor.ckpt
  weights/hawor/model_config.yaml
  weights/external/detector.pt
  _DATA/data/mano/MANO_RIGHT.pkl
  _DATA/data/mano_mean_params.npz
```

Obtain the weights and licensed MANO assets using the upstream installation
instructions. This camera-space adapter does not require SLAM or infiller
weights. It calls the model API inspected at
[HaWoR commit 66c7d410](https://github.com/ThunderVVV/HaWoR/tree/66c7d4108d58a716deccd192cb7645170cdc7bd7).

```python
import numpy as np
from stera.models import HandTracker

tracker = HandTracker(
    model="hawor",
    model_path="/path/to/HaWoR",
    yolo_conf=0.2,
)
rgb_K = np.load("calibrations/rgb_K.npy")
sequence = tracker.detect_sequence("rgb.mp4", intrinsics=rgb_K)
# Or pass a directory of naturally sorted JPEG/PNG frames.
# Missing detections remain empty lists; indices always match input frames.
print(tracker._backend.last_stats)
```

Use RGB intrinsics in the input image orientation, not depth intrinsics.
The backend keeps the highest-confidence detection for each hand side per
frame and runs temporal inference on contiguous detections. It does not
fill detection gaps, resample video frame rates, or anchor wrists to LiDAR.
Returned `HandPose` keypoints use the RGB optical camera frame in metres.
MANO vertices remain local posed geometry, with camera translation stored
separately, as in the WiLoR export. The existing episode exporter writes
these arrays and MANO parameters without a schema change. HaWoR's model takes
a scalar focal length (the mean of `fx` and `fy`); camera translation and exported 2D
projections use the supplied RGB calibration. `CUDA_VISIBLE_DEVICES` from
the caller is respected, so batch workers can select their own GPUs.

Run the CPU contract tests without model weights:

```bash
PYTHONPATH=src python -m pytest -q tests/test_hawor.py
```

These tests use simulated detector/model outputs. Before a full batch run,
validate a short sequence with the actual CUDA environment and checkpoints;
the local tests do not establish GPU compatibility or reconstruction accuracy.

## Documentation

- 📚 [Docs](https://fpvlabs.ai/stera/docs)
- 🚀 [Quickstart](https://fpvlabs.ai/stera/docs/process/get-started/quickstart)
- ⚙️ [API reference](https://fpvlabs.ai/stera/docs/process/api)
- 🛠️ [Installation](https://fpvlabs.ai/stera/docs/process/get-started/installation)

## License

Apache 2.0 © [FPV Labs](https://fpvlabs.ai)
