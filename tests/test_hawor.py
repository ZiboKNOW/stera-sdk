"""CPU contract tests; HaWoR weights and CUDA are not needed for these tests."""

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import h5py
import numpy as np
import pytest

from stera.data.export import write_episode
from stera.models import HandTracker


@pytest.fixture
def camera_matrix():
    # Unequal focal lengths and a principal point away from the image centre
    # expose coordinate assumptions that a centred, square camera would hide.
    return np.array([[500.0, 0.0, 27.0], [0.0, 700.0, 19.0], [0.0, 0.0, 1.0]])


def test_unified_tracker_routes_configuration_and_sequence(monkeypatch, camera_matrix):
    from stera.models import hawor

    backend = Mock()
    factory = Mock(return_value=backend)
    monkeypatch.setattr(hawor, "HaWoRHandTracker", factory)
    tracker = HandTracker(model="HaWoR", model_path="/models/HaWoR", yolo_conf=0.25)
    config = factory.call_args.args[0]
    assert config.hawor_dir == "/models/HaWoR"
    assert config.yolo_conf == 0.25
    backend.load.assert_called_once_with()

    result = [[], [object()], []]
    backend.detect_sequence.return_value = result
    assert tracker.detect_sequence("rgb.mp4", intrinsics=camera_matrix) is result
    backend.detect_sequence.assert_called_once_with("rgb.mp4", intrinsics=camera_matrix)


def test_unknown_hawor_option_fails_before_model_load(monkeypatch):
    from stera.models import hawor

    factory = Mock()
    monkeypatch.setattr(hawor, "HaWoRHandTracker", factory)
    with pytest.raises(TypeError, match="yolo_conff"):
        HandTracker(model="hawor", yolo_conff=0.2)
    factory.assert_not_called()


@pytest.mark.parametrize("model, expected", [("hawor", "rgb_K"), ("wilor", "depth_K")])
def test_synced_frame_intrinsics_follow_backend(model, expected, camera_matrix):
    # Avoid real model loading: this is a test of the public dispatcher.
    tracker = object.__new__(HandTracker)
    tracker.model_name = model
    tracker._backend = Mock()
    frame = SimpleNamespace(
        rgb=np.zeros((48, 64, 3), dtype=np.uint8),
        depth=np.zeros((12, 16), dtype=np.uint16),
        rgb_K=camera_matrix,
        depth_K=np.diag([100.0, 100.0, 1.0]),
    )
    tracker.detect_hands(frame)
    call = tracker._backend.detect_hands.call_args
    assert call.args[0] is frame.rgb
    assert call.kwargs["intrinsics"] is getattr(frame, expected)

    override = camera_matrix.copy()
    tracker.detect_hands(frame, intrinsics=override)
    assert tracker._backend.detect_hands.call_args.kwargs["intrinsics"] is override


def test_frame_only_backend_rejects_sequence_explicitly():
    tracker = object.__new__(HandTracker)
    tracker.model_name = "mediapipe"
    tracker._backend = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="mediapipe.*detect_sequence"):
        tracker.detect_sequence("rgb.mp4")


def _write_images(folder, names):
    folder.mkdir()
    for idx, name in enumerate(names):
        image = np.full((48, 64, 3), idx * 40, dtype=np.uint8)
        assert cv2.imwrite(str(folder / name), image)
    return folder


def _export_fake_session(tmp_path, sequence):
    """Exercise the real SDK HDF5 writer with a tiny in-memory session."""
    frames = [
        SimpleNamespace(rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=None, timestamp=10+i/12)
        for i in range(len(sequence))
    ]
    session = SimpleNamespace(
        rgb_intrinsics=None,
        depth_intrinsics=None,
        R_optical_to_link=None,
        num_rgb_frames=len(frames),
        duration=len(frames)/12,
        frames=lambda: iter(frames),
        mesh=lambda: None,
        all_camera_poses=list,
        all_imu_samples=list,
        hand_poses={i: hands for i, hands in enumerate(sequence) if hands},
    )
    write_episode(session, tmp_path, skip_rgb_mp4=True, skip_thumbnail=True)
    return h5py.File(tmp_path / "annotation.hdf5", "r")


@pytest.mark.parametrize(
    "intrinsics",
    [
        None,
        np.ones((2, 3)),
        np.diag([0.0, 700.0, 1.0]),
        np.diag([500.0, -1.0, 1.0]),
        np.diag([np.nan, 700.0, 1.0]),
        np.diag([500.0, np.inf, 1.0]),
    ],
)
def test_invalid_rgb_intrinsics_rejected(intrinsics):
    from stera.models.hawor.tracker import _validate_intrinsics

    with pytest.raises(ValueError):
        _validate_intrinsics(intrinsics)


def test_camera_translation_uses_both_focals_and_principal_point(camera_matrix):
    from stera.models.hawor.tracker import _camera_translation

    boxes = np.array([[10.0, 12.0, 30.0, 42.0], [30.0, 0.0, 50.0, 20.0]])
    pred_cam = np.array([[2.0, 0.1, -0.2], [4.0, -0.3, 0.4]])
    translated = _camera_translation(pred_cam, boxes, camera_matrix)
    # Independent numeric expectations: HaWoR's crop dilation is 1.2 and
    # its scalar focal estimate uses the mean of the two RGB focal lengths.
    z0 = 1200.0 / (30.0 * 1.2 * 2.0)
    z1 = 1200.0 / (20.0 * 1.2 * 4.0)
    expected = np.array([
        [0.1 - 7.0 * z0 / 500.0, -0.2 + 8.0 * z0 / 700.0, z0],
        [-0.3 + 13.0 * z1 / 500.0, 0.4 - 9.0 * z1 / 700.0, z1],
    ])
    np.testing.assert_allclose(translated, expected, rtol=1e-6)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_invalid_camera_scale_is_not_exported(scale, camera_matrix):
    from stera.models.hawor.tracker import _camera_translation

    with pytest.raises(ValueError):
        _camera_translation([[scale, 0.1, 0.2]], [[10, 12, 30, 42]], camera_matrix)


def test_missing_hawor_checkout_reports_installation_path(tmp_path):
    from stera.models.hawor import HaWoRConfig, HaWoRHandTracker

    missing = tmp_path / "not-installed"
    tracker = HaWoRHandTracker(HaWoRConfig(hawor_dir=str(missing)))
    with pytest.raises(FileNotFoundError, match="not-installed"):
        tracker.load()


def test_missing_checkpoint_reports_asset_before_importing_torch(tmp_path):
    from stera.models.hawor import HaWoRConfig, HaWoRHandTracker

    source = tmp_path / "lib" / "models" / "hawor.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Source marker; this file must never be imported.\n")
    tracker = HaWoRHandTracker(HaWoRConfig(hawor_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError, match="hawor.ckpt"):
        tracker.load()


def _inference_result(offset=0.0):
    local_joints = np.arange(63, dtype=np.float32).reshape(21, 3) / 1000
    cam_t = np.array([0.1 + offset, -0.2, 1.5], dtype=np.float32)
    return {
        "joints": local_joints + cam_t,
        "vertices": np.arange(778 * 3, dtype=np.float32).reshape(778, 3) / 10000,
        "rotmat": np.repeat(np.eye(3, dtype=np.float32)[None], 16, axis=0),
        "betas": np.arange(10, dtype=np.float32) / 10,
        "pred_cam": np.array([2.0, 0.1, -0.2], dtype=np.float32),
        "cam_t": cam_t,
    }


@pytest.fixture
def unloaded_tracker(monkeypatch):
    from stera.models.hawor import HaWoRConfig, HaWoRHandTracker

    tracker = HaWoRHandTracker(HaWoRConfig())
    monkeypatch.setattr(tracker, "load", Mock())
    return tracker


def test_sequence_keeps_empty_frames_and_natural_order(
    tmp_path, monkeypatch, unloaded_tracker, camera_matrix
):
    folder = _write_images(tmp_path / "images", ["10.jpg", "2.jpg", "1.jpg"])
    box = np.array([10.0, 12.0, 30.0, 42.0])
    captured = []

    def detect(imgfiles):
        assert [p.name for p in map(Path, imgfiles)] == [
            "1.jpg", "2.jpg", "10.jpg"
        ]
        return [{"left": (box, 0.8)}, {}, {"left": (box, 0.9), "right": (box, 0.7)}]

    def infer(imgfiles, boxes, side, intrinsics, image_size):
        captured.append((list(imgfiles), side))
        np.testing.assert_array_equal(intrinsics, camera_matrix)
        return [_inference_result(i / 10) for i in range(len(imgfiles))]

    monkeypatch.setattr(unloaded_tracker, "_detect_boxes", detect)
    monkeypatch.setattr(unloaded_tracker, "_infer_segment", infer)
    sequence = unloaded_tracker.detect_sequence(folder, intrinsics=camera_matrix)
    assert len(sequence) == 3
    assert sequence[1] == []
    assert [hand.hand_side for hand in sequence[0]] == ["left"]
    assert {hand.hand_side for hand in sequence[2]} == {"left", "right"}
    assert all(hand.frame_idx == idx for idx, hands in enumerate(sequence) for hand in hands)
    # An absent frame splits a temporal segment instead of joining two
    # nonconsecutive images as if motion had been observed in between.
    assert len(captured) == 3
    assert all(len(paths) == 1 for paths, _ in captured)


def test_video_decode_preserves_original_frame_count(
    tmp_path, monkeypatch, unloaded_tracker, camera_matrix
):
    video = tmp_path / "12fps.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
    assert writer.isOpened(), "OpenCV must support MP4 output for the decode regression test"
    try:
        for idx in range(7):
            writer.write(np.full((48, 64, 3), idx * 30, dtype=np.uint8))
    finally:
        writer.release()
    captured_means = []

    def detect(imgfiles):
        captured_means.extend(float(cv2.imread(str(path)).mean()) for path in imgfiles)
        return [{} for _ in imgfiles]

    inference = Mock(side_effect=AssertionError("Empty detections must not run HaWoR"))
    monkeypatch.setattr(unloaded_tracker, "_detect_boxes", detect)
    monkeypatch.setattr(unloaded_tracker, "_infer_segment", inference)
    sequence = unloaded_tracker.detect_sequence(video, intrinsics=camera_matrix)
    assert sequence == [[] for _ in range(7)]
    np.testing.assert_allclose(captured_means, np.arange(7) * 30, atol=5)
    inference.assert_not_called()


def test_hand_pose_and_export_preserve_numeric_mano_fields(tmp_path, camera_matrix):
    from stera.models.hawor.tracker import _to_hand_pose

    values = _inference_result()
    hand = _to_hand_pose(
        values["joints"], values["vertices"], values["rotmat"], values["betas"],
        values["pred_cam"], values["cam_t"], camera_matrix, 0.85, "left", 1,
    )
    assert hand._backend == "hawor"
    assert hand.hand_side == "left"
    assert hand.frame_idx == 1
    np.testing.assert_allclose([kp.as_array() for kp in hand.all_keypoints], values["joints"])
    np.testing.assert_allclose(hand.fingers["thumb"][0].as_array(), values["joints"][1])
    np.testing.assert_allclose(hand.fingers["pinky"][-1].as_array(), values["joints"][20])
    expected_2d = values["joints"][:, :2] / values["joints"][:, 2:]
    expected_2d = expected_2d * [500.0, 700.0] + [27.0, 19.0]

    with _export_fake_session(tmp_path / "episode", [[], [hand], []]) as h5f:
        hands = h5f["hand-pose"]
        assert hands.attrs["backend"] == "hawor"
        assert hands.attrs["coord_frame"] == "camera_3d"
        assert hands["left_joints"].shape == (3, 21, 3)
        np.testing.assert_allclose(hands["left_joints"][1], values["joints"])
        assert np.isnan(hands["left_joints"][[0, 2]]).all()
        assert np.isnan(hands["right_joints"][:]).all()
        np.testing.assert_array_equal(hands["left_valid"][:], [False, True, False])
        np.testing.assert_array_equal(hands["right_valid"][:], [False, False, False])
        np.testing.assert_allclose(hands["left_confidence"][:], [0, 0.85, 0])
        np.testing.assert_allclose(hands["timestamps"][:], [10, 10 + 1/12, 10 + 2/12])
        extras = {
            "mano_vertices": values["vertices"],
            "mano_global_orient": values["rotmat"][:1],
            "mano_hand_pose": values["rotmat"][1:],
            "mano_betas": values["betas"],
            "pred_cam": values["pred_cam"],
            "pred_cam_t": values["cam_t"],
            "cam_t": values["cam_t"],
            "focal_length": np.array([500.0, 700.0]),
            "kpts_2d_rgb": expected_2d,
        }
        for name, expected in extras.items():
            dataset = hands[f"left_{name}"]
            assert dataset.shape == (3,) + expected.shape
            np.testing.assert_allclose(dataset[1], expected, rtol=1e-6)
            assert np.isnan(dataset[[0, 2]]).all()
        assert hands["left_mano_vertices"].compression == "gzip"


class _ArrayTensor:
    """Minimal tensor protocol for exercising the adapter without PyTorch."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def reshape(self, *shape):
        return _ArrayTensor(self.array.reshape(*shape))

    def __getitem__(self, key):
        return _ArrayTensor(self.array[key])

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self.array


@pytest.mark.parametrize("side", ["left", "right"])
def test_mano_inference_handedness_and_calibrated_camera(
    side, unloaded_tracker, camera_matrix
):
    local_joints = np.tile([0.1, 0.2, 0.3], (1, 21, 1)).astype(np.float32)
    local_vertices = np.tile([0.4, 0.5, 0.6], (1, 778, 1)).astype(np.float32)
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    rotations = np.tile(quarter_turn, (1, 16, 1, 1))
    model = SimpleNamespace(
        inference=Mock(return_value={
            "pred_rotmat": _ArrayTensor(rotations),
            "pred_shape": _ArrayTensor(np.zeros((1, 10), dtype=np.float32)),
            # upstream inference already unflips tx for left hands.
            "pred_cam": _ArrayTensor([[2.0, 0.1, -0.2]]),
            "pred_trans": _ArrayTensor([[999.0, 999.0, 999.0]]),
        }),
        mano=SimpleNamespace(query=Mock(return_value=SimpleNamespace(
            joints=_ArrayTensor(local_joints), vertices=_ArrayTensor(local_vertices),
        ))),
    )
    unloaded_tracker._model = model
    unloaded_tracker._torch = SimpleNamespace(no_grad=nullcontext)
    unloaded_tracker._device = "cpu"
    predictions = unloaded_tracker._infer_segment(
        ["frame.png"], np.array([[10.0, 12.0, 30.0, 42.0]]),
        side, camera_matrix, (64, 48),
    )
    call = model.inference.call_args
    assert call.kwargs["img_center"] == ([36.0, 19.0] if side == "left" else [27.0, 19.0])
    assert call.kwargs["do_flip"] is (side == "left")
    assert call.kwargs["img_focal"] == 600.0
    prediction = predictions[0]
    z = 1200 / 72
    cam_t = np.array([0.1 - 7*z/500, -0.2 + 8*z/700, z])
    np.testing.assert_allclose(prediction["cam_t"], cam_t, rtol=1e-6)
    expected_local = [-0.1, 0.2, 0.3] if side == "left" else [0.1, 0.2, 0.3]
    expected_vertices = [-0.4, 0.5, 0.6] if side == "left" else [0.4, 0.5, 0.6]
    expected_rotation = quarter_turn.T if side == "left" else quarter_turn
    np.testing.assert_allclose(
        prediction["joints"], np.tile(expected_local + cam_t, (21, 1)), rtol=1e-6,
    )
    np.testing.assert_allclose(prediction["vertices"], np.tile(expected_vertices, (778, 1)))
    np.testing.assert_array_equal(prediction["rotmat"], np.tile(expected_rotation, (16, 1, 1)))
    # Neither the upstream camera nor adapter handedness operations mutate
    # the original local MANO outputs supplied by the model.
    np.testing.assert_array_equal(local_vertices[0, 0], np.array([0.4, 0.5, 0.6], np.float32))


def test_detector_uses_device_object_and_best_hand_without_changing_gpu_assignment(
    tmp_path, monkeypatch, unloaded_tracker
):
    folder = _write_images(tmp_path / "images", ["0.png"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    device_marker = object()
    device_factory = Mock(return_value=device_marker)
    unloaded_tracker._torch = SimpleNamespace(device=device_factory)
    unloaded_tracker._device = "cuda:0"
    boxes = np.array([
        [1, 2, 11, 12], [3, 4, 13, 14], [5, 6, 15, 16],
        [7, 8, 17, 18], [10, 10, 9, 9],
    ], dtype=np.float32)

    def predict(image, **kwargs):
        assert image.shape == (48, 64, 3)
        # A torch.device bypasses Ultralytics' string select_device path,
        # which can rewrite a worker's existing CUDA_VISIBLE_DEVICES mask.
        assert kwargs["device"] is device_marker
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        return [SimpleNamespace(boxes=SimpleNamespace(
            xyxy=_ArrayTensor(boxes),
            cls=_ArrayTensor([0, 0, 1, 2, 1]),
            conf=_ArrayTensor([0.3, 0.8, 0.7, 0.99, 0.99]),
        ))]

    unloaded_tracker._detector = SimpleNamespace(predict=Mock(side_effect=predict))
    detections = unloaded_tracker._detect_boxes([folder / "0.png"])
    assert len(detections) == 1
    assert set(detections[0]) == {"left", "right"}
    np.testing.assert_array_equal(detections[0]["left"][0], boxes[1])
    np.testing.assert_array_equal(detections[0]["right"][0], boxes[2])
    assert detections[0]["left"][1] == pytest.approx(0.8)
    assert detections[0]["right"][1] == pytest.approx(0.7)
    device_factory.assert_called_once_with("cuda:0")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
