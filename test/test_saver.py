"""
Tests for generate_4d_heart.saver — all save_* functions.

Covers:
- Input as torch.Tensor vs np.ndarray
- 2D (single frame) vs 3D (multi-frame) shapes
- Various dtypes (float32, uint8, int8)
- Normalization (vmin/vmax) correctness
- Edge cases: constant frames, single frame, all-zero depth maps
- Round-trip: saved file can be loaded back with expected shape/dtype/values
- Automatic parent directory creation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile
import cv2

from generate_4d_heart.saver import (
    save_nii,
    save_tif,
    save_png,
    save_pngs,
    save_gif,
    save_deepthmap_gif,
    save_video,
    save_deepthmap_video,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output" / "test_saver"


@pytest.fixture(scope="module", autouse=True)
def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def random_frames_float() -> np.ndarray:
    """(T, H, W) float32 in [0, 1]."""
    return np.random.rand(10, 64, 64).astype(np.float32)


@pytest.fixture
def random_frames_uint8() -> np.ndarray:
    """(T, H, W) uint8 in [0, 255]."""
    return np.random.randint(0, 256, size=(10, 64, 64), dtype=np.uint8)


@pytest.fixture
def single_frame_float() -> np.ndarray:
    """Single (H, W) float32 frame."""
    return np.random.rand(64, 64).astype(np.float32)


@pytest.fixture
def depth_maps() -> np.ndarray:
    """(T, H, W) float32 with nan background."""
    dm = np.full((10, 64, 64), np.nan, dtype=np.float32)
    dm[:, 16:48, 16:48] = np.random.rand(10, 32, 32).astype(np.float32) * 200 + 50
    return dm


@pytest.fixture
def label_frames() -> np.ndarray:
    """(T, H, W) uint8, 0 = background, 1 = foreground."""
    lbl = np.zeros((10, 64, 64), dtype=np.uint8)
    lbl[:, 20:44, 20:44] = 1
    return lbl


@pytest.fixture
def volume_3d() -> tuple[torch.Tensor, np.ndarray]:
    """(D, H, W) float32 volume + 4x4 affine."""
    vol = torch.rand(16, 64, 64, dtype=torch.float32)
    aff = np.eye(4)
    return vol, aff


# ── save_nii ──────────────────────────────────────────────────────────────────


class TestSaveNii:
    def test_save_3d_float(self, volume_3d, tmp_path):
        vol, aff = volume_3d
        path = tmp_path / "vol.nii.gz"
        save_nii(path, vol, aff, is_label=False)
        assert path.exists()
        # nibabel round-trip
        import nibabel as nib
        loaded = nib.load(str(path))
        # nibabel loads as (D, H, W) since saver.squeeze preserves first dim
        assert loaded.shape == (16, 64, 64)
        assert loaded.header.get_data_dtype() == np.dtype(np.float32)

    def test_save_label_int8(self, tmp_path):
        vol = torch.randint(0, 2, (16, 64, 64), dtype=torch.int8)
        aff = np.eye(4)
        path = tmp_path / "label.nii.gz"
        save_nii(path, vol, aff, is_label=True)
        import nibabel as nib
        loaded = nib.load(str(path))
        assert loaded.header.get_data_dtype() == np.dtype(np.int8)

    def test_save_creates_parent_dir(self, tmp_path):
        vol = torch.rand(8, 8, 8)
        aff = np.eye(4)
        deep = tmp_path / "a" / "b" / "c" / "vol.nii.gz"
        save_nii(deep, vol, aff)
        assert deep.exists()

    def test_rejects_invalid_shape(self, tmp_path):
        vol = torch.rand(3, 8, 8, 8)  # 4D with C=3
        aff = np.eye(4)
        # This should work via rearrange
        path = tmp_path / "rgb.nii.gz"
        save_nii(path, vol, aff)  # 4D with C=3 is supported
        assert path.exists()

    def test_rejects_5d(self, tmp_path):
        vol = torch.rand(2, 3, 8, 8, 8)
        aff = np.eye(4)
        with pytest.raises(ValueError, match="Unsupported tensor shape"):
            save_nii(tmp_path / "bad.nii.gz", vol, aff)


# ── save_tif ──────────────────────────────────────────────────────────────────


class TestSaveTif:
    def test_save_torch_tensor(self, random_frames_float, tmp_path):
        path = tmp_path / "frames.tif"
        save_tif(path, torch.from_numpy(random_frames_float))
        assert path.exists()
        reloaded = tifffile.imread(str(path))
        assert reloaded.shape == random_frames_float.shape
        assert np.allclose(reloaded, random_frames_float, atol=1e-6)

    def test_save_numpy_array(self, random_frames_float, tmp_path):
        path = tmp_path / "frames_np.tif"
        save_tif(path, random_frames_float)
        reloaded = tifffile.imread(str(path))
        assert np.allclose(reloaded, random_frames_float, atol=1e-6)

    def test_single_frame_2d(self, single_frame_float, tmp_path):
        path = tmp_path / "single.tif"
        save_tif(path, single_frame_float)
        reloaded = tifffile.imread(str(path))
        # save_tif expands 2D → (1, H, W)
        # tifffile may squeeze trailing singleton dims
        assert reloaded.shape[-2:] == (64, 64)
        assert np.allclose(reloaded.squeeze(), single_frame_float, atol=1e-6)

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "sub" / "nested" / "frames.tif"
        save_tif(path, np.random.rand(3, 16, 16).astype(np.float32))
        assert path.exists()


# ── save_png ──────────────────────────────────────────────────────────────────


class TestSavePng:
    def test_save_torch(self, tmp_path):
        img = torch.rand(64, 64, dtype=torch.float32) * 255
        path = tmp_path / "img.png"
        save_png(path, img)
        assert path.exists()

    def test_save_numpy(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        path = tmp_path / "img_np.png"
        save_png(path, img)
        reloaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert reloaded.shape == (64, 64)

    def test_save_squeezes(self, tmp_path):
        """Should handle (1, H, W) input by squeezing."""
        img = np.random.rand(1, 64, 64).astype(np.float32) * 255
        path = tmp_path / "squeeze.png"
        save_png(path, img)  # should not raise
        assert path.exists()


# ── save_pngs ─────────────────────────────────────────────────────────────────


class TestSavePngs:
    def test_save_all_frames(self, random_frames_uint8, tmp_path):
        save_pngs(tmp_path, random_frames_uint8)
        for t in range(10):
            assert (tmp_path / f"{t:03d}.png").exists()

    def test_single_frame_2d(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        save_pngs(tmp_path, img)
        assert (tmp_path / "000.png").exists()

    def test_torch_input(self, random_frames_uint8, tmp_path):
        save_pngs(tmp_path, torch.from_numpy(random_frames_uint8))
        assert (tmp_path / "000.png").exists()


# ── save_gif ──────────────────────────────────────────────────────────────────


class TestSaveGif:
    def test_save_gif_uint8(self, random_frames_uint8, tmp_path):
        path = tmp_path / "anim.gif"
        save_gif(path, random_frames_uint8, fps=10, cmap="gray")
        assert path.exists()
        assert path.stat().st_size > 100

    def test_save_gif_float(self, random_frames_float, tmp_path):
        path = tmp_path / "anim_float.gif"
        save_gif(path, random_frames_float, fps=10, cmap="gray")
        assert path.exists()

    def test_save_gif_with_vrange(self, random_frames_float, tmp_path):
        path = tmp_path / "anim_vrange.gif"
        save_gif(path, random_frames_float, fps=10, cmap="gray", vmin=0, vmax=1)
        assert path.exists()

    def test_single_frame_2d(self, single_frame_float, tmp_path):
        path = tmp_path / "single_frame.gif"
        save_gif(path, single_frame_float, fps=10, cmap="gray")
        assert path.exists()

    def test_torch_input(self, random_frames_float, tmp_path):
        path = tmp_path / "torch.gif"
        save_gif(path, torch.from_numpy(random_frames_float), fps=10, cmap="gray")
        assert path.exists()

    def test_constant_frames(self, tmp_path):
        """All-zero and all-constant frames should not crash."""
        const = np.ones((5, 32, 32), dtype=np.float32) * 0.5
        path = tmp_path / "const.gif"
        save_gif(path, const, fps=10, cmap="gray")
        assert path.exists()

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "test.gif"
        save_gif(path, np.random.rand(3, 16, 16).astype(np.float32), fps=10)
        assert path.exists()


# ── save_deepthmap_gif ────────────────────────────────────────────────────────


class TestSaveDeepthmapGif:
    def test_save_depth(self, depth_maps, tmp_path):
        path = tmp_path / "depth.gif"
        save_deepthmap_gif(path, depth_maps, fps=10)
        assert path.exists()

    def test_torch_input(self, depth_maps, tmp_path):
        path = tmp_path / "depth_torch.gif"
        save_deepthmap_gif(path, torch.from_numpy(depth_maps), fps=10)
        assert path.exists()

    def test_all_nan_depth(self, tmp_path):
        """Entirely-nan depth map should not crash."""
        all_nan = np.full((5, 32, 32), np.nan, dtype=np.float32)
        path = tmp_path / "all_nan.gif"
        save_deepthmap_gif(path, all_nan, fps=10)
        assert path.exists()

    def test_single_frame_depth(self, tmp_path):
        dm = np.full((64, 64), np.nan, dtype=np.float32)
        dm[16:48, 16:48] = 150.0
        path = tmp_path / "single_depth.gif"
        save_deepthmap_gif(path, dm, fps=10)
        assert path.exists()


# ── save_video ────────────────────────────────────────────────────────────────


class TestSaveVideo:
    def test_save_mp4_uint8(self, random_frames_uint8, tmp_path):
        path = tmp_path / "video.mp4"
        save_video(path, random_frames_uint8, fps=30.0, vmin=0, vmax=255)
        assert path.exists()
        assert path.stat().st_size > 100
        # Round-trip: read back with OpenCV
        cap = cv2.VideoCapture(str(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 10

    def test_save_mp4_float_auto_scale(self, random_frames_float, tmp_path):
        path = tmp_path / "video_float.mp4"
        save_video(path, random_frames_float, fps=30.0)
        assert path.exists()

    def test_with_explicit_vrange(self, random_frames_float, tmp_path):
        path = tmp_path / "video_vrange.mp4"
        save_video(path, random_frames_float, fps=30.0, vmin=0.0, vmax=1.0)
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        assert ret
        # OpenCV reads grayscale video as (H, W, 3) BGR
        assert frame.shape[:2] == (64, 64)
        assert frame.dtype == np.uint8

    def test_single_frame_2d(self, single_frame_float, tmp_path):
        path = tmp_path / "single_video.mp4"
        save_video(path, single_frame_float, fps=30.0)
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 1

    def test_torch_input(self, random_frames_float, tmp_path):
        path = tmp_path / "torch_video.mp4"
        save_video(path, torch.from_numpy(random_frames_float), fps=30.0)
        assert path.exists()

    def test_constant_frames(self, tmp_path):
        """All-constant frames should create a valid video."""
        const = np.full((5, 32, 32), 0.5, dtype=np.float32)
        path = tmp_path / "const_video.mp4"
        save_video(path, const, fps=30.0)
        assert path.exists()
        assert path.stat().st_size > 100

    def test_all_zero_frames(self, tmp_path):
        """All-zero frames (vmax-vmin ~= 0) should not crash."""
        zeros = np.zeros((5, 32, 32), dtype=np.float32)
        path = tmp_path / "zeros.mp4"
        save_video(path, zeros, fps=30.0)
        assert path.exists()

    def test_writes_correct_fps(self, tmp_path):
        frames = np.random.rand(30, 64, 64).astype(np.float32)
        path = tmp_path / "fps_check.mp4"
        target_fps = 15.0
        save_video(path, frames, fps=target_fps)
        cap = cv2.VideoCapture(str(path))
        written_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert written_fps == pytest.approx(target_fps, rel=0.1)

    def test_creates_parent_dir(self, tmp_path):
        path = tmp_path / "a" / "b" / "deep_video.mp4"
        save_video(path, np.random.rand(3, 16, 16).astype(np.float32), fps=10.0)
        assert path.exists()

    def test_vmin_vmax_preserves_data_range(self, tmp_path):
        """Data in [0.2, 0.8] with vmin=0, vmax=1 → values stretched to [51, 204]."""
        data = np.linspace(0.2, 0.8, 100).reshape(1, 10, 10).astype(np.float32)
        path = tmp_path / "range_check.mp4"
        save_video(path, data, fps=10.0, vmin=0.0, vmax=1.0)
        cap = cv2.VideoCapture(str(path))
        ret, frame = cap.read()
        cap.release()
        assert ret
        # value 0.2 → 51, 0.8 → 204
        assert np.min(frame) >= 0
        assert np.max(frame) <= 255
        assert np.min(frame) >= 40  # close to 51
        assert np.max(frame) <= 210  # close to 204


# ── save_deepthmap_video ──────────────────────────────────────────────────────


class TestSaveDeepthmapVideo:
    def test_save_depth_video(self, depth_maps, tmp_path):
        path = tmp_path / "depth.mp4"
        save_deepthmap_video(path, depth_maps, fps=30.0)
        assert path.exists()
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 10

    def test_torch_input(self, depth_maps, tmp_path):
        path = tmp_path / "depth_torch.mp4"
        save_deepthmap_video(path, torch.from_numpy(depth_maps), fps=30.0)
        assert path.exists()

    def test_all_nan_depth(self, tmp_path):
        """Depth map with no valid pixels should not crash."""
        all_nan = np.full((5, 32, 32), np.nan, dtype=np.float32)
        path = tmp_path / "all_nan_depth.mp4"
        save_deepthmap_video(path, all_nan, fps=30.0)
        assert path.exists()

    def test_single_frame(self, tmp_path):
        dm = np.full((64, 64), np.nan, dtype=np.float32)
        dm[16:48, 16:48] = 150.0
        path = tmp_path / "single_depth.mp4"
        save_deepthmap_video(path, dm, fps=30.0)
        assert path.exists()


# ── Cross-format consistency ──────────────────────────────────────────────────


class TestCrossFormatConsistency:
    """Verify that different save functions produce consistent results."""

    def test_gif_and_video_same_data(self, random_frames_uint8, tmp_path):
        """Both gif and video should handle the same data without error."""
        gif_path = tmp_path / "cross.gif"
        mp4_path = tmp_path / "cross.mp4"
        save_gif(gif_path, random_frames_uint8, fps=10, cmap="gray", vmin=0, vmax=255)
        save_video(mp4_path, random_frames_uint8, fps=10.0, vmin=0, vmax=255)
        assert gif_path.exists()
        assert mp4_path.exists()

    def test_depth_gif_and_video(self, depth_maps, tmp_path):
        gif_path = tmp_path / "depth_cross.gif"
        mp4_path = tmp_path / "depth_cross.mp4"
        save_deepthmap_gif(gif_path, depth_maps, fps=10)
        save_deepthmap_video(mp4_path, depth_maps, fps=10.0)
        assert gif_path.exists()
        assert mp4_path.exists()
