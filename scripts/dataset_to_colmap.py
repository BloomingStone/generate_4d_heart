from __future__ import annotations

import argparse
import json
import math
import struct
import shutil
from pathlib import Path

import numpy as np


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
	"""Return rotation matrices for one axis with batch angles."""
	cos = np.cos(angle)
	sin = np.sin(angle)
	one = np.ones_like(angle)
	zero = np.zeros_like(angle)

	if axis == "X":
		r_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
	elif axis == "Y":
		r_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
	elif axis == "Z":
		r_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
	else:
		raise ValueError(f"Invalid axis: {axis}")

	return np.stack(r_flat, axis=-1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str) -> np.ndarray:
	"""Convert Euler angles (radian) to rotation matrices, shape (..., 3, 3)."""
	if euler_angles.ndim == 1:
		euler_angles = euler_angles[None, :]
	if euler_angles.shape[-1] != 3:
		raise ValueError("Euler angles must have shape (..., 3)")
	if len(convention) != 3:
		raise ValueError("Convention must have 3 letters")
	if convention[1] in (convention[0], convention[2]):
		raise ValueError(f"Invalid convention {convention}")

	matrices = [
		_axis_angle_rotation(c, e)
		for c, e in zip(convention, np.moveaxis(euler_angles, -1, 0))
	]
	return matrices[0] @ matrices[1] @ matrices[2]


def rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
	"""Convert world-to-camera rotation matrix to COLMAP qvec: qw qx qy qz."""
	# Equivalent to COLMAP's read_write_model.py implementation.
	k = np.array(
		[
			[r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[1, 2] - r[2, 1]],
			[r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[2, 0] - r[0, 2]],
			[r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[0, 1] - r[1, 0]],
			[r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]],
		],
		dtype=np.float64,
	) / 3.0
	eigvals, eigvecs = np.linalg.eigh(k)
	qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
	if qvec[0] < 0:
		qvec = -qvec
	return qvec


def build_w2c_from_json(meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Build batched R,t (world->camera) and intrinsics from rotate_dsa metadata."""
	frames = meta["frames"]
	n = len(frames)

	sod = float(meta["c_arm_geometry"]["sod"])
	sdd = float(meta["c_arm_geometry"]["sdd"])
	delx = float(meta["c_arm_geometry"]["delx"])
	dely = float(meta["c_arm_geometry"]["dely"])
	width = int(meta["c_arm_geometry"]["width"])
	height = int(meta["c_arm_geometry"]["height"])

	fx = sdd / delx
	fy = sdd / dely
	cx = width / 2.0
	cy = height / 2.0

	convention = meta["rotate_parameters"]["convention"]
	alpha = np.array([float(f["alpha_degree"]) for f in frames], dtype=np.float64) * math.pi / 180.0
	beta = np.array([float(f["beta_degree"]) for f in frames], dtype=np.float64) * math.pi / 180.0

	# Match project parser: negate alpha/beta for RAS and DSA definition.
	angles = np.stack((-alpha, -beta, np.zeros(n, dtype=np.float64)), axis=-1)
	r_motion = euler_angles_to_matrix(angles, convention)

	# COLMAP-like camera orientation correction from project parser.
	r_colmap_orient = euler_angles_to_matrix(np.array([math.pi / 2.0, math.pi, 0.0]), "XZY")[0]

	m_colmap_orient = np.eye(4, dtype=np.float64)
	m_colmap_orient[:3, :3] = r_colmap_orient

	m_translation = np.eye(4, dtype=np.float64)
	m_translation[:3, 3] = np.array([0.0, sod, 0.0], dtype=np.float64)

	m_rotation = np.repeat(np.eye(4, dtype=np.float64)[None, ...], n, axis=0)
	m_rotation[:, :3, :3] = r_motion

	m_c2w = m_rotation @ m_translation @ m_colmap_orient
	m_w2c = np.linalg.inv(m_c2w)

	r_w2c = m_w2c[:, :3, :3]
	t_w2c = m_w2c[:, :3, 3]

	intrinsics = np.array([fx, fy, cx, cy], dtype=np.float64)
	image_size = np.array([width, height], dtype=np.int32)
	return r_w2c, t_w2c, intrinsics, image_size


def find_meta_json(dataset_dir: Path) -> Path:
	candidates = [
		dataset_dir / "rotate_dsa.json",
		dataset_dir / "rotate_dsa" / "rotate_dsa.json",
	]
	for p in candidates:
		if p.exists():
			return p
	raise FileNotFoundError(
		f"Cannot find rotate_dsa.json under {dataset_dir}. Tried: {candidates}"
	)


def find_image_dir(dataset_dir: Path) -> Path:
	candidates = [
		dataset_dir / "rotate_dsa",
		dataset_dir,
	]
	for p in candidates:
		if p.exists() and any(p.glob("*.png")):
			return p
	raise FileNotFoundError(
		f"Cannot find image folder with png files under {dataset_dir}"
	)


def export_images(src_dir: Path, dst_dir: Path, copy_mode: str) -> list[str]:
	dst_dir.mkdir(parents=True, exist_ok=True)
	names: list[str] = []

	for src in sorted(src_dir.glob("*.png")):
		dst = dst_dir / src.name
		if dst.exists() or dst.is_symlink():
			dst.unlink()

		if copy_mode == "copy":
			shutil.copy2(src, dst)
		elif copy_mode == "hardlink":
			try:
				dst.hardlink_to(src)
			except OSError:
				shutil.copy2(src, dst)
		else:
			dst.symlink_to(src.resolve())
		names.append(src.name)

	if not names:
		raise RuntimeError(f"No png images found in {src_dir}")
	return names


def _apply_nerfstudio_world_transform(c2w: np.ndarray, keep_original_world_coordinate: bool) -> np.ndarray:
	"""Match nerfstudio's COLMAP-to-JSON axis conversion."""
	c2w = c2w.copy()
	c2w[0:3, 1:3] *= -1
	if not keep_original_world_coordinate:
		c2w = c2w[np.array([0, 2, 1, 3]), :]
		c2w[2, :] *= -1
	return c2w


def _write_bytes(fid, format_str: str, *values) -> None:
	fid.write(struct.pack("<" + format_str, *values))


def _write_string(fid, value: str) -> None:
	fid.write(value.encode("utf-8") + b"\x00")


def write_colmap_binary_model(
	sparse0_dir: Path,
	image_names: list[str],
	r_w2c: np.ndarray,
	t_w2c: np.ndarray,
	intrinsics: np.ndarray,
	image_size: np.ndarray,
) -> None:
	sparse0_dir.mkdir(parents=True, exist_ok=True)

	width, height = int(image_size[0]), int(image_size[1])
	fx, fy, cx, cy = [float(v) for v in intrinsics]

	cameras_bin = sparse0_dir / "cameras.bin"
	images_bin = sparse0_dir / "images.bin"
	points3d_bin = sparse0_dir / "points3D.bin"

	if r_w2c.shape[0] != len(image_names):
		raise ValueError(
			f"Frame count mismatch: {r_w2c.shape[0]} poses vs {len(image_names)} images"
		)

	# COLMAP binary model layout follows read_write_model.py.
	with cameras_bin.open("wb") as f:
		_write_bytes(f, "Q", 1)
		_write_bytes(f, "i", 1)
		_write_bytes(f, "Q", width)
		_write_bytes(f, "Q", height)
		_write_bytes(f, "d", fx)
		_write_bytes(f, "d", fy)
		_write_bytes(f, "d", cx)
		_write_bytes(f, "d", cy)

	with images_bin.open("wb") as f:
		_write_bytes(f, "Q", len(image_names))
		for image_id, image_name in enumerate(image_names, start=1):
			q = rotmat_to_qvec(r_w2c[image_id - 1])
			t = t_w2c[image_id - 1]
			_write_bytes(f, "i", image_id)
			_write_bytes(f, "dddd", float(q[0]), float(q[1]), float(q[2]), float(q[3]))
			_write_bytes(f, "ddd", float(t[0]), float(t[1]), float(t[2]))
			_write_bytes(f, "i", 1)
			_write_string(f, image_name)
			_write_bytes(f, "Q", 0)

	with points3d_bin.open("wb") as f:
		_write_bytes(f, "Q", 0)


def write_transforms_json(
	output_dir: Path,
	image_names: list[str],
	r_w2c: np.ndarray,
	t_w2c: np.ndarray,
	intrinsics: np.ndarray,
	image_size: np.ndarray,
	*,
	keep_original_world_coordinate: bool,
	use_single_camera_mode: bool,
) -> None:
	width, height = int(image_size[0]), int(image_size[1])
	fx, fy, cx, cy = [float(v) for v in intrinsics]

	frames = []
	for image_id, image_name in enumerate(image_names, start=1):
		rotation = r_w2c[image_id - 1]
		translation = t_w2c[image_id - 1].reshape(3, 1)
		w2c = np.concatenate([rotation, translation], axis=1)
		w2c = np.concatenate([w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
		c2w = np.linalg.inv(w2c)
		c2w = _apply_nerfstudio_world_transform(c2w, keep_original_world_coordinate)

		frame = {
			"file_path": Path(f"./images/{image_name}").as_posix(),
			"transform_matrix": c2w.tolist(),
			"colmap_im_id": image_id,
		}
		if not use_single_camera_mode:
			frame.update(
				{
					"camera_model": "PINHOLE",
					"fl_x": fx,
					"fl_y": fy,
					"cx": cx,
					"cy": cy,
					"w": width,
					"h": height,
				}
			)
		frames.append(frame)

	out: dict[str, object] = {}
	if use_single_camera_mode:
		out.update(
			{
				"camera_model": "PINHOLE",
				"fl_x": fx,
				"fl_y": fy,
				"cx": cx,
				"cy": cy,
				"w": width,
				"h": height,
			}
		)

	out["frames"] = frames

	if not keep_original_world_coordinate:
		applied_transform = np.eye(4)[:3, :]
		applied_transform = applied_transform[np.array([0, 2, 1]), :]
		applied_transform[2, :] *= -1
		out["applied_transform"] = applied_transform.tolist()

	with (output_dir / "transforms.json").open("w", encoding="utf-8") as f:
		json.dump(out, f, indent=4)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert rotated DSA dataset to COLMAP binary model format"
	)
	parser.add_argument(
		"dataset_dir",
		type=Path,
		help="Dataset root dir, e.g. .../rbf_reader_multipli_contrast_RCA",
	)
	parser.add_argument(
		"output_dir",
		type=Path,
		help="Output COLMAP folder (will create images/ and sparse/0/)",
	)
	parser.add_argument(
		"--format",
		choices=["bin", "txt"],
		default="bin",
		help="Output model format. Default is bin for COLMAP native binary files.",
	)
	parser.add_argument(
		"--copy-mode",
		choices=["symlink", "hardlink", "copy"],
		default="symlink",
		help="How to place images into output/images",
	)
	parser.add_argument(
		"--keep-original-world-coordinate",
		action="store_true",
		help="Keep COLMAP world coordinates unchanged in transforms.json",
	)
	parser.add_argument(
		"--use-single-camera-mode",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Write camera intrinsics once at the top level of transforms.json",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	dataset_dir: Path = args.dataset_dir.resolve()
	output_dir: Path = args.output_dir.resolve()

	meta_path = find_meta_json(dataset_dir)
	image_src_dir = find_image_dir(dataset_dir)

	with meta_path.open("r", encoding="utf-8") as f:
		meta = json.load(f)

	r_w2c, t_w2c, intrinsics, image_size = build_w2c_from_json(meta)

	images_out = output_dir / "images"
	sparse0_out = output_dir / "sparse" / "0"

	image_names = export_images(image_src_dir, images_out, copy_mode=args.copy_mode)
	if args.format == "bin":
		write_colmap_binary_model(
			sparse0_out,
			image_names=image_names,
			r_w2c=r_w2c,
			t_w2c=t_w2c,
			intrinsics=intrinsics,
			image_size=image_size,
		)
	else:
		raise NotImplementedError("Text export was removed; use --format bin")

	write_transforms_json(
		output_dir,
		image_names=image_names,
		r_w2c=r_w2c,
		t_w2c=t_w2c,
		intrinsics=intrinsics,
		image_size=image_size,
		keep_original_world_coordinate=args.keep_original_world_coordinate,
		use_single_camera_mode=args.use_single_camera_mode,
	)

	print(f"Input dataset : {dataset_dir}")
	print(f"Metadata      : {meta_path}")
	print(f"Image source  : {image_src_dir}")
	print(f"Output COLMAP : {output_dir}")
	print(f"Frames        : {len(image_names)}")
	print("Done.")


if __name__ == "__main__":
	main()
