"""固定视角 DSA 投影 CLI。

C 臂固定在某个 C-arm 角度下采集，只有心脏相位/造影剂随时间演化。

用法示例
--------
# AP 正位、固定视角、90 帧
pixi run python scripts/fix_angle_dsa.py out/ap \\
    --volume-nii  test/test_data/volume_with_dvf/image.nii.gz \\
    --cavity-nii  test/test_data/volume_with_dvf/cavity.nii.gz \\
    --coronary-nii test/test_data/volume_with_dvf/coronary.nii.gz \\
    --angle AP

# RAO 30° + CRAN 20°，xcist 后端并输出剂量
pixi run python scripts/fix_angle_dsa.py out/rao30cran20 \\
    ... --angle RAO_30_CRAN_20 --backend xcist --dose

# 直接给角度值（alpha, beta）
pixi run python scripts/fix_angle_dsa.py out/custom ... --angle "45,10"
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from cyclopts import App

from generate_4d_heart.rotate_dsa.contrast_simulator import (
    FlowContrast, Hu2MuContrast, IdentityContrast, StaticIodineContrast,
)
from generate_4d_heart.rotate_dsa.data_reader import (
    RBFReader, StaticVolumeReader, VolumeDVFReader,
)
from generate_4d_heart.rotate_dsa.fix_angle_dsa import FixAngleDSA, parse_angle
from generate_4d_heart.rotate_dsa.rotate_drr import (
    CArmGeometry, RotatedParameters, TorchDRR, XcistDRR,
)

app = App()


def _contrast(kind: str):
    match kind:
        case "static":
            return StaticIodineContrast()
        case "flow":
            return FlowContrast()
        case "identity":
            return IdentityContrast()
        case "hu2mu":
            return Hu2MuContrast()
        case _:
            raise ValueError(f"未知造影剂模拟器: {kind}")


def _reader(
    kind: str,
    volume_nii: Path,
    cavity_nii: Path,
    coronary_nii: Path,
    contrast: str,
    dvf_dir: Path | None,
    roi_json: Path | None,
):
    sim = _contrast(contrast)
    match kind:
        case "rbf":
            return RBFReader(volume_nii=volume_nii, cavity_nii=cavity_nii,
                             coronary_nii=coronary_nii, contrast_simulator=sim)
        case "static":
            return StaticVolumeReader(volume_path=volume_nii, cavity_path=cavity_nii,
                                      coronary_path=coronary_nii, contrast_simulator=sim)
        case "dvf":
            if dvf_dir is None or roi_json is None:
                raise ValueError("--reader dvf 需要同时给 --dvf-dir 和 --roi-json")
            return VolumeDVFReader(volume_nii=volume_nii, cavity_nii=cavity_nii,
                                   coronary_nii=coronary_nii, dvf_dir=dvf_dir,
                                   roi_json=roi_json, contrast_simulator=sim)
        case _:
            raise ValueError(f"未知 reader 类型: {kind}")


@app.default
def fix_angle_dsa(
    output_dir: Path,
    volume_nii: Path,
    cavity_nii: Path,
    coronary_nii: Path,
    *,
    angle: str = "AP",
    coronary_type: Literal["LCA", "RCA"] = "LCA",
    reader: Literal["rbf", "static", "dvf"] = "rbf",
    contrast: Literal["static", "flow", "identity", "hu2mu"] = "static",
    dvf_dir: Path | None = None,
    roi_json: Path | None = None,
    backend: Literal["torch", "xcist"] = "torch",
    orientation: Literal["AP", "PA"] = "AP",
    sdd: float = 1200.0,
    sod: float = 760.0,
    height: int = 512,
    width: int | None = None,
    delx: float = 0.368,
    dely: float | None = None,
    total_frame: int = 90,
    fps: float = 30.0,
    patch_size: int = 256,
    gif_fps: int = 30,
    dose: bool = False,
    dose_n_voxel: int = 48,
    dose_fov_mm: float = 320.0,
    dose_slice_mm: float = 5.0,
    dose_slices: int = 48,
):
    """固定视角 DSA 投影。

    Parameters
    ----------
    output_dir : Path
        输出目录（主产物命名为 ``fix_angle_dsa*``）。
    volume_nii, cavity_nii, coronary_nii : Path
        CT(HU) / 心腔标签 / 冠脉标签。
    angle : str
        固定角度：``AP`` / ``LAT`` / ``RLAT`` / ``RAO_30`` / ``LAO_60`` /
        ``CRAN_20`` / ``CAUD_10`` / ``RAO_30_CRAN_20`` / ``"30,10"``。
    coronary_type : str
        ``LCA`` 或 ``RCA``。
    reader : str
        数据读取器：``rbf``（推荐）/ ``static`` / ``dvf``。
    contrast : str
        造影剂模拟：``static`` / ``flow`` / ``identity`` / ``hu2mu``。
    backend : str
        投影后端：``torch``（DiffDRR, 需 CUDA）或 ``xcist``（gecatsim, CPU）。
    dose : bool
        仅 ``--backend xcist`` 有效；启用剂量输出（``dose.npy`` / ``dose.mGy``）。
        未提供参考 CT 时用 DRR 的 μ 体积反推 HU（``hu_source="mu_approx"``，偏低估）。
    """
    alpha, beta = parse_angle(angle)
    print(f"[fix-angle] alpha={alpha:.1f}°, beta={beta:.1f}°  (输入 {angle!r})")

    rd = _reader(reader, volume_nii, cavity_nii, coronary_nii, contrast, dvf_dir, roi_json)

    w = height if width is None else width
    dly = delx if dely is None else dely
    c_arm = CArmGeometry(sdd=sdd, sod=sod, height=height, width=w,
                         delx=delx, dely=dly, orientation_type=orientation)
    rot = RotatedParameters(total_frame=total_frame, fps=fps,
                            alpha_start=alpha, beta_start=beta,
                            angular_velocity=0.0)

    if backend == "torch":
        drr = TorchDRR(c_arm_cfg=c_arm, rotate_cfg=rot, patch_size=patch_size)
    else:
        from generate_4d_heart.xcist.dose import DoseConfig

        drr = XcistDRR(
            c_arm_cfg=c_arm, rotate_cfg=rot, mode="mono",
            dose_cfg=(DoseConfig(n_voxel=dose_n_voxel, fov_mm=dose_fov_mm,
                                 slice_thickness_mm=dose_slice_mm) if dose else None),
            dose_slices=dose_slices,
        )

    dsa = FixAngleDSA(reader=rd, drr=drr, alpha=alpha, beta=beta)
    dsa.run_and_save(output_dir, coronary_type=coronary_type, gif_fps=gif_fps)
    print(f"[fix-angle] 完成，产物: {output_dir}")


if __name__ == "__main__":
    app()
