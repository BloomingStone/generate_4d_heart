# /// script
# requires-python = ">=3.10"
# dependencies = ["pydicom", "numpy", "tifffile", "cyclopts"]
# ///

import sys
import os
from pathlib import Path
import json
import datetime
from typing import Literal
import numpy as np
import tifffile
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, XRayAngiographicImageStorage, generate_uid
from pydicom.datadict import add_private_dict_entry
from pydicom.valuerep import VR
from cyclopts import run

# --- Private tag for the per-frame cardiac phase array ---
# Creator must be a unique institution string; element offset within the block
# resolves to the full tag (0x0071, offset) and is registered for correct VR.
GEN4D_PHASE_PRIVATE_CREATOR = os.environ.get("GEN4D_PHASE_PRIVATE_CREATOR", "GEN4D_PHASE_1.0")
PHASE_PRIVATE_GROUP = 0x0071
PHASE_PRIVATE_ELEMENT = 0x10  # block offset -> (0x0071, 0x10) in this block
PHASE_PRIVATE_TAG = 0x00711010
# Register so reading the file back shows VR=FL instead of UN (run once per process)
try:
    add_private_dict_entry(
        GEN4D_PHASE_PRIVATE_CREATOR, PHASE_PRIVATE_TAG, "FL", "Cardiac Phase Array"
    )
except ValueError:
    pass  # already registered in this process

GEN4D_REF_MEAN = float(os.environ.get("GEN4D_REF_MEAN", "7218.3694"))
GEN4D_REF_STD = float(os.environ.get("GEN4D_REF_STD", "15221.4602"))
GEN4D_REF_P05 = float(os.environ.get("GEN4D_REF_P05", "226.0"))
GEN4D_REF_P25 = float(os.environ.get("GEN4D_REF_P25", "393.0"))
GEN4D_REF_P50 = float(os.environ.get("GEN4D_REF_P50", "810.0"))
GEN4D_REF_P75 = float(os.environ.get("GEN4D_REF_P75", "2421.0"))
GEN4D_REF_P95 = float(os.environ.get("GEN4D_REF_P95", "51850.0"))


def float_to_uint16_gamma(arr: np.ndarray, gamma: float=0.1) -> np.ndarray:
    """
    Convert a float array to uint16 with gamma correction.
    """
    if arr.min() < 0:
        arr = arr - arr.min()

    arr = arr**gamma
    return np.clip(arr / arr.max() * 65535, 0, 65535).astype(np.uint16)

def float_to_uint16_by_features(arr: np.ndarray) -> np.ndarray:
    if arr.min() < 0:
        arr = arr - arr.min()
    f = arr.ravel().astype(np.float64)
    t_p05 = float(np.percentile(f, 5))
    t_p25 = float(np.percentile(f, 25))
    t_p50 = float(np.percentile(f, 50))
    t_p75 = float(np.percentile(f, 75))
    t_p95 = float(np.percentile(f, 95))
    t_mean = float(np.mean(f))
    t_std = float(np.std(f))
    
    x = np.array([t_p05, t_p25, t_p50, t_p75, t_p95, t_mean, t_std])
    y = np.array([GEN4D_REF_P05, GEN4D_REF_P25, GEN4D_REF_P50, GEN4D_REF_P75, GEN4D_REF_P95, GEN4D_REF_MEAN, GEN4D_REF_STD])
    coefficients = np.polyfit(x, y, 1)
    a, b = coefficients

    print(f"  Fit: uint16 = {a:.2f} * float + {b:.2f}")
    result = np.clip(arr.astype(np.float64) * a + b, 0, 65535)
    return result.astype(np.uint16)

def float_to_uint16_log(arr: np.ndarray) -> np.ndarray:
    """
    Convert a float array to uint16 with log scaling.
    """
    if arr.min() < 0:
        arr = arr - arr.min()
    
    arr_max = arr.max()
    arr = np.log1p(arr) 
    return np.clip(arr / arr.max() * 65535, 0, 65535).astype(np.uint16)


def load_rotate_dsa_json(json_path: Path) -> dict:
    """Load and validate the rotate_dsa.json metadata file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Reading JSON: {json_path}")
    print(f"  coronary_type: {data.get('coronary_type', 'N/A')}")
    print(f"  frames: {len(data.get('frames', []))}")
    c_arm = data.get("c_arm_geometry", {})
    print(f"  c_arm: SDD={c_arm.get('sdd')}, SOD={c_arm.get('sod')}")
    print(f"  c_arm: delx={c_arm.get('delx')}, dely={c_arm.get('dely')}")
    print(f"  c_arm: height={c_arm.get('height')}, width={c_arm.get('width')}")
    rot = data.get("rotate_parameters", {})
    print(f"  rotate: fps={rot.get('fps')}, total_frame={rot.get('total_frame')}")
    print(f"  rotate: alpha_start={rot.get('alpha_start')}, beta_start={rot.get('beta_start')}")
    return data



def make_xa(
    arr: np.ndarray, 
    json_data: dict,
    float_to_uint16_method: Literal["gamma", "features"],
    window_center: int|None,
    window_width: int|None,
    gamm_value: float=0.1
):
    ndim = arr.ndim
    if ndim == 2:
        frames, h, w = 1, arr.shape[0], arr.shape[1]
    elif ndim == 3:
        frames, h, w = arr.shape
    else:
        raise ValueError(f"bad shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        print(f"  Converting float {arr.dtype} -> uint16")
        if float_to_uint16_method == "gamma":
            arr = float_to_uint16_gamma(arr, gamma=gamm_value)
        elif float_to_uint16_method == "features":
            arr = float_to_uint16_by_features(arr)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)

    now = datetime.datetime.now()
    sd = now.strftime("%Y%m%d")
    st = now.strftime("%H%M%S.%f")[:13]

    # --- Extract JSON metadata ---
    if json_data is None:
        raise ValueError("Missing JSON metadata; cannot create DICOM header")
    
    c_arm: dict|None = json_data.get("c_arm_geometry", None)
    rot_params: dict|None = json_data.get("rotate_parameters", None)
    frame_list: list = json_data.get("frames", [])
    coronary_type: str|None = json_data.get("coronary_type", "Unknown-Branch")

    if c_arm is None:
        raise ValueError("Missing c_arm_geometry in JSON; cannot create DICOM header")
    if rot_params is None:
        raise ValueError("Missing rotate_parameters in JSON; cannot create DICOM header")
    
    delx = c_arm.get("delx")
    dely = c_arm.get("dely")
    if delx is None or dely is None:
        raise ValueError("Missing delx or dely in c_arm_geometry; cannot create DICOM header")
    
    imager_spacing = [float(delx), float(dely)]
    print(f"  ImagerPixelSpacing: {imager_spacing}")

    # --- Build DICOM dataset ---
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.FileMetaInformationVersion = b"\x00\x01"
    ds.file_meta.MediaStorageSOPClassUID = XRayAngiographicImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.file_meta.ImplementationVersionName = "tif2dcm"

    # --- SOP / Patient ---
    ds.SOPClassUID = XRayAngiographicImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "XA"
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.PatientName = "generated^ROTA"
    ds.PatientID = generate_uid(prefix=None)[:16]
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = sd
    ds.StudyTime = st
    ds.SeriesDate = sd
    ds.SeriesTime = st
    ds.ContentDate = sd
    ds.ContentTime = st
    ds.AccessionNumber = ""
    ds.StudyID = "1"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # --- Image type derived from orientation_type ---
    orientation_type = c_arm.get("orientation_type", "PA")
    ds.ImageType = ["ORIGINAL", "PRIMARY", "SINGLE PLANE"]
    ds.ScanOptions = "ROTA"

    # --- C-arm geometry from JSON ---
    sdd = c_arm.get("sdd")
    sod = c_arm.get("sod")
    if sdd is None or sod is None:
        raise ValueError("Missing sdd or sod in c_arm_geometry; cannot create DICOM header")

    ds.DistanceSourceToDetector = str(sdd)
    ds.DistanceSourceToPatient = str(sod)
    mag = sdd / sod
    ds.EstimatedRadiographicMagnificationFactor = f"{mag:.6f}"
    print(f"  Magnification: {mag:.6f}")

    # --- Frame / dimension ---
    ds.NumberOfFrames = str(frames)
    ds.Rows = h
    ds.Columns = w
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    # PixelSpacing: physical pixel spacing (imager spacing / magnification)
    # ImagerPixelSpacing: pixel spacing at the detector plane
    ds.ImagerPixelSpacing = imager_spacing
    ds.PixelSpacing = [round(s / mag, 7) for s in imager_spacing]
    print(f"  PixelSpacing (physical, from json): {ds.PixelSpacing}")
    ds.PatientPosition = "HFS"

    # --- Window ---
    if window_center is None or window_width is None:
        print("  WindowCenter/WindowWidth not provided; computing from image percentiles p05/p95")
        vmin, vmax = float(np.percentile(arr, 0.5)), float(np.percentile(arr, 99.5))
        ds.WindowCenter = (vmin + vmax) / 2
        ds.WindowWidth = vmax - vmin
    else:
        print(f"  Using provided WindowCenter={window_center}, WindowWidth={window_width}")
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width

    # --- Rotation parameters ---
    fps = rot_params.get("fps", 80)
    ds.CineRate = str(int(fps))
    ds.RecommendedDisplayFrameRate = str(int(fps))

    # FrameTimeVector: time between frames in ms
    # For 80 fps, each frame is 12.5 ms
    frame_interval_ms = 1000.0 / fps
    # Use per-frame times from JSON if available
    if frame_list:
        # time values in the JSON are cumulative seconds; compute per-frame delta in ms
        ds.FrameTimeVector = [
            f"{frame_interval_ms:.1f}"
        ] * frames  # uniform frame time
        # Positioner angles per frame
        ds.PositionerMotion = "DYNAMIC"
        ds.TableMotion = "STATIC"

        primary_angles = []
        secondary_angles = []
        for frm in frame_list:
            primary_angles.append(round(float(frm["alpha_degree"]), 2))
            secondary_angles.append(round(float(frm["beta_degree"]), 2))

        ds.PositionerPrimaryAngleIncrement = primary_angles
        ds.PositionerSecondaryAngleIncrement = secondary_angles

        # Starting angles
        ds.PositionerPrimaryAngle = f"{primary_angles[0]:.2f}"
        ds.PositionerSecondaryAngle = f"{secondary_angles[0]:.2f}"
        print(f"  Angles: primary={primary_angles[0]:.2f}..{primary_angles[-1]:.2f}")
        print(f"          secondary={secondary_angles[0]:.2f}..{secondary_angles[-1]:.2f}")

        # --- Cardiac phase array as a private tag (float32, one per frame) ---
        phases = np.asarray([float(frm["phase"]) for frm in frame_list], dtype=np.float32)
        if len(phases) != frames:
            print(
                f"  Warning: phase array length ({len(phases)}) != frames ({frames}); trimming",
                file=sys.stderr,
            )
            if len(phases) > frames:
                phases = phases[:frames]
            else:
                phases = np.pad(phases, (0, frames - len(phases)))
        block = ds.private_block(PHASE_PRIVATE_GROUP, GEN4D_PHASE_PRIVATE_CREATOR, create=True)
        block.add_new(PHASE_PRIVATE_ELEMENT, VR.FL, phases.tolist())
        print(
            f"  Private tag {PHASE_PRIVATE_GROUP:04X}{PHASE_PRIVATE_ELEMENT:04X} "
            f"(Cardiac Phase): {len(phases)} float32, "
            f"range [{phases.min():.4f}, {phases.max():.4f}]"
        )
    else:
        ds.FrameTimeVector = "0"

    # --- Series / protocol description ---
    ds.SeriesDescription = f"Rotational DSA - {coronary_type}"
    ds.ProtocolName = f"Rotational DSA - {coronary_type}"

    # --- Reference DICOM values (manufacturer info) ---
    ds.Manufacturer = "Neusoft Medical Systems"
    ds.ManufacturerModelName = "NeuAngio 43C"
    ds.StationName = "CV"
    ds.InstitutionName = ""
    ds.DeviceSerialNumber = generate_uid(prefix=None)[:16]
    ds.SoftwareVersions = "V1.0"
    ds.DetectorType = "DIRECT"
    ds.ContrastBolusAgent = ""

    # Irradiation event UID
    ds.IrradiationEventUID = generate_uid()

    # --- Pixel data ---
    ds.PixelData = arr.tobytes()

    return ds


def main(
    tif_path: Path,
    json_path: Path|None = None,
    window_center: int|None = None,
    window_width: int|None = None,
    float_to_uint16_method: Literal["gamma", "features"]="gamma",
):
    """
    Convert a TIF file to a DICOM XA file, using optional JSON metadata for C-arm geometry and rotation parameters.
    Usage: uv run scripts/convert_tif_to_dicom.py tif_path [--json_path] [--window_center] [--window_width] [--float_to_uint16_method]
    Args:
        tif_path (Path): Path to the input TIF file.
        json_path (Path|None): Path to the optional JSON metadata file, if not provided, will look for rotate_dsa.json in the same directory as the TIF.
        window_center (float|None): Optional window center for DICOM header.
        window_width (float|None): Optional window width for DICOM header.
        float_to_uint16_method (Literal["gamma", "features"]): Method to convert float arrays to uint16. "gamma" applies gamma correction, "features" uses statistical features for scaling.
    """
    if not tif_path.exists() or not tif_path.is_file():
        raise FileNotFoundError(f"Error: not found: {tif_path}")


    if json_path is None:
        # Auto-detect: look for rotate_dsa.json in the same directory as TIF
        json_path = tif_path.parent / "rotate_dsa.json"
    
    if not json_path.exists() or not json_path.is_file():
        raise FileNotFoundError(f"Error: JSON not found: {json_path}")

    # Load JSON metadata
    json_data = load_rotate_dsa_json(json_path)

    # Read TIF
    print(f"Reading TIF: {tif_path}")
    arr = tifffile.imread(tif_path)
    print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  Range: [{arr.min():.6g}, {arr.max():.6g}], mean: {arr.mean():.6g}")

    # Validate frame count if JSON is available
    n_json_frames = len(json_data["frames"])
    n_tif_frames = arr.shape[0] if arr.ndim == 3 else 1
    if n_json_frames != n_tif_frames:
        raise ValueError(
            f"  Warning: JSON frames ({n_json_frames}) != TIF frames ({n_tif_frames})"
        )

    # Create DICOM
    ds = make_xa(arr, json_data=json_data, float_to_uint16_method=float_to_uint16_method, 
                window_center=window_center, window_width=window_width)

    # Write output
    out = tif_path.with_suffix(".dcm")
    print(f"Writing DICOM: {out}")
    ds.save_as(out, write_like_original=False)
    print("Done.")


if __name__ == "__main__":
    run(main)
