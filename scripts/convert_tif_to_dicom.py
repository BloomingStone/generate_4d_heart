# /// script
# requires-python = ">=3.10"
# dependencies = ["pydicom", "numpy", "tifffile"]
# ///

import sys
import os
import json
import datetime
import numpy as np
import tifffile
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, XRayAngiographicImageStorage, generate_uid

PIXEL_SPACING = [0.616, 0.616]

REF_MEAN = 7218.3694
REF_STD = 15221.4602
REF_P05 = 226.0
REF_P25 = 393.0
REF_P50 = 810.0
REF_P75 = 2421.0
REF_P95 = 51850.0


def float_to_uint16(arr):
    if arr.min() < 0:
        arr = arr - arr.min()
    f = arr.ravel().astype(np.float64)
    t_p05, t_p95 = float(np.percentile(f, 5)), float(np.percentile(f, 95))
    a = (REF_P95 - REF_P05) / (t_p95 - t_p05)
    b = REF_P05 - a * t_p05
    print(f"  Fit: uint16 = {a:.2f} * float + {b:.2f}")
    result = np.clip(arr.astype(np.float64) * a + b, 0, 65535)
    return result.astype(np.uint16)


def load_rotate_dsa_json(json_path):
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


def make_xa(arr, json_data=None, spacing=None):
    ndim = arr.ndim
    if ndim == 2:
        frames, h, w = 1, arr.shape[0], arr.shape[1]
    elif ndim == 3:
        frames, h, w = arr.shape
    else:
        raise ValueError(f"bad shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        print(f"  Converting float {arr.dtype} -> uint16")
        arr = float_to_uint16(arr)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)

    if spacing is None:
        spacing = PIXEL_SPACING

    now = datetime.datetime.now()
    sd = now.strftime("%Y%m%d")
    st = now.strftime("%H%M%S.%f")[:13]

    # --- Extract JSON metadata ---
    c_arm = {}
    rot_params = {}
    frame_list = []
    coronary_type = "LCA"
    if json_data is not None:
        c_arm = json_data.get("c_arm_geometry", {})
        rot_params = json_data.get("rotate_parameters", {})
        frame_list = json_data.get("frames", [])
        coronary_type = json_data.get("coronary_type", coronary_type)

    # Pixel spacing from JSON, fallback to argument or default
    imager_spacing = None
    if c_arm:
        delx = c_arm.get("delx")
        dely = c_arm.get("dely")
        if delx is not None and dely is not None:
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
    mag = 1.0
    if c_arm:
        sdd = c_arm.get("sdd")
        sod = c_arm.get("sod")
        if sdd is not None:
            ds.DistanceSourceToDetector = str(sdd)
        if sod is not None:
            ds.DistanceSourceToPatient = str(sod)
        if sdd is not None and sod is not None and sod > 0:
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
    if imager_spacing is not None:
        ds.ImagerPixelSpacing = imager_spacing
        ds.PixelSpacing = [round(s / mag, 7) for s in imager_spacing]
        print(f"  PixelSpacing (physical): {ds.PixelSpacing}")
    else:
        ds.PixelSpacing = spacing
    ds.PatientPosition = "HFS"

    # --- Window ---
    ds.WindowCenter = 6000
    ds.WindowWidth = 17000

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


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run convert_tif_to_dicom.py TIF_PATH [JSON_PATH]", file=sys.stderr)
        print("")
        print("  TIF_PATH   Path to the multi-frame TIFF file containing DSA image data")
        print("  JSON_PATH  Path to rotate_dsa.json (optional, auto-detected if omitted)")
        sys.exit(1)

    tif_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(tif_path):
        print(f"Error: not found: {tif_path}", file=sys.stderr)
        sys.exit(1)

    # Determine JSON path
    json_path = None
    if len(sys.argv) >= 3:
        json_path = os.path.abspath(sys.argv[2])
        if not os.path.isfile(json_path):
            print(f"Error: JSON not found: {json_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Auto-detect: look for rotate_dsa.json in the same directory as TIF
        tif_dir = os.path.dirname(tif_path)
        candidate = os.path.join(tif_dir, "rotate_dsa.json")
        if os.path.isfile(candidate):
            json_path = candidate
            print(f"  Auto-detected JSON: {json_path}")
        else:
            print("  No rotate_dsa.json found; using minimal DICOM header")

    # Load JSON metadata
    json_data = None
    if json_path:
        json_data = load_rotate_dsa_json(json_path)

    # Read TIF
    print(f"Reading TIF: {tif_path}")
    arr = tifffile.imread(tif_path)
    print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  Range: [{arr.min():.6g}, {arr.max():.6g}], mean: {arr.mean():.6g}")

    # Validate frame count if JSON is available
    if json_data and len(json_data.get("frames", [])) > 0:
        n_json_frames = len(json_data["frames"])
        n_tif_frames = arr.shape[0] if arr.ndim == 3 else 1
        if n_json_frames != n_tif_frames:
            print(
                f"  Warning: JSON frames ({n_json_frames}) != TIF frames ({n_tif_frames})",
                file=sys.stderr,
            )

    # Create DICOM
    ds = make_xa(arr, json_data=json_data)

    # Write output
    out_dir = os.path.dirname(tif_path)
    out_name = os.path.splitext(os.path.basename(tif_path))[0] + ".dcm"
    out = os.path.join(out_dir, out_name)
    print(f"Writing DICOM: {out}")
    ds.save_as(out, write_like_original=False)
    print("Done.")


if __name__ == "__main__":
    main()
