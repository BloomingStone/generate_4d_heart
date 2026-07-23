# /// script
# requires-python = ">=3.10"
# dependencies = ["pydicom", "numpy", "tifffile"]
# ///

import sys
import os
import datetime
import numpy as np
import tifffile
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

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

def make_xa(arr, spacing=None):
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

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.FileMetaInformationVersion = b"\x00\x01"
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.file_meta.ImplementationVersionName = "tif2dcm"
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "XA"
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
    ds.NumberOfFrames = frames
    ds.Rows = h
    ds.Columns = w
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = spacing
    ds.PatientPosition = "HFS"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "DXA"]
    ds.WindowCenter = 6000
    ds.WindowWidth = 17000
    ds.PixelData = arr.tobytes()
    return ds

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run convert_tif_to_dicom.py TIF_PATH", file=sys.stderr)
        sys.exit(1)
    tif_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(tif_path):
        print(f"Error: not found: {tif_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading TIF: {tif_path}")
    arr = tifffile.imread(tif_path)
    print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  Range: [{arr.min():.6g}, {arr.max():.6g}], mean: {arr.mean():.6g}")
    ds = make_xa(arr)
    out_dir = os.path.dirname(tif_path)
    out_name = os.path.splitext(os.path.basename(tif_path))[0] + ".dcm"
    out = os.path.join(out_dir, out_name)
    print(f"Writing DICOM: {out}")
    ds.save_as(out, write_like_original=False)
    print("Done.")

if __name__ == "__main__":
    main()
