"""
package_model.py - Package model ke model.tar.gz untuk upload ke S3.

Isi model.tar.gz yang dihasilkan:
    model_best.pkl           <- model artifact dari training
    inference_sagemaker.py   <- entry point SageMaker
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PKL = ROOT / "aws" / "model_artifact" / "model_best.pkl"
INFERENCE_FILE = ROOT / "aws" / "inference_sagemaker.py"
OUTPUT_TAR = ROOT / "aws" / "model_artifact" / "model.tar.gz"


def main() -> None:
    if not MODEL_PKL.exists():
        print(f"ERROR: {MODEL_PKL} tidak ditemukan.")
        print("Jalankan pipeline_sagemaker.py dulu untuk training model.")
        raise SystemExit(1)

    if not INFERENCE_FILE.exists():
        print(f"ERROR: {INFERENCE_FILE} tidak ditemukan.")
        raise SystemExit(1)

    print("Membuat model.tar.gz...")
    print(f"  Sumber model  : {MODEL_PKL}")
    print(f"  Sumber handler: {INFERENCE_FILE}")
    print(f"  Output        : {OUTPUT_TAR}")

    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        tar.add(MODEL_PKL, arcname="model_best.pkl")
        tar.add(INFERENCE_FILE, arcname="inference_sagemaker.py")

    print("\nBerhasil. Isi model.tar.gz:")
    with tarfile.open(OUTPUT_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  {member.name}  ({member.size:,} bytes)")

    print(f"\nUkuran total: {OUTPUT_TAR.stat().st_size / 1024:.1f} KB")
    print("\nLangkah berikutnya:")
    print("  python aws/upload_to_s3.py")


if __name__ == "__main__":
    main()
