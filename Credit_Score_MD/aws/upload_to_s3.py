"""
upload_to_s3.py - Buat S3 bucket kalau belum ada dan upload model.tar.gz.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BUCKET_NAME = "credit-score-2802545466"
REGION = "us-east-1"
S3_KEY = "model/model.tar.gz"

ROOT = Path(__file__).resolve().parent.parent
TAR_PATH = ROOT / "aws" / "model_artifact" / "model.tar.gz"


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:

    if not TAR_PATH.exists():
        print(f"ERROR: {TAR_PATH} tidak ditemukan.")
        print("Jalankan package_model.py dulu.")
        sys.exit(1)

    s3_uri = f"s3://{BUCKET_NAME}/{S3_KEY}"
    print("=" * 60)
    print(f"Bucket : {BUCKET_NAME}")
    print(f"Region : {REGION}")
    print(f"Upload : {TAR_PATH} -> {s3_uri}")
    print("=" * 60)

    # Buat bucket (abaikan error kalau sudah ada)
    rc = run(["aws", "s3", "mb", f"s3://{BUCKET_NAME}", "--region", REGION])
    if rc not in (0, 1):
        print(f"Gagal membuat bucket (return code {rc})")
        sys.exit(rc)

    # Upload
    rc = run(["aws", "s3", "cp", str(TAR_PATH), s3_uri])
    if rc != 0:
        print("Upload gagal.")
        sys.exit(rc)

    print("\nUpload berhasil.")
    print(f"S3 URI: {s3_uri}")
    print("\nLangkah berikutnya:")
    print(f"  Edit aws/deploy_endpoint.py -> set BUCKET_NAME = '{BUCKET_NAME}'")
    print("  python aws/deploy_endpoint.py")


if __name__ == "__main__":
    main()
