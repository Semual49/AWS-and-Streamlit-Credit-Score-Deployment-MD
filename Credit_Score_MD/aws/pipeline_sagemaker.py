"""
pipeline_sagemaker.py - Jalankan training pipeline di dalam SageMaker Notebook.

Script ini membaca data dari folder aws/data/data_C.csv,
memanggil pipeline_local.py yang ada di src/models/,
dan menyimpan model ke aws/model_artifact/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Root project = dua level di atas file ini (aws/ -> root)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "aws" / "data" / "data_C.csv"
MODEL_OUT = ROOT / "aws" / "model_artifact" / "model_best.pkl"
METRICS_OUT = ROOT / "aws" / "model_artifact" / "metrics.json"
LOG_PATH = ROOT / "logs" / "notebook_pipeline.log"


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: Data tidak ditemukan di {DATA_PATH}")
        print("Letakkan data_C.csv di folder aws/data/ lalu jalankan ulang.")
        sys.exit(1)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    pipeline_script = ROOT / "src" / "models" / "pipeline_local.py"
    if not pipeline_script.exists():
        print(f"ERROR: pipeline_local.py tidak ditemukan di {pipeline_script}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--data", str(DATA_PATH),
        "--model-out", str(MODEL_OUT),
        "--metrics-out", str(METRICS_OUT),
        "--log-path", str(LOG_PATH),
    ]

    print("=" * 60)
    print("Memulai training pipeline...")
    print(f"Data   : {DATA_PATH}")
    print(f"Model  : {MODEL_OUT}")
    print(f"Metrics: {METRICS_OUT}")
    print(f"Log    : {LOG_PATH}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("ERROR: Pipeline gagal. Cek log di atas.")
        sys.exit(result.returncode)

    print("\nPipeline selesai.")
    print(f"Model tersimpan di: {MODEL_OUT}")


if __name__ == "__main__":
    main()
