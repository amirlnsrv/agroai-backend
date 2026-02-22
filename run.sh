#!/usr/bin/env bash
#!/source ../venv/bin/activate

set -e
cd "$(dirname "$0")"
export AGROAI_DATA_DIR="$(pwd)/data"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
