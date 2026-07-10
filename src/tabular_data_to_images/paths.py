from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"
MODELS_DIR = REPO_ROOT / "trained_models"

# REPO_ROOT doubles as the root under which the original, raw BCCC dataset is expected to
# live (not committed — place the extracted BCCC-cPacket-Cloud-DDoS-2024 export at
# REPO_ROOT/datasets/bccc-cpacket-cloud-ddos-2024/ before running dataset/processing_scripts/).
BCCC_RAW_DATASET_DIR = REPO_ROOT / "datasets" / "bccc-cpacket-cloud-ddos-2024"
MERGED_CSV_PATH = BCCC_RAW_DATASET_DIR / "merged_csv.csv"
