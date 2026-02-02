# utils/paths.py
from pathlib import Path

# 项目根目录（本文件位于 ai-shopping-assistant/utils/paths.py）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_DIR  = PROJECT_ROOT / "train"
TEST_DIR   = PROJECT_ROOT / "test"

def ensure_dirs():
    """按需创建常用目录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
