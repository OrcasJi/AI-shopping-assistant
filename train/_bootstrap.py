# train/_bootstrap.py
"""
确保从任何工作目录运行训练脚本时，都能 import 到项目内模块（如 utils.paths）。
用法：在每个训练脚本开头写 `import train._bootstrap` 即可。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
