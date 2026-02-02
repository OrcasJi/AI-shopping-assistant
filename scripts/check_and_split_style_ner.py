# scripts/check_and_split_style_ner.py
"""
一键检查 Style NER 训练/验证集，并在需要时重新分层切分。
- 自动合并 <repo>/data/ 下的标注文件（*.json, *.jsonl）
- 按句子是否包含 B/I 分层切分
- 输出句子级 & 近似 token 级分布，并做泄漏检查
"""

import json, random, sys
from collections import Counter
from pathlib import Path

# ==== 路径：相对脚本自身定位到项目根 ====
CURRENT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
TRAIN_FILE   = DATA_DIR / "style_train.json"
VAL_FILE     = DATA_DIR / "style_val.json"
ALL_FILE     = DATA_DIR / "style_all.json"

VAL_RATIO  = 0.2
SEED       = 42

# ---------- 基础 IO ----------
def load_json(p: Path):
    return json.load(open(p, "r", encoding="utf-8"))

def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------- 校验 ----------
def is_valid_item(x) -> bool:
    return (
            isinstance(x, dict)
            and isinstance(x.get("text"), str)
            and isinstance(x.get("labels"), list)
            and isinstance(x.get("word_ends"), list)
            and len(x["labels"]) == len(x["word_ends"])
    )

def validate_dataset(items, name="dataset"):
    if not items:
        return True
    ok = 0
    for x in items[:200]:  # 抽样
        if is_valid_item(x):
            ok += 1
    if ok == 0:
        print(f"[ERROR] {name} 结构不符合要求（需包含 text/labels/word_ends 且 labels 与 word_ends 等长）。")
        return False
    return True

# ---------- 统计 ----------
def sent_pos_ratio(items):
    tot = len(items)
    pos = sum(1 for x in items if any(t != "O" for t in x.get("labels", [])))
    return pos, tot

def token_dist(items):
    m = Counter()
    for x in items:
        for lab in x.get("labels", []):
            if lab == "O": m[0] += 1
            elif isinstance(lab, str) and lab.startswith("B"): m[1] += 1
            elif isinstance(lab, str) and lab.startswith("I"): m[2] += 1
            else: m[0] += 1
    return m

def pretty_dist(cnter):
    total = sum(cnter.values()) or 1
    names = {0:"O",1:"B-STYLE",2:"I-STYLE"}
    return ", ".join(f"{names.get(k,k)}={cnter.get(k,0)} ({cnter.get(k,0)/total:.2%})" for k in [0,1,2])

def dup_stats(train, val):
    train_txt = set(x.get("text","").strip() for x in train)
    val_txt   = set(x.get("text","").strip() for x in val)
    inter = train_txt & val_txt
    return len(train_txt), len(val_txt), len(inter), sorted(list(inter))[:5]

# ---------- 切分 ----------
def stratified_split(items, val_ratio=0.2, seed=42):
    random.seed(seed)
    pos = [x for x in items if any(t != "O" for t in x.get("labels", []))]
    neg = [x for x in items if all(t == "O" for t in x.get("labels", []))]
    random.shuffle(pos); random.shuffle(neg)
    nvp = max(1, int(len(pos) * val_ratio)) if pos else 0
    nvn = max(1, int(len(neg) * val_ratio)) if neg else 0
    val   = pos[:nvp] + neg[:nvn]
    train = pos[nvp:] + neg[nvn:]
    random.shuffle(train); random.shuffle(val)
    return train, val

# ---------- 合并 ----------
def read_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def merge_all_json(data_dir: Path):
    """
    合并 data_dir 下所有潜在的标注文件：
      - 匹配：style_*.json / *_style.json / *.jsonl
      - 排除：style_train.json / style_val.json / style_all.json
      - 自动去重（按 text）
    """
    candidates = list(data_dir.glob("style_*.json")) + \
                 list(data_dir.glob("*_style.json")) + \
                 list(data_dir.glob("*.jsonl"))

    exclude = {TRAIN_FILE.name, VAL_FILE.name, ALL_FILE.name}
    merged = []
    for p in candidates:
        if p.name in exclude:
            continue
        try:
            if p.suffix == ".jsonl":
                part = read_jsonl(p)
            else:
                part = load_json(p)
            if not isinstance(part, list):
                print(f"[WARN] {p} 不是 list，已跳过。")
                continue
            merged.extend(part)
            print(f"[INFO] 载入 {p.relative_to(PROJECT_ROOT)} -> {len(part)} 条")
        except Exception as e:
            print(f"[WARN] 读取 {p} 失败: {e}")

    # 去重
    uniq = {}
    for x in merged:
        if not isinstance(x, dict) or "text" not in x:
            continue
        uniq[x["text"].strip()] = x

    all_data = list(uniq.values())
    save_json(all_data, ALL_FILE)
    print(f"[INFO] 已生成 {ALL_FILE.relative_to(PROJECT_ROOT)}，共 {len(all_data)} 条")
    return all_data

# ---------- 检查 ----------
def check_dataset(train, val):
    tr_pos, tr_tot = sent_pos_ratio(train)
    va_pos, va_tot = sent_pos_ratio(val)

    print("\n=== Sentence-level ===")
    print(f"Train: {tr_tot} | has_BI: {tr_pos}" + ("" if tr_tot==0 else f" ({tr_pos/tr_tot:.2%})"))
    print(f"Val  : {va_tot} | has_BI: {va_pos}"   + ("" if va_tot==0 else f" ({va_pos/va_tot:.2%})"))

    print("\n=== Token-level (approx) ===")
    tr_tok = token_dist(train); va_tok = token_dist(val)
    print("Train:", pretty_dist(tr_tok))
    print("Val  :", pretty_dist(va_tok))

    print("\n=== Leakage check (by text) ===")
    tr_unique, va_unique, inter_n, preview = dup_stats(train, val)
    print(f"unique Train texts: {tr_unique}")
    print(f"unique Val   texts: {va_unique}")
    print(f"intersection n    : {inter_n}")
    if inter_n:
        print("samples (up to 5):")
        for s in preview:
            print(" -", s[:100])

    need_resplit = False
    if va_tot == 0:
        print("[ERROR] 验证集为空。")
        need_resplit = True
    elif va_pos == 0:
        print("[WARN] 验证集没有 B/I 正样本。")
        need_resplit = True
    if inter_n > 0:
        print("[WARN] 训练集与验证集存在重复文本。")
        need_resplit = True

    return need_resplit

# ---------- 主逻辑 ----------
if __name__ == "__main__":
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATA_DIR     = {DATA_DIR}")
    DATA_DIR.mkdir(exist_ok=True)

    # 若 train/val 不存在，则尝试从 all 或合并生成
    if not TRAIN_FILE.exists() or not VAL_FILE.exists():
        print("[INFO] 没有找到 train/val，将生成新的切分...")
        all_data = load_json(ALL_FILE) if ALL_FILE.exists() else merge_all_json(DATA_DIR)

        if not all_data:
            print("\n[ERROR] 没有在 data/ 下找到可合并的数据文件。")
            print("请将标注好的文件放到 data/，文件格式应为 list[{'text','labels','word_ends'}]；")
            sys.exit(1)

        if not validate_dataset(all_data, "style_all.json"):
            sys.exit(1)

        train, val = stratified_split(all_data, VAL_RATIO, SEED)
        save_json(train, TRAIN_FILE)
        save_json(val,   VAL_FILE)
        print("[DONE] 已生成 train/val")
        check_dataset(train, val)
        sys.exit(0)

    # 已有 train/val：检查并在需要时重切
    print("[INFO] 检查现有 train/val...")
    train = load_json(TRAIN_FILE)
    val   = load_json(VAL_FILE)

    if not train and not val:
        print("[ERROR] train/val 均为空。请先把原始标注文件放入 data/ 再运行。")
        sys.exit(1)

    if not validate_dataset(train, "style_train.json") or not validate_dataset(val, "style_val.json"):
        sys.exit(1)

    if check_dataset(train, val):
        print("\n[INFO] 重新合并并分层切分...")
        all_data = load_json(ALL_FILE) if ALL_FILE.exists() else merge_all_json(DATA_DIR)
        if not all_data:
            print("[ERROR] 合并后仍为空，请检查 data/ 下的原始数据文件。")
            sys.exit(1)
        if not validate_dataset(all_data, "style_all.json"):
            sys.exit(1)
        train, val = stratified_split(all_data, VAL_RATIO, SEED)
        save_json(train, TRAIN_FILE)
        save_json(val,   VAL_FILE)
        print("[DONE] 已重新切分 train/val")
        check_dataset(train, val)
    else:
        print("\n[INFO] 数据集切分正常，无需修改。")
