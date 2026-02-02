# models/style_ner.py
"""
Style NER model for fashion style extraction
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification

# optional: rapidfuzz for lexicon alignment
try:
    from rapidfuzz import fuzz  # noqa
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

_CLEAN_RE = re.compile(r"[^\w\s-]")                 # keep letters/digits/underscore/space/hyphen
_ALPHA_RE = re.compile(r"^[a-zA-Z][a-zA-Z\s\-]*$")  # letters/spaces/hyphen only; must start with a letter


class StyleNER:
    """BERT-based Style Named Entity Recognition"""

    def __init__(
            self,
            model_name: str = "bert-base-uncased",
            num_labels: int = 3,
            local_model_path: Optional[Union[str, Path]] = None,
            label_map: Optional[Dict[str, int]] = None,
            lexicon_path: Optional[Union[str, Path]] = None,
            lexicon_min_ratio: int = 80,
            min_phrase_len: int = 3,
            max_words: int = 2,   # limit max words in a style phrase
    ):
        """
        Args:
            model_name: HF model name used when no finetuned weights exist.
            num_labels: number of labels (O/B-STYLE/I-STYLE).
            local_model_path: directory for finetuned weights/tokenizer; default = <PROJECT_ROOT>/models/style_ner
            label_map: custom label map; default {"O":0,"B-STYLE":1,"I-STYLE":2}.
            lexicon_path: optional style lexicon file; one style per line.
            lexicon_min_ratio: min token_set_ratio(phrase, lexicon_entry) to accept (0-100).
            min_phrase_len: minimal cleaned phrase length.
            max_words: maximal number of tokens allowed in a style phrase.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] device={self.device}")

        # labels
        self.label_map = label_map or {"O": 0, "B-STYLE": 1, "I-STYLE": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # >>> Resolve PROJECT_ROOT & absolute default dir
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        default_dir = PROJECT_ROOT / "models" / "style_ner"
        self.model_name = model_name
        self.local_model_path = Path(local_model_path).resolve() if local_model_path else default_dir.resolve()
        self.local_model_path.mkdir(parents=True, exist_ok=True)

        # tokenizer
        local_has_tokenizer = (self.local_model_path / "tokenizer.json").exists()
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(self.local_model_path) if local_has_tokenizer else model_name
        )

        # model
        if (self.local_model_path / "pytorch_model.bin").exists():
            print(f"[INFO] loading StyleNER from {self.local_model_path}")
            self.model = BertForTokenClassification.from_pretrained(
                str(self.local_model_path), num_labels=num_labels
            )
        else:
            print(f"[INFO] initializing new StyleNER from {model_name}")
            self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)

        # filters
        self.min_phrase_len = int(min_phrase_len)
        self.max_words = int(max_words)
        self.lexicon_min_ratio = int(lexicon_min_ratio)

        # stopwords (expanded)
        self.non_style_words = {
            # generic
            "style", "look", "design", "fashion", "apparel", "wear", "outfit",
            "and", "or", "for", "with", "the", "a", "an", "of", "to", "in", "on",
            "is", "are", "was", "were", "it", "this", "that", "these", "those",
            "i", "me", "my", "you", "your", "we", "our", "they", "their",
            "any", "some", "please", "show", "find", "search", "want", "need",
            "under", "over", "between", "around", "about", "approx", "approximately", "dollars", "dollar",
            # categories (not styles)
            "jacket", "jackets", "coat", "coats", "dress", "dresses", "shoe", "shoes",
            "sneaker", "sneakers", "boot", "boots", "jeans", "pants", "shirt", "shirts",
            "skirt", "skirts", "hoodie", "hoodies", "sweater", "sweaters",
            # frequent subword debris
            "y", "##y", "##s", "##er", "##es",
        }

        # optional lexicon (default: <PROJECT_ROOT>/data/style_lexicon.txt)
        self.lexicon = None  # type: Optional[List[str]]
        if lexicon_path is None:
            default_lex = PROJECT_ROOT / "data" / "style_lexicon.txt"
            if default_lex.is_file():
                lexicon_path = default_lex
        if lexicon_path and Path(lexicon_path).is_file():
            self.lexicon = self._load_lexicon(lexicon_path)
            print(f"[INFO] loaded style lexicon: {lexicon_path} ({len(self.lexicon)} entries)")
        elif lexicon_path:
            print(f"[WARN] lexicon file not found: {lexicon_path}")

    # ---------------- Public helpers ----------------
    def add_extra_tokens(self, tokens: List[str]) -> int:
        """
        Optionally add high-frequency style tokens to reduce WordPiece splits.
        Call this *before* training and inference (ensure same tokenizer).
        Returns the number of tokens actually added.
        """
        if not tokens:
            return 0
        tokens = [t for t in tokens if isinstance(t, str) and t]
        n = self.tokenizer.add_tokens(tokens)
        if n > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"[INFO] added {n} tokens to tokenizer (new vocab size={len(self.tokenizer)})")
        return n

    # ---------------- Save/Load ----------------
    def save(self, save_dir: Union[str, Path]):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        with open(save_dir / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)
        print(f"[INFO] StyleNER saved to: {save_dir}")

    @classmethod
    def load(cls, load_dir: Union[str, Path], **kwargs):
        load_dir = Path(load_dir)
        with open(load_dir / "label_map.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)
        return cls(local_model_path=load_dir, num_labels=len(label_map), label_map=label_map, **kwargs)

    # ---------------- Inference ----------------
    def extract_styles(
            self,
            text: str,
            return_confidence: bool = False,
            threshold: float = 0.0
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Extract style phrases using char-span merging.
        - Uses offset_mapping to glue WordPiece parts.
        - If a token starts with '##' but labeled as B-STYLE, treat as I-STYLE.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
            prob = torch.softmax(out.logits, dim=2).squeeze(0).cpu().numpy()  # [seq, num_labels]
            pred_ids = np.argmax(prob, axis=1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
        labels = [self.reverse_label_map[i] for i in pred_ids]

        spans: List[Tuple[int, int, List[float]]] = []
        cur_start, cur_end, cur_probs = -1, -1, []

        for idx, (tok, lab, (st, ed)) in enumerate(zip(tokens, labels, offsets)):
            if st == 0 and ed == 0:
                continue
            if lab == "O":
                if cur_start != -1:
                    spans.append((cur_start, cur_end, cur_probs))
                    cur_start, cur_end, cur_probs = -1, -1, []
                continue

            if tok.startswith("##") and lab == "B-STYLE":
                lab = "I-STYLE"

            if lab == "B-STYLE":
                if cur_start != -1:
                    spans.append((cur_start, cur_end, cur_probs))
                cur_start, cur_end = st, ed
                cur_probs = [float(prob[idx, self.label_map["B-STYLE"]])]
            elif lab == "I-STYLE":
                if cur_start == -1:
                    cur_start, cur_end = st, ed
                    cur_probs = [float(prob[idx, self.label_map["I-STYLE"]])]
                else:
                    cur_end = ed
                    cur_probs.append(float(prob[idx, self.label_map["I-STYLE"]]))

        if cur_start != -1:
            spans.append((cur_start, cur_end, cur_probs))

        # materialize → clean → filter → optional lexicon alignment
        candidates: List[Tuple[str, float]] = []
        for st, ed, pr in spans:
            chunk = text[st:ed]
            if not chunk.strip():
                chunk = self._fallback_merge(tokens, labels)
            phrase = _CLEAN_RE.sub("", (chunk or "")).strip().lower()
            if not phrase:
                continue

            phrase = re.sub(r"\s+", " ", phrase)
            phrase = re.sub(r"-+", "-", phrase).strip("- ").strip()

            if not self._is_candidate(phrase):
                continue

            conf = float(np.mean(pr)) if pr else 0.0
            if conf >= threshold:
                if self.lexicon and _HAS_RAPIDFUZZ:
                    best = self._lexicon_align(phrase)
                    if best is None:
                        continue
                    phrase = best
                candidates.append((phrase, conf))

        # dedup by max confidence
        dedup: Dict[str, float] = {}
        for ph, sc in candidates:
            dedup[ph] = max(dedup.get(ph, 0.0), sc)
        results = sorted(dedup.items(), key=lambda x: x[1], reverse=True)

        return results if return_confidence else [ph for ph, _ in results]

    # ---------------- utils ----------------
    def _is_candidate(self, phrase: str) -> bool:
        """Check cleaned phrase against stopwords, min length, alpha-only policy, and max words."""
        if not phrase or len(phrase) < self.min_phrase_len:
            return False
        if not _ALPHA_RE.match(phrase.replace("_", " ")):
            return False
        if phrase in self.non_style_words:
            return False
        tokens = [t for t in re.split(r"\s+", phrase) if t]
        # limit to at most N words (default 2)
        if self.max_words and len(tokens) > self.max_words:
            return False
        if tokens and all(t in self.non_style_words for t in tokens):
            return False
        return True

    def _lexicon_align(self, phrase: str) -> Optional[str]:
        """Return best-matching lexicon phrase if score >= lexicon_min_ratio, else None."""
        if not self.lexicon:
            return phrase
        if not _HAS_RAPIDFUZZ:
            return phrase
        best, best_score = None, -1
        for entry in self.lexicon:
            score = fuzz.token_set_ratio(phrase, entry)
            if score > best_score:
                best, best_score = entry, score
        if best_score >= self.lexicon_min_ratio:
            return best
        return None

    def _load_lexicon(self, path: Union[str, Path]) -> List[str]:
        lex = []
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().lower()
                if not s:
                    continue
                s = _CLEAN_RE.sub("", s)
                s = re.sub(r"\s+", " ", s).strip("- ").strip()
                if s and _ALPHA_RE.match(s.replace("_", " ")):
                    lex.append(s)
        return sorted(set(lex))

    def _fallback_merge(self, tokens: List[str], labels: List[str]) -> str:
        """Rare fallback: merge consecutive B/I tokens with WordPiece rule (## prefix)."""
        buf, collecting = [], False
        for tok, lab in zip(tokens, labels):
            if lab.startswith("B"):
                if buf:
                    break
                buf.append(tok)
                collecting = True
            elif lab.startswith("I") and collecting:
                buf.append(tok)
            elif collecting:
                break
        return self.merge_subwords(buf) or ""

    @staticmethod
    def merge_subwords(tokens: List[str]) -> Optional[str]:
        """Merge subword tokens into a complete word/phrase (WordPiece aware)."""
        merged = ""
        for tok in tokens:
            if tok.startswith("##"):
                merged += tok[2:]
            else:
                merged += ("" if not merged else " ") + tok
        merged = _CLEAN_RE.sub("", merged).strip()
        merged = re.sub(r"\s+", " ", merged)
        merged = re.sub(r"-+", "-", merged).strip("- ").strip()
        return merged.lower() if merged else None


class StyleNERDataset(Dataset):
    """Dataset for Style NER with label alignment using offset_mapping (fixed B/I for subwords)"""

    def __init__(self, data: List[Dict], tokenizer: BertTokenizerFast,
                 label_map: Dict[str, int], max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        # cache ids
        self._O = self.label_map.get("O", 0)
        self._B = self.label_map.get("B-STYLE", 1)
        self._I = self.label_map.get("I-STYLE", 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text: str = item["text"]
        word_labels: List[str] = item["labels"]      # word-level labels
        word_ends: List[int] = item["word_ends"]     # right boundary (char index) for each word

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        aligned = []
        cur_word = -1
        last_word_for_token = -1  # to decide B or I for subwords inside same word

        for (start, end) in offsets:
            # special/padding tokens -> ignore
            if start == 0 and end == 0:
                aligned.append(-100)
                continue

            # move to the word that covers [start, end)
            if cur_word == -1 or start >= word_ends[cur_word]:
                cur_word += 1
                last_word_for_token = -1  # reset when entering a new word

            if cur_word < 0 or cur_word >= len(word_labels):
                aligned.append(self._O)
                continue

            base_lab = word_labels[cur_word]
            if base_lab == "O":
                tok_lab_id = self._O
            else:
                # first subword of the word -> B; subsequent subwords -> I
                if last_word_for_token != cur_word:
                    tok_lab_id = self._B
                    last_word_for_token = cur_word
                else:
                    tok_lab_id = self._I

            aligned.append(tok_lab_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(aligned, dtype=torch.long),
        }
