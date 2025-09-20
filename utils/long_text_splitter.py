import re, numpy as np
from typing import List
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer


# ---------- helpers ----------
END_PUNCT = "…!?."  # hard sentence boundaries
SOFT_PUNCT = ",;:—-" # soft split points for long sentences
MAX_CHARS = 160 # Maximum for "ru" BpeTokenizer 182 minus margin
MAX_XTTS_TOKENS = 400

class LongTextSplitter:
    def __init__(self):
        self.tokenizer = VoiceBpeTokenizer(vocab_file="./NeuroVoice/models/tts/base/coqui/XTTS-v2/vocab.json")
    def _ru_abbrev_patterns(self):
        # basic RU abbreviations we don't want to split on
        # (keep minimal to avoid false negatives)
        return [
            r"т\.\s*д\.", r"т\.\s*п\.", r"т\.\s*е\.", r"т\.\s*к\.",
            r"и\.\s*т\.\s*д\.", r"и\.\s*т\.\s*п\.",
            r"г\.", r"ул\.", r"рис\.", r"др\.", r"им\.", r"стр\."
        ]

    def sentence_split_ru(self, text) -> List[str]:
        """Conservative RU sentence splitter that respects common abbreviations."""
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return []

        # Temporarily protect abbreviations by replacing their dots with U+2024 (one-dot leader)
        protected = {}
        for i, pat in enumerate(self._ru_abbrev_patterns()):
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                key = f"@@ABBR{i}_{m.start()}@@"
                protected[key] = m.group(0).replace(".", "․")  # U+2024
                text = text[:m.start()] + key + text[m.end():]

        # Split on ., !, ?, … followed by space+capital/quote/end
        parts = re.split(r'(?<=[\.\!\?\…])\s+(?=[«"“(]*[A-ZА-ЯЁ0-9])', text)

        # Restore abbreviations
        rec = []
        for p in parts:
            for k, v in protected.items():
                p = p.replace(k, v)
            rec.append(p.strip())
        # Merge stray pieces lacking terminal punctuation into next piece
        merged = []
        for s in rec:
            if not merged:
                merged.append(s)
                continue
            if merged[-1] and merged[-1][-1] not in END_PUNCT:
                merged[-1] = (merged[-1] + " " + s).strip()
            else:
                merged.append(s)
        return [s for s in merged if s]

    def token_len(self, sent, language: str) -> int:
        return len(self.tokenizer.encode(sent, lang=language))

    def subsegment_long_sentence(self, s: str, max_chars: int, language: str,
                                max_tokens: int) -> List[str]:
        """Split a single long sentence by soft punctuation, then by words if needed."""
        if len(s) <= max_chars and self.token_len(s, language) <= max_tokens:
            return [s]

        # First pass: split by SOFT_PUNCT while keeping delimiters
        chunks = []
        cur = ""
        for tok in re.split(f"([{re.escape(SOFT_PUNCT)}])", s):
            if not tok: continue
            candidate = (cur + tok).strip()
            if len(candidate) <= max_chars and self.token_len(candidate, language) <= max_tokens:
                cur = candidate
            else:
                if cur: chunks.append(cur.strip().rstrip(",;:—-"))
                cur = tok.strip()
        if cur: chunks.append(cur.strip().rstrip(",;:—-"))

        # If some pieces are still too big, fallback to word-wrapping
        final = []
        for piece in chunks:
            if len(piece) <= max_chars and self.token_len(piece, language) <= max_tokens:
                final.append(piece)
                continue
            words = piece.split()
            buf = []
            for w in words:
                cand = (" ".join(buf+[w])).strip()
                if len(cand) <= max_chars and self.token_len(cand, language) <= max_tokens:
                    buf.append(w)
                else:
                    if buf:
                        final.append(" ".join(buf))
                    buf = [w]
            if buf:
                final.append(" ".join(buf))
        # ensure end punctuation for prosody
        out = []
        for i, seg in enumerate(final):
            seg = seg.strip()
            if seg and seg[-1] not in END_PUNCT:
                seg = seg + "."
            out.append(seg)
        return out

    def pack_by_budget(self, segments: List[str], language: str,
                    max_chars: int, max_tokens: int) -> List[str]:
        """Greedily pack sentence pieces into chunks under both budgets."""
        chunks, buf = [], ""
        cur_tok = 0
        for s in segments:
            sep = "" if (buf == "" or buf.endswith(" ")) else " "
            cand = (buf + sep + s).strip() if buf else s
            cand_tok = self.token_len(cand, language)
            if len(cand) <= max_chars and cand_tok <= max_tokens:
                buf, cur_tok = cand, cand_tok
            else:
                if buf:
                    if buf[-1] not in END_PUNCT: buf += "."
                    chunks.append(buf)
                # s might still be too big (shouldn’t, but guard)
                if len(s) > max_chars or self.token_len(s, language) > max_tokens:
                    # split s further
                    subs = self.subsegment_long_sentence(s, max_chars, language, max_tokens)
                    for sub in subs:
                        if len(sub) <= max_chars and self.token_len(sub, language) <= max_tokens:
                            chunks.append(sub)
                        else:
                            # last-resort word chop
                            words = sub.split()
                            left = []
                            for w in words:
                                cand2 = (" ".join(left+[w])).strip()
                                if len(cand2) <= max_chars and self.token_len(cand2, language) <= max_tokens:
                                    left.append(w)
                                else:
                                    if left: chunks.append(" ".join(left)+".")
                                    left = [w]
                            if left: chunks.append(" ".join(left)+".")
                    buf, cur_tok = "", 0
                else:
                    buf, cur_tok = s, self.token_len(s, language)
        if buf:
            if buf[-1] not in END_PUNCT: buf += "."
            chunks.append(buf)
        return chunks

    def chunk_text_for_xtts(self, text, language: str="ru",
                            target_char_limit: int=MAX_CHARS,
                            token_margin: int=16) -> List[str]:
        """Top-level splitter that respects XTTS constraints."""
        max_tokens = MAX_XTTS_TOKENS - token_margin
        if max_tokens < 50: max_tokens = 50  # sanity
        # 1) coarse sentences
        sents = self.sentence_split_ru(text)
        # 2) sub-segment any long sentence to fit individual budgets
        segs = []
        for s in sents:
            segs.extend(self.subsegment_long_sentence(s, target_char_limit, language, max_tokens))
        # 3) pack segments into final chunks under budgets
        chunks = self.pack_by_budget(segs, language, target_char_limit, max_tokens)
        return chunks

