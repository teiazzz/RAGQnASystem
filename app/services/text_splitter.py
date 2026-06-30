"""递归文本切片。

不强依赖 LangChain，保留同类策略：优先按段落/句子边界切分，最后才按固定长度硬切。
默认参数与改造清单一致：chunk_size=500，overlap=50。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class RecursiveTextSplitter:
    chunk_size: int = 500
    overlap: int = 50
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "；", "，", " ")

    def split_text(self, text: str) -> list[str]:
        """返回清洗后的非空 chunk 列表。"""
        text = re.sub(r"[ \t]+", " ", text or "").strip()
        if not text:
            return []
        units = self._split_units(text, self.separators)
        return self._merge_units(units)

    def _split_units(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep = separators[0]
        if sep not in text:
            return self._split_units(text, separators[1:])

        units: list[str] = []
        parts = text.split(sep)
        for idx, part in enumerate(parts):
            if not part:
                continue
            piece = part + (sep if idx < len(parts) - 1 else "")
            if len(piece) > self.chunk_size:
                units.extend(self._split_units(piece, separators[1:]))
            else:
                units.append(piece)
        return units

    def _merge_units(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            if current and current_len + unit_len > self.chunk_size:
                chunk = "".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                overlap_text = chunk[-self.overlap :] if self.overlap > 0 else ""
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(unit)
            current_len += unit_len

        chunk = "".join(current).strip()
        if chunk:
            chunks.append(chunk)
        return chunks
