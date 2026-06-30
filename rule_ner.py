"""规则版 NER（无需 BERT / torch / transformers）。

原项目的 NER 是 BERT+RNN，需要下载训练权重（百度网盘）+ transformers + torch，
且在 4GB 显存机器上加载困难。这里抽取原 ``ner/inference.py`` 中**不依赖模型**的两段逻辑：

  1. :class:`rule_find`        —— 基于 Aho-Corasick 的词典多模匹配（词典 = data/ent_aug/*.txt）
  2. :class:`tfidf_alignment`  —— 把匹配到的实体名用 TF-IDF 余弦相似度对齐到 KG 标准实体名

精度比 BERT 版略低（识别不了词典外的新词 / 错别字），但对导诊 / 健康问答 demo 足够，
且零额外依赖（仅 ahocorasick + scikit-learn）。

> 注：本文件逻辑直接取自原 ``ner/inference.py``，仅剥离了对 BERT 模型预测和
> ``ner.dataset``（其内部 ``import torch``）的依赖。后续 Phase 1 重构时，
> 这里会被替换为「LLM 直接抽取实体」或正式的 NER 服务。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class rule_find:
    """基于 Aho-Corasick 自动机的多模式实体匹配器。

    词典文件位于 ``data/ent_aug/{type}.txt``，按行存储实体名。
    """

    def __init__(self) -> None:
        self.idx2type = idx2type = [
            "食物", "药品商", "治疗方法", "药品", "检查项目", "疾病", "疾病症状", "科目",
        ]
        self.type2idx = type2idx = {t: i for i, t in enumerate(idx2type)}
        self.ahos = [ahocorasick.Automaton() for _ in range(len(self.type2idx))]

        for type in idx2type:
            with open(os.path.join("data", "ent_aug", f"{type}.txt"), encoding="utf-8") as f:
                all_en = f.read().split("\n")
            for en in all_en:
                en = en.split(" ")[0]
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self, sen: str) -> List[Tuple[int, int, str, str]]:
        """对句子做规则匹配，返回 ``[(begin, end, type, word), ...]``，已去除位置重叠。"""
        rule_result: List[Tuple[int, int, str, str]] = []
        mp: dict = {}
        all_res: list = []
        all_ty: list = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for _ in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


class tfidf_alignment:
    """把 NER 抽出的实体名通过 TF-IDF 余弦相似度对齐到 KG 标准实体名。"""

    def __init__(self) -> None:
        eneities_path = os.path.join("data", "ent_aug")
        files = os.listdir(eneities_path)
        files = [docu for docu in files if ".py" not in docu]

        self.tag_2_embs: dict = {}
        self.tag_2_tfidf_model: dict = {}
        self.tag_2_entity: dict = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), "r", encoding="utf-8") as f:
                entities = f.read().split("\n")
                entities = [ent for ent in entities if 1 <= len(ent.split(" ")[0]) <= 15]
                en_name = [ent.split(" ")[0] for ent in entities]
                ty = ty.strip(".txt")
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list) -> Dict[str, str]:
        """对齐实体；相似度 ≥ 0.5 的才保留。返回 ``{type: 标准实体名}``。"""
        new_result: dict = {}
        for s, e, cls, ent in ent_list:
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]
            if max_score >= 0.5:
                new_result[cls] = self.tag_2_entity[cls][max_idx]
        return new_result


def get_ner_result(sen: str, rule: rule_find, tfidf_r: tfidf_alignment) -> Dict[str, str]:
    """规则版 NER 入口：词典多模匹配 → TF-IDF 对齐到标准实体名。

    :return: ``{实体类型: 标准实体名}``，例如 ``{"疾病": "高血压"}``
    """
    rule_result = rule.find(sen)
    return tfidf_r.align(rule_result)
