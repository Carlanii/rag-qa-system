"""本地 Embedding 模型（基于 transformers，无需外部 API）"""
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """使用 transformers 的本地 embedding 模型"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is not None:
            return
        print(f"正在加载 embedding 模型: {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        print("Embedding 模型加载完成")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        if torch.cuda.is_available():
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        with torch.no_grad():
            model_output = self.model(**encoded)
        sentence_embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
