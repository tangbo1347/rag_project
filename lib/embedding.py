
import os

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

from lib.config import config_get_item

class TransformersEmbeddings(Embeddings):
    def __init__(self):
        """
        初始化自定义 embedding 模型
        :param api_url: 网络 API 的地址
        :param api_key: API 密钥
        """
        local_model_path = config_get_item("embedding", "model")
        self.model = SentenceTransformer(local_model_path)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        调用网络 API 获取 embedding
        :param texts: 要处理的文本列表
        :param endpoint: API 的具体 endpoint（区分文档或查询）
        :return: 嵌入向量列表
        """
        try:
            embeddings = self.model.encode(texts).tolist()

            return embeddings
        except:
            raise


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        文档的 embedding
        """
        print("local embed_documents was called")
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        查询的 embedding
        """
        print("local embed_query was called")
        return self._embed([text])[0]