from typing import Optional, List

import requests

from langchain.embeddings.base import Embeddings
from langchain.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation

from lib.config import config_get_item

class CustomEmbeddings(Embeddings):
    def __init__(self):
        """
        初始化自定义 embedding 模型
        :param api_url: 网络 API 的地址
        :param api_key: API 密钥
        """
        self.api_generate_url = config_get_item("eli", "eli_generate")
        self.api_embedding_url = config_get_item("eli", "eli_embedding")
        self.api_key = config_get_item("eli", "eli_api_key")

    def _embed(self, texts: list[str], endpoint: str) -> list[list[float]]:
        """
        调用网络 API 获取 embedding
        :param texts: 要处理的文本列表
        :param endpoint: API 的具体 endpoint（区分文档或查询）
        :return: 嵌入向量列表
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "eli-embedding-small-1",
            "texts": texts,
        }

        response = requests.post(
            self.api_embedding_url,
            json=payload,
            headers=headers,
            timeout=(300, 300),
            verify=False,
        )
        
        if response.status_code == 200:
            return response.json()['vectors']
        else:
            raise ValueError(f"Embedding API 请求失败: {response.status_code} - {response.text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        文档的 embedding
        """
        print("eli embed_documents was called")
        return self._embed(texts, endpoint="api/v1/bi_encoder/encode")

    def embed_query(self, text: str) -> list[float]:
        """
        查询的 embedding
        """
        print("eli embed_query was called")
        return self._embed([text], endpoint="api/v1/bi_encoder/encode")[0]


class CustomLLM(BaseLLM):

    api_generate_url: str  # 你的 API 地址
    api_key: Optional[str] = None  # 可能需要的 API 密钥
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """调用 API 并返回结果"""
        token_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json"
        }

        payload = {
            "prompt":prompt,
            "model":"deepseekr1-14b",
            "max_new_tokens":256,
            "temperature":0,
            "max_suggestions":0,
            "top_p":0,
            "top_k":0,
            "stop_seq":"string",
            "client":"string",
            "stream":False,
            "stream_batch_tokens":10
        }
        
        response = requests.post(self.api_generate_url, json=payload, headers=token_headers, verify=False)

        if response.status_code == 200:
            print(response.json())
            return response.json().get("completions")[0]  # 根据 API 返回格式调整
        else:
            raise ValueError(f"API 请求失败: {response.text}")
        
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """返回 LLMResult，而不是字符串"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])  # 必须封装在 `Generation` 里

        return LLMResult(generations=generations)  # 返回 LLMResult
    
    @property
    def _identifying_params(self):
        """返回唯一标识此 LLM 的参数"""
        return {"api_url": self.api_generate_url}

    @property
    def _llm_type(self) -> str:
        return "custom-api"