from typing import Optional, List

import requests

from langchain.embeddings.base import Embeddings

from lib.config import config_get_item

from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any

import os
from openai import OpenAI
from langchain_core.outputs import LLMResult, Generation

class DeepSeekLLM(LLM):

    deepseek_api_key: str
    api_generate_url: str
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """向 DeepSeek API 发送请求并返回响应"""

        client = OpenAI(api_key=self.deepseek_api_key, base_url=self.api_generate_url)


        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        print("---------------------")
        print(response.choices[0].message.content)
        print("---------------------")
        return response.choices[0].message.content

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """返回 LLMResult，而不是字符串"""
        print("0000000000000000000000000000000000000000000000")
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])  # 必须封装在 `Generation` 里

        return LLMResult(generations=generations)  # 返回 LLMResult

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型的参数信息"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    @property
    def _llm_type(self) -> str:
        return "deepseek-llm"
