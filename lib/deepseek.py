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

        return response.choices[0].message.content

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """返回 LLMResult，而不是字符串"""
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



class DeepSeekLLM_Agent(LLM):

    deepseek_api_key: str
    api_generate_url: str
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """向 DeepSeek API 发送请求并返回响应"""

        client = OpenAI(api_key=self.deepseek_api_key, base_url=self.api_generate_url)

        formatted_prompt = f"""
            你是一个智能代理，需要按照以下格式回答：
            ---
            Thought: 你的推理过程
            Action: 需要调用的工具名称 (如果不需要工具，则不包含该部分)
            Action Input: 需要传递给工具的参数 (如果不需要工具，则不包含该部分)
            Final Answer: 你的最终答案 (如果已经得到答案，不需要工具)
            ---
            用户问题：
            {prompt}
            """
        messages = [
            {"role": "system", "content": "请务必使用 ReAct 结构回答。"},
            {"role": "user", "content": formatted_prompt}
]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )

        content = response.choices[0].message.content.strip()

        print("-------------")
        print(content)
        print("-------------")

        if "Action:" in content and "Final Answer:" in content:
        # **如果同时有 Action 和 Final Answer，只保留 Action**
            result = content.split("Final Answer:")[0].strip()
            return result
        elif "Action:" in content:
            return content
        else:
            return "Final Answer: 无法解析正确的回答。"  # 避免 LangChain 崩溃


    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        """返回 LLMResult，而不是字符串"""
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