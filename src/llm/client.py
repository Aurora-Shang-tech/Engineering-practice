import os
from openai import OpenAI

class LLMClient:
    """封装 OpenAI-compatible LLM API"""

    def __init__(self):
        # 从环境变量读取配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL")

        if not self.api_key:
            raise ValueError("缺少环境变量OPENAI_API_KEY")

        if not self.model:
            raise ValueError("缺少环境变量OPENAI_MODEL")

        # 创建OpanAI-compatible客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[dict]) -> str:
        """发送对话请求并返回模型文本"""

        response = self.client.chat.completions.create(
                model = self.model,
                messages=messages,
                temperature=0,
        )

        return response.choices[0].message.content
