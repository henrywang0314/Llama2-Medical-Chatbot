from typing import List, Optional

from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig

class ChatGLM3(LLM):
    max_token: int = 2048
    temperature: float = 0.95
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
  
    def __init__(self):
        super().__init__()
        model_config = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        # FP 16
        self.model = AutoModel.from_pretrained("THUDM/chatglm3-6b", config=model_config,
                                               trust_remote_code=True).half().cuda()

        # Faster Faster!
        # self.model = AutoModel.from_pretrained("THUDM/chatglm3-6b", config=model_config,
        #                                        trust_remote_code=True).quantize(4).cuda()
    @property
    def _llm_type(self) -> str:
        return "GLM"

    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
            self.tokenizer, prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token, temperature=self.temperature,
            top_p=self.top_p)
        return response