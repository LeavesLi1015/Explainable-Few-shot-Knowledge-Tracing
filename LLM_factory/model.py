from typing import List, Union, Optional, Literal
import dataclasses

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai
from openai import OpenAI
from zhipuai import ZhipuAI

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


class LLMModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'
    
    def get_prompts(self, model_name:str, data_mode:str):
        raise NotImplementedError
        # return generic_get_prompt(model_name, data_mode)


    def generate_prediction(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
        raise NotImplementedError

    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        raise NotImplementedError
    
    def generate_explaination(self, fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
        raise NotImplementedError

    def create_analysis_messages(self, fewshots: List[str], prompts: dict) -> List[Message]:
        raise NotImplementedError

    def create_explaination_messages(self, fewshots: List[str], prediction: str, prompts: dict) -> List[Message]:
        raise NotImplementedError

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        raise NotImplementedError

    def create_fewshots(self, student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode) -> List[str]:
        raise NotImplementedError

    def create_prediction_messages(self, fewshot: List[str], user_data: str, prompts: dict) -> List[Message]:
        raise NotImplementedError

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError

