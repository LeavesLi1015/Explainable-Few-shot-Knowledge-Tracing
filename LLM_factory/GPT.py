from typing import List, Union, Optional, Literal
import dataclasses
import tqdm
import random
from LLM_factory.model import Message, LLMModelBase
from LLM_factory.prompt_factory import generic_get_prompts
from LLM_factory.fewshot_generator import generic_create_user_data, generic_create_analysis_messages,generic_create_explaination_messages, generic_create_fewshots, generic_create_prediction_messages
import time

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
import openai
from openai import OpenAI
from openai import AzureOpenAI

import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_BASE_URL"] = ""

gpt_client = OpenAI()

@retry(wait=wait_random_exponential(min=60, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 4096,
    temperature: float = 0.2,
    num_comps=1,
) -> Union[List[str], str]:
    response = gpt_client.chat.completions.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )

    # set time to wait 6 seconds for limit of 10 requests per minute
    time.sleep(3)

    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore



class GPTChat(LLMModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        return gpt_chat(self.name, messages, max_tokens, temperature, num_comps)

    def generate_prediction(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
        prediction_messages = self.create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
        return gpt_chat(self.name, prediction_messages, num_comps=1)

    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        analysis_messages = self.create_analysis_messages(fewshots, prompts)
        return gpt_chat(self.name, analysis_messages, num_comps=1)
    
    def generate_explaination(self, fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
        explaination_messages = self.create_explaination_messages(fewshots, prediction, prompts)
        return gpt_chat(self.name, explaination_messages, num_comps=1)
    
    def create_prediction_messages(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> List[Message]:
        return generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
    
    def create_analysis_messages(self, fewshots: List[str], prompts: dict) -> List[Message]:
        return generic_create_analysis_messages(fewshots, prompts)

    def create_explaination_messages(self, fewshots: List[str], prediction, prompts: dict) -> List[Message]:
        return generic_create_explaination_messages(fewshots, prediction, prompts)
    
    def create_fewshots(self, student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts) -> List[str]:
        return generic_create_fewshots(student_info, 
                                       test_exercise_info, 
                                       extra_datas, 
                                       fewshots_num, 
                                       fewshots_strategy, 
                                       data_mode, 
                                       prompts, 
                                       self.generate_analysis,
                                       self.generate_explaination)

    def get_prompts(self, data_mode: str):
        return generic_get_prompts('gpt', data_mode)
    
    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        return generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode)

class GPT4(GPTChat):
    def __init__(self):
        super().__init__("gpt-4")


class GPT35(GPTChat):
    def __init__(self):
        super().__init__("gpt-3.5-turbo")
