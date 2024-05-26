from typing import List, Union, Optional, Literal
import dataclasses
import tqdm
import random
from LLM_factory.model import Message, LLMModelBase
from LLM_factory.prompt_factory import generic_get_prompts
from LLM_factory.fewshot_generator import generic_create_user_data, generic_create_analysis_messages,generic_create_explaination_messages, generic_create_fewshots, generic_create_prediction_messages
import time
from transformers import AutoTokenizer, AutoModel

from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from zhipuai import ZhipuAI
import os

ZHIPU_API_KEY = "" # your zhipu api key

zhipuClient = ZhipuAI(api_key = ZHIPU_API_KEY)

'''
prompt dictionary keys:
{
    'sys_instr',
    'fewshot_u_ex',
    'fewshot_s_ex',            
    'predict_instr',
    'analysis_instr',
    'explain_instr',
    'self_refl_instr'
}
'''


def log_slice(logs: List, end: int, is_shuffle: bool = True):
    if is_shuffle:
        random.shuffle(logs)
    if end >= len(logs):
        return logs
    return logs[:end]


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def glm_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature:  float = 0.2,
    num_comps=1
) -> Union[List[str], str]:
    # print(model, max_tokens, temperature, top_p, num_comps)
    # print(f"Sending messages to GLM: {[message.content for message in messages]}")
    response = zhipuClient.chat.completions.create(
        model=model,
        messages=[dataclasses.asdict(message) for message in messages],
        max_tokens=max_tokens,
        temperature=temperature
    )

    # set time to wait x seconds for limited requests per minute
    time.sleep(1)
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore

def glm_local_chat(tokenizer, model, message: str, max_length: int = 4096):
    try:
        response = model.chat(tokenizer, message, history=[], max_length=max_length)
    except ValueError as e:
        print(e)
        print(message)
        print(f"message token: {len(tokenizer.encode(message, add_special_tokens=True, return_tensors='pt').squeeze().tolist())}")
    if response[-1][-1]['role'] != 'assistant':
        raise ValueError("last response not assistant")
    return response[-1][-1]['content']

def messages_to_local(tokenizer, messages: List[Message], max_length: int = 3072) -> str:
    local_message = ""
    # every message should be in the format of "<|role|>\n content \n"
    for message in messages:
        local_message += f"<|{message.role}|>\n{message.content}\n"
    # if the token of local_message is max out, truncate it to ensure the last message is messages[-1]
    last_message = f"<|{messages[-1].role}|>\n{messages[-1].content}\n"

    last_message_token = tokenizer.encode(last_message, add_special_tokens=True, return_tensors='pt').squeeze().tolist()
    local_token = tokenizer.encode(local_message, add_special_tokens=True, return_tensors='pt').squeeze().tolist()
    if len(local_token) >= max_length:
        local_token = local_token[:max_length-len(last_message_token)]
        local_message = tokenizer.decode(local_token) + last_message
    return local_message

class GLMChat(LLMModelBase):
    def __init__(self, model_name: str):
        self.name = model_name
        self.is_chat = True
        self.local = False
        if self.name not in ["glm-4", "glm-3-turbo"]:
            self.local = True
            # initialize local glm
            if self.name == "glm-3-6b":
                self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
                self.model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
                self.model = self.model.eval()
            else:
                raise ValueError(f"{self.name} local glm not implemented")

    def generate_chat(self, messages: List[Message], max_tokens: int = 4096, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        if self.local:
            # use local glm_chat function
            local_glm_message = messages_to_local(self.tokenizer, messages)
            return glm_local_chat(self.tokenizer, self.model, local_glm_message)
        else:
            return glm_chat(self.name, messages, max_tokens, temperature, num_comps)


    def generate_prediction(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
        prediction_messages = self.create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
        if self.local:
            # use local glm_chat function
            local_glm_message = messages_to_local(self.tokenizer, prediction_messages)
            return glm_local_chat(self.tokenizer, self.model, local_glm_message)
        else:
            return glm_chat(self.name, prediction_messages, num_comps=1)

    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        analysis_messages = self.create_analysis_messages(fewshots, prompts)
        if self.local:
            # use local glm_chat function
            local_glm_message = messages_to_local(self.tokenizer, analysis_messages)
            return glm_local_chat(self.tokenizer, self.model, local_glm_message)
        else:
            return glm_chat(self.name, analysis_messages, num_comps=1)
    
    def generate_explaination(self, fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
        explaination_messages = self.create_explaination_messages(fewshots, prediction, prompts)
        if self.local:
            # use local glm_chat function
            local_glm_message = messages_to_local(self.tokenizer, explaination_messages)
            return glm_local_chat(self.tokenizer, self.model, local_glm_message)
        else:
            return glm_chat(self.name, explaination_messages, num_comps=1)
    
    def create_prediction_messages(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> List[Message]:
        return generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
    
    def create_analysis_messages(self, fewshots: List[str], prompts: dict) -> List[Message]:
        return generic_create_analysis_messages(fewshots, prompts)

    def create_explaination_messages(self, fewshots: List[str], prediction, prompts: dict) -> List[Message]:
        return generic_create_explaination_messages(fewshots, prediction, prompts)
    
    def create_fewshots(self, student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts) -> List[str]:
        if self.local:
            # do not generate explanations for fewshots
            return generic_create_fewshots(student_info, 
                                        test_exercise_info, 
                                        extra_datas, 
                                        fewshots_num, 
                                        fewshots_strategy, 
                                        data_mode, 
                                        prompts, 
                                        generate_analysis_func=self.generate_analysis,
                                        generate_explanation_func=self.generate_explaination,
                                        generate_analysis=False,
                                        generate_explanation=False)
        else:
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
        return generic_get_prompts('glm', data_mode)

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode) -> str:
        return generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode)

class GLM4(GLMChat):
    def __init__(self):
        super().__init__("glm-4")


class GLM3(GLMChat):
    def __init__(self):
        super().__init__("glm-3-turbo")

class GLM3_6B(GLMChat):
    def __init__(self):
        super().__init__("glm-3-6b")
