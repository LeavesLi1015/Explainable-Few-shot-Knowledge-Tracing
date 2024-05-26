from typing import List, Union, Optional, Literal
import dataclasses
import tqdm
import random
from LLM_factory.model import Message, LLMModelBase
import pandas as pd
from utils import sample_fs_id


def generic_create_analysis_messages(fewshots: List[str], prompts: dict) -> List[Message]:
    # create analysis messages for the prompt
    messages = []
    if len(fewshots) == 0:
        raise ValueError("fewshots should not be empty, when creating analysis messages")
    messages.append(
        Message(
            role='system',
            content=prompts['sys_instr']
        )
    )
    messages.append(
        Message(
            role='user',
            content='\n'.join(fewshots) + '\n' + prompts['analysis_instr']
        )
    )
    return messages

def generic_create_explaination_messages(fewshots: List[str], prediction: str, prompts: dict) -> List[Message]:
    # create explanation messages for the prediction
    messages = []
    if len(fewshots) == 0:
        raise ValueError("fewshots should not be empty, when creating explanation messages")

    messages.append(
        Message(
            role='system',
            content=prompts['sys_instr']
        )
    )
    if len(fewshots) > 1:
        messages.append(
            Message(
                role='user',
                content="Previous exercise information:" + '\n'.join(fewshots[:-1])
            )
        )
    messages.append(
        Message(
            role='user',
            content="Current exercise information:" + '\n'.join(fewshots[-1]) + f"The student answer is: {'right' if prediction == '1' else 'wrong'}."
        )
    )
    messages.append(
        Message(
            role='user',
            content=prompts['explain_instr']
        )
    )
    return messages

def generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot=True) -> List[Message]:
    if len(fewshots) > 0: # few-shot
        messages = [
            Message(
                role="system",
                content=prompts['sys_instr'],
            )
        ]
        if use_selected_fewshot:
            fs_content = '\n'.join(fewshots)
            messages.append(
                Message(
                    role="user",
                    content=fs_content + '\n' + user_data + '\n' + prompts['predict_instr'],
                )
            )
        else:
            messages.append(
                Message(
                    role="user",
                    content=prompts['fewshot_u_ex'] + "\n" + prompts['predict_instr'] + '\n',
                )
            )
            messages.append(
                Message(
                    role="assistant",
                    content=prompts['fewshot_s_ex']
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=user_data + "\n" + prompts['predict_instr'],
                )
            )
        # print_messages(messages[0].content, messages[-1].content)
    else: # zero-shot
        messages = [
            Message(
                role="system",
                content=prompts['sys_instr'],
            ),
            Message(
                role="user",
                content=user_data + "\n" + prompts['predict_instr'],
            ),
        ]
        # print_messages(messages[0].content, messages[-1].content)
    return messages

def generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode) -> str:
    # create prediction data for a test exercise to send to llm
    # data_mode: "onehot", "sparse", "moderate", "rich"
    user_data = ''
    if data_mode == "onehot":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise_id: {test_exercise_info['exercise_id']}, knowledge concepts: {test_exercise_info['skill_ids']}\n"
    elif data_mode == "sparse":
        user_data += '<Exercise to Predict>\n'
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "moderate":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise content: {test_exercise_info['exercise_desc']}\n"
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "rich":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}, when creating user data")
    
    user_data += '<Output Predicted is_correct>\n'
    return user_data

def generic_create_fewshots(student_info, 
                            test_exercise_info, 
                            extra_datas, 
                            fewshots_num, 
                            fewshots_strategy, 
                            data_mode, 
                            prompts,
                            generate_analysis_func, 
                            generate_explanation_func, 
                            generate_analysis = True, 
                            generate_explanation = True) -> List[str]:
    if data_mode == "onehot":
        return generic_create_onehot_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis_func, generate_explanation_func, generate_analysis, generate_explanation)
    elif data_mode == "sparse":
        return generic_create_sparse_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis_func, generate_explanation_func, generate_analysis, generate_explanation)
    elif data_mode == "moderate":
        return generic_create_moderate_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis_func, generate_explanation_func, generate_analysis, generate_explanation)
    elif data_mode == "rich":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}, when creating fewshots")

def generic_create_onehot_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                   prompts, generate_analysis_func, generate_explanation_func,
                                   generate_analysis = True, generate_explanation = True) -> List[str]:
    # new create_fewshots function for llmsforkt
    ret_fewshots = []
    # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
    fs_ex_ids = sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
    if fewshots_num == 0:
        return ret_fewshots
    first = True

    for i, ex_id in fs_ex_ids:
        # get all the info of exercise and convert it to string
        fewshot = ''
        # add onehot info
        other_ex_info = extra_datas['exercise_info']
        ex_id_skill_ids = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_ids'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'
        # generate explaination of the fewshot
        if first:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_ids}, is_correct: {is_correct}\n"
            print(f'initializing fewshot with exercise {ex_id}')
            first = False
        else:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_ids}, is_correct: {is_correct}\n"
            if generate_explanation:
                print(f'generating explaination for exercise {ex_id} in fewshot')
                explanation = generate_explanation_func(ret_fewshots, test_exercise_info['is_correct'], prompts)
                fewshot += f"Explaination: {explanation}\n"
                # print(explanation)
                # exit()

        ret_fewshots.append(fewshot)

    # print('\n'.join(ret_fewshots))
    return ret_fewshots

def generic_create_sparse_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                   prompts, generate_analysis_func, generate_explanation_func,
                                   generate_analysis = True, generate_explanation = True) -> List[str]:
    # new create_fewshots function for llmsforkt
    ret_fewshots = []
    # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
    fs_ex_ids = sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
    if fewshots_num == 0:
        return ret_fewshots
    first = True

    for i, ex_id in fs_ex_ids:
        # get all the info of exercise and convert it to string
        fewshot = ''
        # add onehot info
        other_ex_info = extra_datas['exercise_info']
        ex_id_skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'
        # generate explaination of the fewshot
        if first:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"
            print(f'initializing fewshot with exercise {ex_id}')
            first = False
        else:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"
            if generate_explanation:
                print(f'generating explaination for exercise {ex_id} in fewshot')
                explanation = generate_explanation_func(ret_fewshots, test_exercise_info['is_correct'], prompts)
                fewshot += f"Explaination: {explanation}\n"
                # print(explanation)
                # exit()

        ret_fewshots.append(fewshot)

    # print('\n'.join(ret_fewshots))
    return ret_fewshots

def generic_create_moderate_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                   prompts, generate_analysis_func, generate_explanation_func,
                                   generate_analysis = True, generate_explanation = True) -> List[str]:
    
    # new create_fewshots function for llmsforkt
    ret_fewshots = []
    # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
    fs_ex_ids = sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
    if fewshots_num == 0:
        return ret_fewshots
    first = True

    for i, ex_id in fs_ex_ids:
        # get all the info of exercise and convert it to string
        fewshot = ''
        # add onehot info
        other_ex_info = extra_datas['exercise_info']
        ex_id_skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'
        ex_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['exercise_desc'].values[0]
        fewshot += f"exercise_id: {ex_id}\nexercise content: {ex_desc}\nknowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"    
        # generate explaination of the fewshot
        if first:
            print(f'initializing fewshot with exercise {ex_id}')
            # can change to not generate explanation for the first fewshot
            # if generate_explanation:
            #     print(f'generating explaination for exercise {ex_id} in fewshot')
            #     explanation = generate_explanation_func(fewshot, test_exercise_info['is_correct'], prompts)
            #     fewshot += f"Explaination: {explanation}\n"
            first = False
        else:
            if generate_explanation:
                print(f'generating explaination for exercise {ex_id} in fewshot')
                explanation = generate_explanation_func(ret_fewshots, test_exercise_info['is_correct'], prompts)
                fewshot += f"Explaination: {explanation}\n"
                # print(explanation)
                # exit()
        ret_fewshots.append(fewshot)


    # print('\n'.join(ret_fewshots))
    return ret_fewshots

def generic_create_rich_fewshots() -> List[str]:

    raise NotImplementedError
