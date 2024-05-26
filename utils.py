import random


def check_response_format(response: str) -> str:
    # response if the response is '1' or '0'
    if response == '1' or response == '0':
        return response
    else:
        raise ValueError(f"Invalid response format: {response}, should be '1' or '0'")

def aggregate_data(id, extra_data, data_type):
    # data_type is exercise or student
    agg_data = {}
    if data_type == "exercise":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['exercise_id'] == id][col].values[0]
    elif data_type == "student":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['student_id'] == id][col].values[0]
    else:
        raise ValueError(f"Invalid data_type: {data_type}, when aggregating data")
    return agg_data


def sample_fs_id(sample_list, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy) -> list:
    # return a list of tuples of (idex in sample_list, exercise_id)
    fs_ex_ids = []
    if fewshots_num == 0:
        return fs_ex_ids
    # fewshots_strategy: "random", "first", "last", "BLEU", "RAG"
    if fewshots_strategy == "random":
        # randomly sample exercise_id from student_info to create fewshots
        # sample_list is a numpy array of str
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in random.sample(range(len(sample_list)), fewshots_num)]
    elif fewshots_strategy == "first":
        # use the first fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(fewshots_num)]
    elif fewshots_strategy == "last":
        # use the last fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list)-fewshots_num, len(sample_list))]
    elif fewshots_strategy == "RAG":
        # TODO: implement Strategy
        raise NotImplementedError
    elif fewshots_strategy == "BLEU":
        # TODO: implement Strategy
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid fewshots_strategy: {fewshots_strategy}")
    return fs_ex_ids


def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)