import os
from typing import List, Tuple, Union
import pandas as pd
from .DataLoader import DataLoaderBase
from mylogger import Logger
import random
import numpy as np
import tqdm

onehot_exercise_info_type = {'exercise_id': 'str', 'skill_ids': 'object'}
sparse_exercise_info_type = {'exercise_id': 'str', 'skill_ids': 'object', 'skill_desc': 'object'}
moderate_exercise_info_type = {'exercise_id': 'str', 'exercise_desc': 'str', 'skill_ids': 'object', 'skill_desc': 'object'}


class LLMDataLoader(DataLoaderBase):
    def __init__(self, args, logger: Logger):
        self.data_path = args.data_path
        self.data_mode = args.data_mode
        self.logger = logger

    def load_user_data(self, data_path: str, is_shuffle: bool, train_split: float, create_train_test = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # load user_logs from recordings.csv, each line is a user's exercise record with the format above
        
        recordings_path = os.path.join(data_path, "recordings.jsonl")
        recordings_df = pd.read_json(recordings_path, lines=True)

        # check if the data is already splitted
        if os.path.exists(os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")) and os.path.exists(os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")):
            self.logger.write(f"Train and test data already exist, loading from file")
            create_train_test = False

        if not create_train_test:
            # read from existing train and test data
            train_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")
            test_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")
            train_data = pd.read_json(train_data_path, lines=True)
            test_data = pd.read_json(test_data_path, lines=True)
            train_data = train_data.astype({"student_id": str})
            test_data = test_data.astype({"student_id": str})

            return train_data, test_data

        else:  
            # create train and test data from recordings.jsonl
            self.logger.write(f"Creating train and test data from recordings.jsonl")
            # initialize the train and test data with recodings_df.columns
            train_data = pd.DataFrame(columns=recordings_df.columns)
            test_data = pd.DataFrame(columns=recordings_df.columns)
            for i, row in tqdm.tqdm(recordings_df.iterrows()):
                student_id = str(row["student_id"])
                exercises_logs = np.array(row["exercises_logs"])
                is_corrects = np.array(row["is_corrects"])
                # check if the student has at least two exercise logs to do train/test split
                if len(exercises_logs) < 2:
                    continue
                # randomly split the exercise logs into train and test set
                # is_shuffle means whether to shuffle the data before split
                if is_shuffle:
                    # create the idx of logs to randomly sample from the student's exercise logs
                    idx = list(range(len(exercises_logs)))
                    train_sample_size = int(len(idx)*train_split) if len(idx)*train_split > 1 else 1
                    train_idx = random.sample(idx, train_sample_size)
                    test_idx = [i for i in idx if i not in train_idx]
                else:
                    train_sample_size = int(len(exercises_logs)*train_split) if len(exercises_logs)*train_split > 1 else 1
                    train_idx = list(range(train_sample_size))
                    test_idx = list(range(train_sample_size, len(exercises_logs)))
                # create the train and test data for the student
                train_data_row = {"student_id": student_id, "exercises_logs": exercises_logs[train_idx], "is_corrects": is_corrects[train_idx]}
                test_data_row = {"student_id": student_id, "exercises_logs": exercises_logs[test_idx], "is_corrects": is_corrects[test_idx]}
                train_data = train_data._append(train_data_row, ignore_index=True)
                test_data = test_data._append(test_data_row, ignore_index=True)
                # save the train and test data to file
                train_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")
                test_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")
                train_data.to_json(train_data_path, lines=True, orient='records')
                test_data.to_json(test_data_path, lines=True, orient='records')

        self.logger.write(f"Split Total {len(recordings_df)} recordings, ratio {train_split}, shuffle {is_shuffle}, {len(train_data)} train data, {len(test_data)} test data")
        return train_data, test_data

    def load_onehot_data(self, data_path: str) -> dict:
        # onehot: only question id, skill id
        oh_e_info_path = os.path.join(data_path, "onehot_exercise_info.jsonl")
        oh_e_info_df = pd.read_json(oh_e_info_path, lines=True, dtype=onehot_exercise_info_type)
        # format: exercise_id, list of skill_ids in exercise_id
        self.logger.write("Loaded onehot_exercise_info data")
        return {"exercise_info": oh_e_info_df}

    def load_sparse_data(self, data_path: str) -> dict:
        # spase: skills in each questions
        sp_e_info_path = os.path.join(data_path, "sparse_exercise_info.jsonl")
        sp_e_info_df = pd.read_json(sp_e_info_path, lines=True, dtype=sparse_exercise_info_type)
        # format: exercise_id, list of skill_ids in exercise_id, list of skill_desc of skill_ids
        self.logger.write("Loaded sparse_exercise_info data")
        return {"exercise_info": sp_e_info_df}

    def load_moderate_data(self, data_path: str) -> dict:
        # moderate: question contents and skills
        mo_e_info_path = os.path.join(data_path, "moderate_exercise_info.jsonl")
        mo_e_info_df = pd.read_json(mo_e_info_path, lines=True, dtype=moderate_exercise_info_type)
        # format: exercise_id, exercise_desc, list of skill_ids in exercise_id, list of skill_desc of skill_ids
        self.logger.write("Loaded moderate_exercise_info data")
        return {"exercise_info": mo_e_info_df}

    def load_rich_data(self) -> dict:
        # rich: question contents, skills, and others like user profile, user behavior, etc.
        raise NotImplementedError
