from .Evaluator import EvaluatorBase
from mylogger import Logger
from LLM_factory.model import LLMModelBase
from utils import check_response_format
import random


class LLMEvaluator(EvaluatorBase):
    def __init__(self, eval_strategy: str, logger: Logger, llm: LLMModelBase):
        self.eval_strategy = eval_strategy
        self.logger = logger
        self.llm = llm

    def evaluate(self, student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode):
        eval_result = {'student_id': student_info['student_id'], 
                'pre_exe_id': test_exercise_info['exercise_id']
                }
        if eval_strategy =='simple':
            # generate fewshots and prompts to predict
            user_data = self.llm.create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode)
            prediction_response = self.llm.generate_prediction(fewshots, user_data, prompts)
            try:
                eval_result['prediction'] = check_response_format(prediction_response)
            except ValueError as e:
                # if the repsonse format is not correct, generate again, if again, then randomly choose 0 or 1
                response_tmp = self.llm.generate_prediction(fewshots, user_data, prompts)
                try:
                    eval_result['prediction'] = check_response_format(response_tmp)
                except ValueError as e:
                    self.logger.write(f"Error in prediction response format: at student {student_info['student_id']}, exercise {test_exercise_info['exercise_id']}")
                    self.logger.write(f"response: {prediction_response}")
                    eval_result['prediction'] = '0' if random.random() < 0.5 else '1'
            # generate explaination
            # delete previous "<Exercise to Predict>"
            # add new prediction to fewshots
            user_data = user_data.replace("<Exercise to Predict>", "")
            user_data += ("is_correct: " + eval_result['prediction'] + "\n")
            fewshots.append(user_data)
            eval_result['explaination'] = self.llm.generate_explaination(fewshots, eval_result['prediction'], prompts)

        elif eval_strategy == 'analysis':
            raise NotImplementedError
        elif eval_strategy =='self_correct':
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid eval_strategy: {eval_strategy}")
        
        # self.logger.write(f"Evaluation result: {eval_result}")
        return eval_result