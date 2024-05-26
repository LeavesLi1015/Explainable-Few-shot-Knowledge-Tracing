from .piplines import Pipeline
from mylogger import Logger
from LLM_factory.GLM import GLMChat, GLM4, GLM3, GLM3_6B
from LLM_factory.GPT import GPTChat, GPT4, GPT35
from LLM_factory.model import LLMModelBase
from evaluator_factory import LLMEvaluator
from utils import aggregate_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime

'''
For student in train_data:
    Get all the information of a student and exercises
    For each given exercise to predict in test_data:
        Select proper few-shots
        LLM creates analysis of student knowledge based on few shots
        LLM predicts student performance
        LLM gives explainations of prediction
        Collect evaluation results
    return student's evaluation results
'''

class LLMPipeline(Pipeline):
    def __init__(self, 
                 model_name: str,
                 train_data,
                 test_data,
                 extra_datas,
                 logger: Logger,
                 data_mode: str,
                 fewshots_num: int,
                 fewshots_strategy: str,
                 eval_strategy: str,
                 test_num: int,
                 random_seed: int
                 ):
        self.model_name = model_name
        self.train_data = train_data
        self.test_data = test_data
        self.extra_datas = extra_datas # list of pd
        self.logger = logger
        self.data_mode = data_mode
        self.llm =self.init_llm(model_name)
        self.fewshots_num = fewshots_num
        self.fewshots_strategy = fewshots_strategy
        self.eval_strategy = eval_strategy
        self.evaluator = LLMEvaluator(eval_strategy=eval_strategy, logger=logger, llm=self.llm)
        self.test_num = test_num
        self.random_seed = random_seed


    def init_llm(self, model_name):
        # initialize llm
        if model_name.startswith('glm'):
            if model_name == 'glm-4':
                llm = GLM4()
            elif model_name == 'glm-3-turbo':
                llm = GLM3()
            elif model_name == 'glm-3-6b':
                llm = GLM3_6B()
            else:
                raise ValueError(f"Invalid glm model name: {model_name}")
        elif model_name.startswith('gpt'):
            if model_name == 'gpt-4' or model_name == 'gpt-4-1106-preview' or model_name == 'gpt-4-32k':
                llm = GPT4()
            elif model_name == 'gpt-3.5-turbo':
                llm = GPT35()
            else:
                raise ValueError(f"Invalid gpt model name: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return llm


    def evaluate(self, train_data, test_data, data_mode, extra_datas, fewshots_num, fewshots_strategy, eval_strategy):
        # train_data has n_student lines, each line has one student's logs, extra_datas is a list of pd, each pd has side information of exercises
        # test_data has n_test_student lines, each line has one student's logs

        # initialize eval results
        # key: student_id, value: dict of eval results for one student
        eval_results = {}
        # randomly select number of lines in train_data to evaluate, train_data is a pd
        # print(type(train_data))
        if self.test_num != -1:
            train_data = train_data.sample(n=self.test_num, random_state=self.random_seed)
        else:
            self.logger.write(f"Evaluate all {len(train_data)} students")
        
        # print(type(train_data))
        total_y_pre = []
        total_y_true = []

        # start evaluation, each iteration returns a dict of eval results for one student
        for i, student_info in train_data.iterrows():
            self.logger.write(f"----------------Evaluating student {student_info['student_id']}")
            test_exercises = test_data[test_data['student_id'] == student_info['student_id']]['exercises_logs'].values[0]
            test_corrects = test_data[test_data['student_id'] == student_info['student_id']]['is_corrects'].values[0]
            # key: exercise_id, value: dict of eval results for one exercise
            stu_eval_results = {}
            # extra_exercicse_info = extra_datas['exercise_info']
            flag = True
            # test_exercises is a nump array of exercise_ids
            for i, test_exercise_id in enumerate(test_exercises):
                self.logger.write(f"*****Evaluating student {student_info['student_id']} on exercise {test_exercise_id}")
                is_correct = test_corrects[i]
                # get exercise_info
                test_exercise_info = {'exercise_id': test_exercise_id, 'is_correct': is_correct}
                test_exercise_info.update(aggregate_data(test_exercise_id, extra_datas['exercise_info'], 'exercise'))
                
                prompts = self.llm.get_prompts(data_mode=data_mode)
                if fewshots_strategy == 'first' or fewshots_strategy == 'last':
                    if flag:
                        fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts)
                        flag = False
                else:
                    fewshots = self.llm.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts)
                    
                self.logger.write(f"Fewshots:\n{fewshots}")
                exe_eval_results = self.evaluator.evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode)
                exe_eval_results.update({'is_correct': is_correct})
                stu_eval_results[test_exercise_id] = exe_eval_results
                # break # for debug only, only evaluate one exercise
                # exit()
            # collect each student's eval results
            # ...
            y_pre = []
            y_true = []
            for k, v in stu_eval_results.items(): # k is exercise_id, v is dict of eval results for one exercise
                y_pre.append(int(v['prediction']))
                y_true.append(int(v['is_correct']))
            stu_acc = accuracy_score(y_true, y_pre)
            stu_precision = precision_score(y_true, y_pre)
            stu_recall = recall_score(y_true, y_pre)
            stu_f1 = f1_score(y_true, y_pre)
            stu_test_count = len(y_true)
            self.logger.write(f"y_true: {y_true}, y_pre: {y_pre}")
            self.logger.write(f"Student {student_info['student_id']}, len: {stu_test_count}, acc: {stu_acc}, precision: {stu_precision}, recall: {stu_recall}, f1: {stu_f1}")
            stu_result = {'student_id': student_info['student_id'], 'accuracy': stu_acc, 'eval_results': stu_eval_results}
            eval_results[student_info['student_id']] = stu_result
            total_y_pre.extend(y_pre)
            total_y_true.extend(y_true)
            # break # for debug only, only evaluate one student

        # add final eval results to eval_results, return eval_results
        final_acc = accuracy_score(total_y_true, total_y_pre)
        fianl_precision = precision_score(total_y_true, total_y_pre)
        fianl_recall = recall_score(total_y_true, total_y_pre)
        fianl_f1 = f1_score(total_y_true, total_y_pre)
        self.logger.write(f"Total test student: {len(train_data)}, total test count: {len(total_y_true)}")
        self.logger.write(f"Final accuracy: {final_acc}, precision: {fianl_precision}, recall: {fianl_recall}, f1: {fianl_f1}")
        return eval_results
    

    def run(self):
        # LLMs only need to evaluate
        eval_results = self.evaluate(self.train_data, 
                                     self.test_data, 
                                     self.data_mode, 
                                     self.extra_datas, 
                                     self.fewshots_num, 
                                     self.fewshots_strategy, 
                                     self.eval_strategy
                                     )
        self.display_results(eval_results, self.logger)
    
    def display_results(self, eval_results, logger):
        # display eval results and save to loggers
        # write end time to logger
        logger.write(f"End time: {datetime.datetime.now()}")
        logger.write(f"Eval results:\n{eval_results}")
    