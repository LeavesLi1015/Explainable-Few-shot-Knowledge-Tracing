import os
import argparse
from pipline_factory import LLMPipeline
from mylogger import Logger
from dataloader_factory import LLMDataLoader


def get_args():
    parser = argparse.ArgumentParser()
    # LLM model name
    parser.add_argument('--model_type', type=str, default='llm', help='model type llm or ktm')
    parser.add_argument('--model_name', type=str, default='glm-4', help='model name')
    parser.add_argument('--data_path', type=str, default='./datasets', help='data path')
    # data_mode: sparse, moderate, rich
    parser.add_argument('--data_mode', type=str, default='sparse', help='data mode: onehot, sparse, moderate, rich')
    # dataset name
    parser.add_argument('--dataset_name', type=str, default='FrcSub', help='dataset name')
    parser.add_argument('--log_path', type=str, default='./logs', help='log path')
    # train_split
    parser.add_argument('--train_split', type=float, default=0.8, help='train split')
    # parser.add_argument('--is_shuffle', type=bool, default=True, help='shuffle data when splitting')
    parser.add_argument('--is_shuffle', action='store_true', help='shuffle data when splitting')
    # test number
    parser.add_argument('--test_num', type=int, default=20, help='test number')
    # random seed
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')

    # llm fewshot settings
    parser.add_argument('--fewshot_num', type=int, default=4, help='fewshot num, 0 means zero-shot')
    parser.add_argument('--fewshot_strategy', type=str, default='random', help='fewshot strategy, random/first/last/RAG')
    parser.add_argument('--eval_strategy', type=str, default='simple', help='eval strategy, simple/analysis/self_correct')

    args = parser.parse_args()
    return args


def main(args):
    # initial logger
    my_logger = Logger(args)
    # initial dataloader
    data_path = os.path.join(args.data_path, args.data_mode, args.dataset_name)
    if not os.path.exists(data_path):
        print(data_path)
        raise ValueError("data path not exist, check path, mode, or dataset name")
    if args.model_type == 'llm':
        data_loader = LLMDataLoader(args=args, logger=my_logger)
        train_data, test_data = data_loader.load_user_data(data_path=data_path, train_split=args.train_split, is_shuffle=args.is_shuffle)
        # extra_datas is a list of extra data, each one is pd.DataFrame
        if args.data_mode == 'onehot':
            extra_datas = data_loader.load_onehot_data(data_path=data_path)
        elif args.data_mode =='sparse':
            extra_datas = data_loader.load_sparse_data(data_path=data_path)
        elif args.data_mode =='moderate':
            extra_datas = data_loader.load_moderate_data(data_path=data_path)
        elif args.data_mode == 'rich':
            extra_datas = data_loader.load_rich_data()
        else:
            raise ValueError("data mode not in ['onehot','sparse','moderate', 'rich']")
    elif args.model_type == 'ktm':
        # TODO: add ktm dataloader
        pass
    else:
        raise ValueError("model type not in ['llm', 'ktm']")

    # initial pipline
    llm_pipline = LLMPipeline(model_name=args.model_name, 
                              train_data=train_data,
                              test_data=test_data, 
                              extra_datas=extra_datas, 
                              logger=my_logger, 
                              data_mode=args.data_mode,
                              fewshots_num=args.fewshot_num, 
                              fewshots_strategy=args.fewshot_strategy,
                              eval_strategy=args.eval_strategy,
                              test_num=args.test_num,
                              random_seed=args.random_seed
                              )
    llm_pipline.run()

if __name__ == "__main__":
    args = get_args()
    main(args)