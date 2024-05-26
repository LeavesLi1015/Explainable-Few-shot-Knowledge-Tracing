import os
import time


class Logger:
    def __init__(self, args):
        self.log_path = args.log_path
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_path = os.path.join(args.log_path, args.model_name, args.data_mode, args.dataset_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = os.path.join(args.log_path, args.model_name, args.data_mode, args.dataset_name)
        file_name = f"{timestamp}_{args.model_type}_{args.model_name}_fsn{args.fewshot_num}_fss{args.fewshot_strategy}_es{args.eval_strategy}.txt"
        log_file = os.path.join(log_file, file_name)
        self.log_file = log_file
        self.write(f"Log created at {timestamp}")
        self.write(f"Args: {args}")

    def write(self, log_message: str, print_log=True):
        with open(self.log_file, "a", encoding="utf-8") as f:
            # timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # log_line = f"[{timestamp}] {log_message}\n"
            log_line = f"{log_message}\n"
            f.write(log_line)
            if print_log:
                print("To log:", log_line)
            f.flush()

    def flush(self):
        """Flush the log file to disk"""
        open(self.log_file, "a").close()

    def write_dict(self, log_dict: dict, print_log=True):
        for key, value in log_dict.items():
            self.write(f"{key}: {value}", print_log=print_log)