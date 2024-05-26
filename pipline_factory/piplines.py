import pandas as pd

class Pipeline:
    def train(self, train_data):
        raise NotImplementedError

    def evaluate(self, test_data):
        raise NotImplementedError

    def run(self, dataset_names):
        raise NotImplementedError