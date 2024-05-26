from .piplines import Pipeline

class KTMPipeline(Pipeline):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def train(self, train_data):

        raise NotImplementedError

    def evaluate(self, test_data):

        raise NotImplementedError

    def run(self, dataset_names):
        
        raise NotImplementedError