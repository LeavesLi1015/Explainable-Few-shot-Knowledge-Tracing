from .DataLoader import DataLoaderBase

class KTDataLoader(DataLoaderBase):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_user_data(self):
        raise NotImplementedError
    
    def load_extra_data(self):
        raise NotImplementedError