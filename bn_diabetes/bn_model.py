import pandas as pd
from pathlib import Path
import pickle


class BNModel:

    def __init__(self,
                 as_new: bool = False,
                 model_path: Path = Path('models/bn1_bic_v2.pickle')):
        self.as_new = as_new
        self.model_path = model_path
        self.model = None

    def load_model(self):
        with open(self.model_path, 'rb') as _file:
            self.model = pickle.load(_file)

    def inference(self, evidence: dict) -> pd.DataFrame:
        return self.model.sample(1, evidence=evidence)
