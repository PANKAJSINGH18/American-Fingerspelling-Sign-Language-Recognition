import pandas as pd
import numpy as np
import json
import tflite_runtime.interpreter as tflite


class Model:

    def __init__(self) -> None:
        self.REQUIRED_SIGNATURE = "serving_default"
        self.REQUIRED_OUTPUT = "outputs"
        self.interpreter = tflite.Interpreter('datamount/model.tflite')
        with open("datamount/character_to_prediction_index.json", "r") as f:
            self.character_map = json.load(f)
        self.rev_character_map = {j: i for i, j in self.character_map.items()}
        self.found_signatures = list(self.interpreter.get_signature_list().keys())
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        self.inference = pd.read_json('datamount/inference_args.json')

    def Inference(self, df):
        
        output = self.prediction_fn(inputs=df[self.inference['selected_columns']])
        prediction_str = "".join([self.rev_character_map.get(s, "") for s in np.argmax(output[self.REQUIRED_OUTPUT], axis=1)])
        return prediction_str
    

if __name__=='__main__':
    _model = Model()

    df = pd.read_csv('landmark.csv')
    df = df.astype('float32')
    result = _model.Inference(df)
    print(result)