import pickle
from unittest import result;
import pandas as pd;
import os;
def predict(model_path,preprocessor_path,input_data, encoder_path=None,predicted_data=True):
        """
Load a trained model and generate predictions on input data.

Args:
    model_path (str):  
        File path to the serialized trained model (.pkl).
    preprocessor_path (str):  
        File path to the serialized preprocessor used for feature transformation.
    input_data (str):  
        File path to the input CSV containing data to predict on.
    encoder_path (Optional[str], optional):  
        File path to the serialized encoder for target label decoding 
        (used if the target was encoded). Defaults to None.
    predicted_data (bool, optional):  
            If True, saves the input data with prediction column. Defaults to True.

Returns:
    List[Any]:  
        A list of model predictions for the provided input data.

Raises:
    FileNotFoundError: If any of the provided file paths do not exist.
    ValueError: If input data is empty or improperly formatted.
    Exception: For errors during preprocessing or prediction.

Example:
    >>> predict("model.pkl", "preprocessor.pkl", "input.csv")
    [1, 0, 1]
    """

        print("Loading the pickled model and preprocessor...")
        data = pickle.load(open(model_path, 'rb'))
        preprocessor = pickle.load(open(preprocessor_path, 'rb'))
        encoder = pickle.load(open(encoder_path, 'rb')) if encoder_path else None
        df= pd.read_csv(input_data)
        X = preprocessor.transform(df)
        predictions = data["model"].predict(X)
        if encoder_path:
            predictions = encoder.inverse_transform(predictions)
        if predicted_data:
            df[data["dependent_feature"]]=predictions
            df.to_csv(os.path.join(os.path.dirname(model_path),"predicted_data.csv"),index=False)         
        return {"prediction": predictions.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,help="Path to the model file")
    parser.add_argument("--input_data", required=True,help="Path to the input data CSV file")
    parser.add_argument("--preprocessor_path", required=True,help="Path to the preprocessor file")
    parser.add_argument("--encoder_path", required=False,help="Path to the encoder file")
    parser.add_argument("--predicted_data", action="store_true",default=True,help="If set, saves input data with predictions to a new CSV file")  
    parser.add_argument(
        "--no-predicted_data", 
        action="store_false", 
        dest="predicted_data",
        help="Disable saving predicted data"
    )  
    args = parser.parse_args()
    print(predict(args.model_path, args.preprocessor_path, args.input_data, args.encoder_path,args.predicted_data))

