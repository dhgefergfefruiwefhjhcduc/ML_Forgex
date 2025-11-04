import pickle
import pandas as pd
import os
import numpy as np
import gensim
from nltk.tokenize import word_tokenize
def predict(model_path,preprocessor_path,input_data, encoder_path=None,predicted_data=True,nlp=False):
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
    nlp (bool, optional):
        If True, process input as text: combine all object-dtype text columns
        (excluding the target) into a single field, apply the same text preprocessing
        used during training (preprocess), load the Word2Vec model from
        `preprocessor_path` (expected to point to a saved Word2Vec model), convert
        each document to an average word-vector using `avg_wordtovec`, and pass the
        resulting vectors to the model for prediction. Defaults to False.

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
        encoder = pickle.load(open(encoder_path, 'rb')) if encoder_path else None
        df= pd.read_csv(input_data)
        if not nlp:
            df.drop(columns=data["drop_col"],inplace=True)
            df.replace(["", "NA", "na", "N/A", "n/a", "?", "--", "-"], np.nan, inplace=True)
            for col in df.columns:
                if df[col].dtype == "object" or df[col].dtype.name == "category":
                    mode_vals = df[col].mode(dropna=True)
                    if not mode_vals.empty:
                        df[col] = df[col].fillna(mode_vals.iloc[0]) 
                    else:
                        df[col] = df[col].fillna("")  
                else:
                    med = df[col].median()
                    if np.isnan(med):
                        med = 0
                    df[col] = df[col].fillna(med) 
            preprocessor = pickle.load(open(preprocessor_path, 'rb'))
            X = preprocessor.transform(df)
        else:
            text_col=[i for i in df.columns if df[i].dtype=="object" and i!=data["dependent_feature"]]
            df["new_text"] = df[text_col].astype(str).agg(" ".join, axis=1)
            from mlforge.cleaning import avg_wordtovec,preprocess
            df["new_text"] = df["new_text"].apply(preprocess)
            word_token=[word_tokenize(i) for i in df["new_text"]]
            mod = gensim.models.Word2Vec.load(os.path.join(preprocessor_path))
            vector_text=[avg_wordtovec(i,mod) for i in word_token]
            X = vector_text
        predictions = data["model"].predict(X)
        if encoder_path:
            predictions = encoder.inverse_transform(predictions)
        if predicted_data:
            df[data["dependent_feature"]] = predictions
            if nlp:
                df.drop(columns=["new_text"],inplace=True)
            df.to_csv(os.path.join(os.path.dirname(model_path),"predicted_data.csv"),index=False)         
        return {"prediction": predictions.tolist()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,help="Path to the model file")
    parser.add_argument("--input_data", required=True,help="Path to the input data CSV file")
    parser.add_argument("--preprocessor_path", required=True,help="Path to the preprocessor file")
    parser.add_argument("--encoder_path", required=False,help="Path to the encoder file")
    parser.add_argument(
    "--no-predicted_data",
    action="store_false",
    dest="predicted_data",
    default=True,
    help="Disable saving input data with predictions to a CSV file (saved by default)."
)
    parser.add_argument("--nlp", action="store_true", default=False, help="Enable NLP/text-mode")
    args = parser.parse_args()
    predict(args.model_path, args.preprocessor_path, args.input_data, args.encoder_path,args.predicted_data,nlp=args.nlp)
    print("Prediction completed and saved (if enabled).")

