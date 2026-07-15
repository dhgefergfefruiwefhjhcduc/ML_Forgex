import pickle
import pandas as pd
import os
import time
import numpy as np
import gensim
from nltk.tokenize import word_tokenize
from mlforgex.logger import logger


def predict(model_path, input_data, predicted_data=True, nlp=False):
    """
    Load a trained model and generate predictions on input data.

    Args:
        model_path (str):
            File path to the serialized trained model (.pkl).
        input_data (str):
            File path to the input CSV containing data to predict on.
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
        FileNotFoundError: If any of the provided file paths do not exist. All file preprocessor,encoder,metadata if exist should be in the artifacts folder.
        ValueError: If input data is empty or improperly formatted.
        Exception: For errors during preprocessing or prediction.

    Example:
        >>> predict("model.pkl", "preprocessor.pkl", "input.csv")
        [1, 0, 1]
    """

    _start_time = time.time()
    logger.info("Loading the pickled model and preprocessor...")
    
    # 1. Establish directory and path variables first
    model_dir = os.path.dirname(model_path)
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    encoder_file_path = os.path.join(model_dir, "encoder.pkl")
    if not nlp:
        preprocessor_file_path = os.path.join(model_dir, "preprocessor.pkl") # String path, not a boolean
    else:
        preprocessor_file_path = os.path.join(model_dir, "word2vec.model") # String path, not a boolean
    # 2. Check file existence
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
    model = pickle.load(open(model_path, "rb"))
    metadata = pickle.load(open(metadata_path, "rb"))
    
    encoder_exists = os.path.exists(encoder_file_path)
    encoder = pickle.load(open(encoder_file_path, "rb")) if encoder_exists else None
    
    df = pd.read_csv(input_data)
    if df.empty:
        raise ValueError("Input data CSV is empty.")
    logger.info(f"Input data loaded successfully. Shape: {df.shape}")

    if not nlp:
        # Standard Tabular Preprocessing
        df.drop(columns=metadata["drop_col"], inplace=True)
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
                
        if os.path.exists(preprocessor_file_path):
            preprocessor = pickle.load(open(preprocessor_file_path, "rb"))
            X = preprocessor.transform(df)
        else:
            X = df # Fallback if no preprocessor transformer exists
            
    else:
        # NLP Preprocessing
        text_col = [
            i for i in df.columns 
            if df[i].dtype == "object" and i != metadata.get("dependent_feature")
        ]
        df["new_text"] = df[text_col].astype(str).agg(" ".join, axis=1)
        
        from mlforgex.cleaning import avg_wordtovec, preprocess
        df["new_text"] = df["new_text"].apply(preprocess)
        word_token = [word_tokenize(i) for i in df["new_text"]]
        
        # Safe-load Word2Vec model using the validated path string
        if not os.path.exists(preprocessor_file_path):
            raise FileNotFoundError(f"Word2Vec model file missing at {preprocessor_file_path}")
            
        mod = gensim.models.Word2Vec.load(preprocessor_file_path)
        vector_text = [avg_wordtovec(i, mod) for i in word_token]
        X = vector_text

    _prep_time = time.time()
    logger.info(f"Preprocessing completed in {_prep_time - _start_time:.2f}s")
    # 3. Predict and Output
    predictions = model.predict(X)
    
    if encoder_exists:
        predictions = encoder.inverse_transform(predictions)
        
    if predicted_data:
        df[metadata["dependent_feature"]] = predictions
        if nlp:
            df.drop(columns=["new_text"], inplace=True)
        df.to_csv(
            os.path.join(model_dir, f"{metadata.get('model_name', 'model')}_predicted_data.csv"), 
            index=False
        )
        
    _end_time = time.time()
    logger.info(f"Prediction pipeline completed successfully in {_end_time - _start_time:.2f}s")
    return {"prediction": predictions.tolist()}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the model file")
    parser.add_argument(
        "--input_data", required=True, help="Path to the input data CSV file"
    )
    parser.add_argument(
        "--no-predicted_data",
        action="store_false",
        dest="predicted_data",
        default=True,
        help="Disable saving input data with predictions to a CSV file (saved by default).",
    )
    parser.add_argument(
        "--nlp", action="store_true", default=False, help="Enable NLP/text-mode"
    )
    args = parser.parse_args()
    predict(args.model_path, args.input_data, args.predicted_data, nlp=args.nlp)
    _end_time = time.time()
    logger.info("Prediction completed and saved (if enabled).")


if __name__ == "__main__":
    predict(input_data="test.csv",model_path="cardekho_artifacts/kneighborsregressor.pkl",nlp=False)    
    # main()
