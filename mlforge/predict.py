import pickle;
import pandas as pd;
def predict(model_path,preprocessor_path,input_data, encoder_path=None):
        print("Loading the pickled model and preprocessor...")
        model = pickle.load(open(model_path, 'rb'))
        preprocessor = pickle.load(open(preprocessor_path, 'rb'))
        encoder = pickle.load(open(encoder_path, 'rb')) if encoder_path else None
        df= pd.read_csv(input_data)
        X = preprocessor.transform(df)
        predictions = model.predict(X)
        if encoder_path:
            predictions = encoder.inverse_transform(predictions)
        return {"prediction": predictions.tolist()}
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--preprocessor_path", required=True)
    parser.add_argument("--encoder_path", required=False)
    args = parser.parse_args()
    print(predict(args.model_path, args.preprocessor_path, args.input_data, args.encoder_path))

