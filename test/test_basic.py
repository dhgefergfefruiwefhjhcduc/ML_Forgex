from mlforge import train_model, predict

def test_train_model():
    result = train_model("mlforge/diabetes_cleaned.csv", "Outcome", rmse_prob=0.3, f1_prob=0.7, n_jobs=-1)
    assert result["status"] == "success"

def test_predict():
    result =predict("mlforge/artifacts/model.pkl", "mlforge/artifacts/preprocessor.pkl", "mlforge/input.csv", "mlforge/artifacts/encoder.pkl")
    assert "prediction" in result
