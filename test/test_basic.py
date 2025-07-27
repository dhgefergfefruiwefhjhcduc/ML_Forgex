from mlforge import train_model, predict

def test_train_model():
    result = train_model("StudentsPerformance.csv", "math_score", rmse_prob=0.3, f1_prob=0.7, n_jobs=-1)
    assert result["status"] == "success"

def test_predict():
    result =predict("artifacts/model.pkl", "artifacts/preprocessor.pkl", "input.csv")
    assert "prediction" in result
