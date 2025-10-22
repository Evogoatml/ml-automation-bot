
import joblib
import pandas as pd

def predict(input_data: dict):
    model = joblib.load("ml_bot/app/model.joblib")
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction.tolist()
