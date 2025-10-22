import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(data_file):
    df = pd.read_csv(data_file.name)
    X = df.select_dtypes(include=['number']).drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, "model.joblib")
    return f"Model trained successfully with accuracy: {acc:.2%}"

def predict_model(feature_json):
    model = joblib.load("model.joblib")
    df = pd.DataFrame([feature_json])
    prediction = model.predict(df)
    return f"Predicted salary: {prediction[0]:,.2f}"

train_interface = gr.Interface(
    fn=train_model,
    inputs=gr.File(label="Upload your CSV dataset"),
    outputs="text",
    title="🚀 Train Model"
)

predict_interface = gr.Interface(
    fn=predict_model,
    inputs=gr.JSON(label="Input features as JSON"),
    outputs="text",
    title="📈 Predict Salary"
)

demo = gr.TabbedInterface([train_interface, predict_interface], ["Train", "Predict"])

if __name__ == "__main__":
    demo.launch()
