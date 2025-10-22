
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    df = pd.read_csv("Latest_Data_Science_Salaries.csv")

    # Dummy example: replace with your target column
    if 'salary_in_usd' not in df.columns:
        raise ValueError("Dataset missing 'salary_in_usd' column")

    # Replace this with real feature selection logic
    X = df.select_dtypes(include=['number']).drop(columns=['salary_in_usd'])
    y = df['salary_in_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, "ml_bot/app/model.joblib")
    return acc
