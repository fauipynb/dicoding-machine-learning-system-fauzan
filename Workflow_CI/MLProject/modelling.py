import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os
import time
from mlflow.models.signature import infer_signature

def train_model():
    # 1. Load Data
    X_train = pd.read_csv("mobile_price_preprocessing/X_train.csv")
    y_train = pd.read_csv("mobile_price_preprocessing/y_train.csv")
    X_test = pd.read_csv("mobile_price_preprocessing/X_test.csv")
    y_test = pd.read_csv("mobile_price_preprocessing/y_test.csv")

    # 2. MLflow Tracking Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mobile_Price_Classification")

    # Matikan autolog model agar tidak konflik dengan log manual di 3.x
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run() as run:
        # 3. Modelling
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        # 4. Metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        # 5. LOG MODEL (MLflow 3.x Syntax)
        # Signature wajib ada agar 'model/' artifact dianggap valid oleh UI
        signature = infer_signature(X_train, model.predict(X_train))
        
        print("Logging model artifact to MLflow...")
        # MLflow 3.x merekomendasikan penggunaan positional argument 
        # atau 'artifact_path' yang dikombinasikan dengan 'input_example'
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Ini akan membuat folder 'model/' di Artifacts
            signature=signature,
            input_example=X_train.iloc[:1]
        )

        # 6. Log Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig("training_confusion_matrix.png")
        mlflow.log_artifact("training_confusion_matrix.png")
        
        # 7. Sinkronisasi Asynchronous (PENTING untuk Windows & 3.x)
        print("Finalizing MLflow background tasks...")
        time.sleep(5) # Memberi waktu bagi background thread untuk menulis model.pkl

    # Pastikan run benar-benar tertutup dan file meta.yaml terupdate
    mlflow.end_run()
    print(f"Training Complete. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
