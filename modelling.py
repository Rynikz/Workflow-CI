import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    """
    Fungsi ini dirancang untuk dijalankan dalam lingkungan CI (Continuous Integration).
    Model dilatih menggunakan parameter yang sudah ditentukan dan dicatat dengan autolog.
    """
    print("Memulai proses pelatihan untuk CI...")
 
    mlflow.sklearn.autolog()

    try:
        df = pd.read_csv("Kelayakan-pendidikan-indonesia_preprocessing/data_bersih.csv")
        print("Dataset berhasil dimuat.")
    except FileNotFoundError:
        print("Error: Dataset 'data_bersih.csv' tidak ditemukan.")
        return

    # Memisahkan fitur (X) dan target (y)
    X = df.drop("Status_Kelayakan", axis=1)
    y = df["Status_Kelayakan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menggunakan parameter terbaik yang ditemukan pada Kriteria 2
    # Ganti nilai ini jika Anda menemukan parameter yang lebih baik
    best_params = {
        'n_estimators': 50,
        'max_depth': 10,
        'min_samples_split': 2,
        'random_state': 42
    }
    
    print(f"Menggunakan parameter: {best_params}")

    # Memulai run MLflow. Autolog akan mencatat semua metrik dan artefak.
    with mlflow.start_run() as run:
        print(f"Memulai run dengan ID: {run.info.run_id}")
        
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Akurasi Model Final: {accuracy:.4f}")
        print("Pelatihan selesai. Model dan metrik tercatat oleh autolog di folder 'mlruns'.")

if __name__ == "__main__":
    main()
