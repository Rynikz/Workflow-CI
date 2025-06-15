import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    """
    Fungsi ini dirancang untuk dijalankan dalam lingkungan CI.
    Model dilatih dan dicatat dengan autolog.
    PENTING: Di akhir, ia menyimpan Run ID ke sebuah file.
    """
    print("Memulai proses pelatihan untuk CI...")

    mlflow.sklearn.autolog()

    df = pd.read_csv("Kelayakan-pendidikan-indonesia_preprocessing/data_bersih.csv")
    X = df.drop("Status_Kelayakan", axis=1)
    y = df["Status_Kelayakan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {
        'n_estimators': 50,
        'max_depth': 10,
        'min_samples_split': 2,
        'random_state': 42
    }
    
    with mlflow.start_run() as run:
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Akurasi Model Final: {accuracy:.4f}")

        # ======================= PERUBAHAN KUNCI DI SINI =======================
        # Menyimpan Run ID yang sedang aktif ke dalam file run_id.txt
        # Ini adalah cara paling andal untuk memberitahu CI workflow apa ID-nya.
        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        
        print(f"Run ID '{run_id}' telah disimpan ke run_id.txt.")
        # ======================================================================

if __name__ == "__main__":
    main()
