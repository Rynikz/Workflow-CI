import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import argparse
import os # Library untuk berinteraksi dengan sistem operasi

def main(data_path):
    """
    Fungsi ini sekarang menyimpan run_id.txt menggunakan path absolut
    untuk memastikan file selalu ditemukan oleh workflow CI.
    """
    print("Memulai proses pelatihan untuk CI...")

    mlflow.sklearn.autolog()

    try:
        df = pd.read_csv(data_path)
        print(f"Dataset berhasil dimuat dari: {data_path}")
    except FileNotFoundError:
        print(f"Error: Dataset tidak ditemukan di path yang diberikan: {data_path}")
        return

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
        run_id = run.info.run_id
        
        # Menggunakan variabel lingkungan GITHUB_WORKSPACE untuk mendapatkan path absolut
        # dari direktori utama proyek di runner GitHub.
        # Ini memastikan file run_id.txt disimpan di lokasi yang benar dan dapat diprediksi.
        workspace = os.getenv("GITHUB_WORKSPACE", ".")
        output_path = os.path.join(workspace, "run_id.txt")
        
        with open(output_path, "w") as f:
            f.write(run_id)
        
        print(f"Run ID '{run_id}' telah disimpan ke path absolut: {output_path}.")
        # ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="Path ke file CSV dataset bersih")
    args = parser.parse_args()
    
    main(args.data_path)
