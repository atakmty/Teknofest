import os
import subprocess
import glob

def run_pipeline():
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    models_dir = "models"
    script_path = os.path.join("src", "pipeline", "preprocess.py")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    import sys
    
    # 1. Önce tüm TRAIN dosyalarını bul ve FIT işlemini (öğrenme) yap
    train_files = glob.glob(os.path.join(raw_dir, "*_train.csv"))
    
    for train_file in train_files:
        filename = os.path.basename(train_file)
        dataset_name = filename.replace("_train.csv", "")
        output_file = os.path.join(processed_dir, f"processed_{filename}")
        model_path = os.path.join(models_dir, f"{dataset_name}_extractor.pkl")
        
        print(f"[{dataset_name}] Egitim verisi isleniyor ve kurallar ogreniliyor...")
        subprocess.run([
            sys.executable, script_path, 
            "--input", train_file, 
            "--output", output_file, 
            "--mode", "train", 
            "--model_path", model_path
        ], check=True)

    # 2. Sonra tüm TEST dosyalarını bul ve TRANSFORM (uygulama) yap
    test_files = glob.glob(os.path.join(raw_dir, "*_test.csv"))
    
    for test_file in test_files:
        filename = os.path.basename(test_file)
        dataset_name = filename.replace("_test.csv", "")
        output_file = os.path.join(processed_dir, f"processed_{filename}")
        model_path = os.path.join(models_dir, f"{dataset_name}_extractor.pkl")
        
        print(f"[{dataset_name}] Test verisi isleniyor (Sizinti korumali)...")
        subprocess.run([
            sys.executable, script_path, 
            "--input", test_file, 
            "--output", output_file, 
            "--mode", "test", 
            "--model_path", model_path
        ], check=True)

    print("\n✅ Islem Tamamlandi! Temizlenmis veriler 'data/processed' altinda.")

if __name__ == "__main__":
    run_pipeline()
