import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

class VariantEnsembleArchitect:
    def __init__(self, master_model_type='xgboost'):
        """
        Master model ve ensemble (meta-model) mimarisi.
        :param master_model_type: 'xgboost' veya 'lightgbm'
        """
        self.master_model_type = master_model_type
        
        # 1. Master model (Genel varyant kurallarını öğrenir)
        # Veri setinizde 2000 Patojenik (1) / 800 Benign (0) var.
        # scale_pos_weight = Negatif / Pozitif = 800 / 2000 = 0.4
        if self.master_model_type == 'xgboost':
            self.master_model = xgb.XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=0.4
            )
        elif self.master_model_type == 'lightgbm':
            self.master_model = lgb.LGBMClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5, 
                random_state=42,
                scale_pos_weight=0.4
            )
        else:
            raise ValueError("Desteklenmeyen model tipi. 'xgboost' veya 'lightgbm' seçin.")
            
        # 2. Panel modellerini saklamak için yapı (örn. Kanser, Kardiyoloji)
        self.panel_models = {}
        
        # 3. Meta-model (Stacking için) - Basit bir Lojistik Regresyon en uygunudur 
        # çünkü sadece base modellerin olasılıklarını birleştirir ve overfit olmaz.
        self.meta_model = LogisticRegression()
        
        # Manuel ağırlıklı ortalama için statik ağırlıklar (Eğer meta-model kullanılmayacaksa)
        self.weights = {'master': 0.5, 'panel': 0.5}

    def train_master_model(self, X_train, y_train):
        """
        Master modeli ana veri setinde (2800 örnek) eğitir.
        """
        print(f"[*] Master Model ({self.master_model_type.upper()}) eğitiliyor...")
        self.master_model.fit(X_train, y_train)
        print("[+] Master Model eğitimi tamamlandı.")
        
    def add_panel_model(self, panel_name, model):
        """
        Başka bir ekibin/sürecin eğittiği spesifik panel modelini sisteme dahil eder.
        """
        self.panel_models[panel_name] = model
        print(f"[+] '{panel_name}' paneli sisteme eklendi.")

    def set_manual_weights(self, master_weight, panel_weight):
        """
        Ağırlıklı ortalama (Weighted Average) için ağırlıkları yapılandırır.
        """
        total = master_weight + panel_weight
        self.weights['master'] = master_weight / total
        self.weights['panel'] = panel_weight / total
        print(f"[*] Ağırlıklar güncellendi: Master={self.weights['master']:.2f}, Panel={self.weights['panel']:.2f}")

    def train_meta_model_cv(self, X_train, y_train, panel_name, cv_folds=5):
        """
        Stacking mekanizması (Cross-Validation ile). Meta-Model (Lojistik Regresyon) eğitilir.
        Ayrı veri (X_val, y_val) ayırmaya gerek kalmadan bütün train verisi üzerinden 
        Out-of-Fold (OOF) tahminleri alınır, böylece veri israfı olmaz.
        """
        if panel_name not in self.panel_models:
            raise ValueError(f"'{panel_name}' paneli bulunamadı.")
            
        print(f"[*] '{panel_name}' paneli ve Master Model ile OOF (Cross-Val) Stacking eğitiliyor (K={cv_folds})...")
        
        # 1. Master model OOF tahminlerini al
        # Not: cross_val_predict master modelin X_train üzerindeki başarısını yanlılık olmadan bulur
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        master_oof = cross_val_predict(
            self.master_model, X_train, y_train, 
            cv=cv_strategy, method='predict_proba'
        )[:, 1]
        
        # 2. Panel model tahminlerini al
        # (Gerçekte panel modeller genellikle daha önceden bağımsız veriyle eğitilmiş olur)
        panel_preds = self.panel_models[panel_name].predict_proba(X_train)[:, 1]
        
        # Olasılıkları birleştir
        meta_features = np.column_stack((master_oof, panel_preds))
        
        # Meta-modeli OOF özellikleri ile eğit
        self.meta_model.fit(meta_features, y_train)
        
        print(f"[+] Meta-Model Öğrenilen Katsayılar: Master Katsayısı={self.meta_model.coef_[0][0]:.3f}, Panel Katsayısı={self.meta_model.coef_[0][1]:.3f}")

    def predict_weighted_average(self, X, panel_name):
        """
        Yöntem 1: Test aşamasında Statik Ağırlıklandırma (Manuel belirlenmiş)
        """
        if panel_name not in self.panel_models:
            raise ValueError(f"'{panel_name}' paneli bulunamadı.")
            
        master_preds = self.master_model.predict_proba(X)[:, 1]
        panel_preds = self.panel_models[panel_name].predict_proba(X)[:, 1]
        
        ensemble_preds = (master_preds * self.weights['master']) + (panel_preds * self.weights['panel'])
        return ensemble_preds

    def predict_stacking(self, X, panel_name):
        """
        Yöntem 2: Test aşamasında Makine Öğrenmesi destekli Stacking tahmini
        """
        if panel_name not in self.panel_models:
            raise ValueError(f"'{panel_name}' paneli bulunamadı.")
            
        master_preds = self.master_model.predict_proba(X)[:, 1]
        panel_preds = self.panel_models[panel_name].predict_proba(X)[:, 1]
        
        meta_features = np.column_stack((master_preds, panel_preds))
        
        # Meta-Model (Lojistik Reg) nihai kararı verir
        ensemble_preds = self.meta_model.predict_proba(meta_features)[:, 1]
        return ensemble_preds

    def explain_with_shap(self, X, feature_names=None):
        """
        SHAP (SHapley Additive exPlanations) kullanarak XGBoost/LightGBM modelinin 
        hangi özelliklere ne kadar önem verdiğini hesaplar ve görselleştirir.
        """
        print("\n[*] SHAP değerleri hesaplanıyor... (Bu işlem büyük verilerde biraz sürebilir)")
        
        # Ağaç tabanlı modeller için TreeExplainer kullanılır
        explainer = shap.TreeExplainer(self.master_model)
        shap_values = explainer.shap_values(X)
        
        # SHAP Summary Plot çizimi
        plt.figure(figsize=(10, 6))
        plt.title("Master Model (XGBoost) SHAP Özellik Önem Dereceleri")
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        
        # Dosyaya kaydet
        plt.savefig("shap_summary_plot.png", dpi=300)
        print("[+] SHAP grafiği başarıyla çizildi ve 'shap_summary_plot.png' olarak dizine kaydedildi.")
        plt.close()


def generate_mock_data(n_pathogenic=2000, n_benign=800):
    """
    Şartnamedeki isimlendirme kurallarına ve Master panel sınıf 
    dengesizliğine uygun sentetik (mock) veri üretir.
    """
    n_total = n_pathogenic + n_benign
    
    # Etiketler (Doğru Sınıf: 1=Patojenik, 0=Benign)
    labels = np.array([1]*n_pathogenic + [0]*n_benign)
    
    # AL_: Frekans / Popülasyon (Sürekli değişken)
    al_cols = [f'AL_{i}' for i in range(1, 6)]
    al_data = np.random.rand(n_total, len(al_cols))
    
    # EK_: Evrimsel Korunmuşluk (Sürekli değişken)
    ek_cols = [f'EK_{i}' for i in range(1, 6)]
    ek_data = np.random.rand(n_total, len(ek_cols))
    
    # CAT_: Kategorik Meta-Veri (Kesikli değişken, örn: 0, 1, 2)
    cat_cols = [f'CAT_{i}' for i in range(1, 4)]
    cat_data = np.random.randint(0, 4, size=(n_total, len(cat_cols)))
    
    # AA: Amino Asit Değişimi. 
    # Not: Ön işleme pipeline'ı tamamlanana kadar model patlamasın diye 
    # AA string'lerini sayısal özelliklere (örn. hidrofobiklik farkı) dönüştürülmüş gibi farz ediyoruz.
    aa_cols = ['AA_Hydrophobicity_Diff', 'AA_MolecularWeight_Diff']
    aa_data = np.random.randn(n_total, len(aa_cols))
    
    # DataFrameleri birleştir
    df_al = pd.DataFrame(al_data, columns=al_cols)
    df_ek = pd.DataFrame(ek_data, columns=ek_cols)
    df_cat = pd.DataFrame(cat_data, columns=cat_cols)
    df_aa = pd.DataFrame(aa_data, columns=aa_cols)
    
    df = pd.concat([df_al, df_ek, df_cat, df_aa], axis=1)
    df['Label'] = labels
    
    # Modeli yanıltmamak için sıralı veriyi karıştır (Shuffle)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# =====================================================================
# GERÇEK VERİ İLE KULLANIM SENARYOSU
# =====================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("GERÇEK VERİ YÜKLENİYOR (processed_Master_train.csv)")
    print("="*60)
    
    import os
    csv_path = os.path.join("data", "processed", "processed_Master_train.csv")
    try:
        real_df = pd.read_csv(csv_path)
        print(f"[*] Veri başarıyla yüklendi. ({real_df.shape[0]} satır, {real_df.shape[1]} sütun)")
    except Exception as e:
        print(f"[!] Veri yüklenirken hata oluştu: {e}")
        exit()
        
    # Veri setinde tahmin edilecek (hedef) kolonunun adını otomatik tespit edelim:
    target_col = 'Label' 
    if target_col not in real_df.columns:
        possible_targets = ['Class', 'CLASS', 'Target', 'TARGET', 'Patojenik']
        for col in possible_targets:
            if col in real_df.columns:
                target_col = col
                break
                
    if target_col not in real_df.columns:
        # Eğer standart isimlerden hiçbiri yoksa son kolonu "Karar" (Hedef/Y) olarak varsay
        target_col = real_df.columns[-1]
        
    print(f"[*] Hedef sınıf (Patojenik/Benign) sütunu olarak '{target_col}' kullanılacak.")
    
    # XGBoost ve modeller metin (Yazı) değerleriyle çalışamaz. 
    # Hatayı ("Master" string to float hatasını) önlemek için tüm metin kolonlarını sayılara çeviriyoruz:
    for col in real_df.select_dtypes(include=['object', 'string']).columns:
        # Örn: Kolon içindeki "Master" yazısını 0, "Test" yazısını 1 gibi matematiksel kodlara çevirir.
        real_df[col] = pd.factorize(real_df[col])[0]
        
    # KULLANICI İSTEĞİ: "Bütün metrikleri ekle kimseyi droplama" deneyi.
    # Modelin ham halindeki o %100 "Yapay" kopyacı başarısını tam test edebilmek için 
    # sızıntı/hile koruması TAMAMEN KALDIRILDI.
    leaky_cols = []
    for leak in leaky_cols:
        if leak in real_df.columns:
            real_df = real_df.drop(leak, axis=1)
            print(f"[!] Gizli sızıntı yapan kopya sütun '{leak}' dataset'ten temizlendi.")
        
    # X (Özellikler/Features) ve y (Etiketler/Target Y) ayırma
    X = real_df.drop(target_col, axis=1).values
    y = real_df[target_col].values
    
    # Yeni indirilen verideki etiket tipleri (-1, 1 gibi) XGBoost'u bozmasın diye 
    # her ihtimale karşı Y değerlerini [0 ve 1] formatına zorunlu formatlıyoruz.
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)
    
    # SHAP ve Özellik Raporları için GERÇEK sütun isimlerini CSV'den kaydedelim
    feature_names = real_df.drop(target_col, axis=1).columns.tolist() 
    
    # Train ve Test bölünmesi (Artık Stacking için feda edeceğimiz bir Validation setine gerek yok!)
    # %80 Train, %20 Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # KULLANICI İSTEĞİ: SMOTE (Sentetik Azınlık Üretimi) işlemi geçici olarak silindi/devre dışı bırakıldı.
    # Model tekrar orijinal dengesiz haliyle (2000'e 800) eğitilecek.
    
    print(f"Train Seti: {X_train.shape[0]} | Test Seti: {X_test.shape[0]} (Train'in tamamı CV ile Meta-Model için kullanılacak)\n")
    
    # --- 1. Mimarinin Kurulması ---
    architect = VariantEnsembleArchitect(master_model_type='xgboost')
    
    # --- 2. Master Modelin Eğitimi ---
    architect.train_master_model(X_train, y_train)
    
    # --- 3. Panel Modelin Sisteme Entegre Edilmesi ---
    # Not: Gerçek senaryoda bu panel modeli, spesifik kanser genleri ve özel verilerle dışarıda eğitilmiş olacak.
    # Burada simulasyon için basit bir Random Forest eğitiyoruz.
    from sklearn.ensemble import RandomForestClassifier
    cancer_panel_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    cancer_panel_model.fit(X_train, y_train) 
    
    architect.add_panel_model('Kanser_Paneli', cancer_panel_model)
    
    print("\n" + "="*60)
    print("ENSEMBLE (BİRLEŞTİRME) STRATEJİLERİ")
    print("="*60)
    
    # --- Strateji 1: Manuel Ağırlıklı Ortalama ---
    # Örn: "Kanser varyantlarında panel modeline %70, Master modele %30 güvenelim"
    architect.set_manual_weights(master_weight=0.3, panel_weight=0.7)
    preds_weighted = architect.predict_weighted_average(X_test, 'Kanser_Paneli')
    
    # --- Strateji 2: ML-Tabanlı Stacking (CV Meta-Model Eğitimi) ---
    # Stacking, artık Validation (X_val) yerine, tüm X_train üzerinden Out-of-Fold 
    # tahminler alarak veri kaybı yaşanmadan Cross-Validation ile eğitilir (K=5).
    architect.train_meta_model_cv(X_train, y_train, 'Kanser_Paneli', cv_folds=5)
    preds_stacking = architect.predict_stacking(X_test, 'Kanser_Paneli')
    
    # --- 4. Sonuçların Kıyaslanması (Test Seti Üzerinde) ---
    print("\n" + "="*60)
    print("TEST SONUÇLARI (ROC-AUC Skorları)")
    print("="*60)
    
    master_preds = architect.master_model.predict_proba(X_test)[:, 1]
    panel_preds = cancer_panel_model.predict_proba(X_test)[:, 1]
    
    print(f"1. Sadece Master Model Başarısı    : {roc_auc_score(y_test, master_preds):.4f}")
    print(f"2. Sadece Kanser Paneli Başarısı   : {roc_auc_score(y_test, panel_preds):.4f}")
    print(f"3. Manuel Ağırlıklı Ortalama       : {roc_auc_score(y_test, preds_weighted):.4f}")
    print(f"4. Stacking (Lojistik Meta-Model)  : {roc_auc_score(y_test, preds_stacking):.4f}")
    
    print("\n" + "="*60)
    print("PIPELINE ADIM 4 & 5: KALİBRASYON VE EŞİK (THRESHOLD) OPTİMİZASYONU")
    print("="*60)
    # [Adım 4]: Stacking'teki Lojistik Regresyon Meta-Modelimiz yapısı gereği "Platt Scaling" 
    # mantığıyla çalışarak ağaçların (XGBoost/RF) dengesiz ham puanlarını, 
    # tıptaki güvenilir risk yüzdelerine (Olasılık Dağılımına) dönüştürür.
    
    # [Adım 5]: Klinik Maliyet (Eşik) Optimizasyonu
    
    # Eşik 1: Standart Eşik (Threshold = 0.5)
    y_pred_05 = (preds_stacking >= 0.5).astype(int)
    print("1) Standart Makine Öğrenmesi Eşiği (Risk Skoru >= %50 ise Patojenik Olarak İşaretle):")
    print(classification_report(y_test, y_pred_05, target_names=["Benign (0)", "Patojenik (1)"]))

    # Eşik 2: Klinik (Tıbbi) Eşik (Threshold = 0.35)
    # Patojenik bir gen mutasyonunu "sağlıklı" sanıp taburcu etme hatası (False Negative), 
    # sağlıklı birine "hastasın" deyip test yapmak (False Positive) hatasından 1000 kat daha tehlikelidir!
    # Bu yüzden hastalığı en zayıf ihtimalde bile "Patojenik" saymalıyız (Recall'ı Tavan Yaptır).
    clinical_threshold = 0.35
    y_pred_clinical = (preds_stacking >= clinical_threshold).astype(int)
    print(f"\n2) Klinik/Tıbbi Eşik Optimizasyonu (Risk Skoru >= %{int(clinical_threshold*100)} ise Patojenik Olarak İşaretle):")
    print("   *(Amaç: Gerçek hastaları gözden kaçırma/FN hatasını SIFIRA indirmek!)*")
    print(classification_report(y_test, y_pred_clinical, target_names=["Benign (0)", "Patojenik (1)"]))

    # --- 5. Modelin Yorumlanabilirliği (Explainable AI - SHAP) ---
    print("\n" + "="*60)
    print("MODEL YORUMLANABİLİRLİĞİ (SHAP Analizi)")
    print("="*60)
    
    # Eğitilmiş Master Model'in kararlarını SHAP ile yorumluyoruz (mock_df'ten gelen isimlerle)
    architect.explain_with_shap(X_test, feature_names=feature_names)

    # --- 6. Özellik Önem Dereceleri (Metinsel Çıktı) ---
    print("\n" + "="*60)
    print("MASTER MODEL: EN ÖNEMLİ 10 ÖZELLİK (Built-in Feature Importances)")
    print("="*60)
    
    importances = architect.master_model.feature_importances_
    # En yüksek skora sahip 10 özelliğin indekslerini bul
    top_idx = np.argsort(importances)[-10:]

    print("Model karar verirken metrik olarak en çok etki eden özellikler:\n")
    # Ters çevir (en yüksek etkiye sahip olandan en düşüğe doğru yazdır)
    for i in top_idx[::-1]:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")
