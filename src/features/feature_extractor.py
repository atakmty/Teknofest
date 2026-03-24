import pandas as pd
import numpy as np
import os
import sys

# To allow relative imports if run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.amino_acid import AA_PROPERTIES

class VariantFeatureExtractor:
    """
    Şartname bağlamı:
    - AL_  : Frekans/Popülasyon — eksik = sıfır (çok nadir varyant)
    - AA_  : Amino Asit Değişimi — 'A>V' formatı
    - EK_  : Evrimsel Korunmuşluk — eksik = eğitim medyanı (fit'te öğrenilir)
    - CAT_ : Kategorik Meta-Veri — one-hot (fit'te kategoriler öğrenilir)

    Kullanım:
        extractor = VariantFeatureExtractor()
        extractor.fit(train_df)          # istatistikleri eğitim setinden öğren
        train_features = extractor.transform(train_df)
        test_features  = extractor.transform(test_df)   # aynı istatistikler uygulanır
    """

    def __init__(self):
        self.aa_props = AA_PROPERTIES

        # fit() ile öğrenilecek istatistikler
        self.ek_medians_: dict = {}
        self.cat_categories_: dict = {}  # her CAT_ kolonu için geçerli kategoriler
        self.feature_columns_: list = []  # transform sonrası kolon sırası
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Yardımcı metodlar
    # ------------------------------------------------------------------ #

    def parse_aa_change(self, aa_str: str):
        """'A>V' → ('A', 'V')   |   geçersiz → (None, None)"""
        if pd.isna(aa_str) or '>' not in str(aa_str):
            return None, None
        parts = str(aa_str).split('>')
        if len(parts) == 2:
            ref, alt = parts[0].strip(), parts[1].strip()
            if ref and alt:
                return ref, alt
        return None, None

    # ------------------------------------------------------------------ #
    # Öznitelik çıkarımı — vektörize
    # ------------------------------------------------------------------ #

    def _extract_aa_features(self, df: pd.DataFrame, aa_col: str = 'AA_change') -> pd.DataFrame:
        """
        iterrows() yerine vektörize hesaplama.
        Geçersiz/bilinmeyen AA çiftleri için NaN üretir (sonra 0 ile doldurulur).
        """
        if aa_col not in df.columns:
            for col in ['Delta_Hydrophobicity', 'Delta_MW', 'Delta_pI',
                        'Delta_Volume', 'Polarity_Change']:
                df[col] = np.nan
            return df

        parsed   = df[aa_col].apply(self.parse_aa_change)
        refs     = parsed.map(lambda x: x[0] if x else None)
        alts     = parsed.map(lambda x: x[1] if x else None)
        valid    = refs.isin(self.aa_props) & alts.isin(self.aa_props)

        def _prop(aa_series, prop_name):
            return aa_series.map(
                lambda aa: self.aa_props[aa][prop_name] if aa in self.aa_props else np.nan
            )

        ref_hydro = _prop(refs, 'hydrophobicity')
        alt_hydro = _prop(alts, 'hydrophobicity')
        ref_mw    = _prop(refs, 'molecular_weight')
        alt_mw    = _prop(alts, 'molecular_weight')
        ref_pi    = _prop(refs, 'pI')
        alt_pi    = _prop(alts, 'pI')
        ref_vol   = _prop(refs, 'volume')
        alt_vol   = _prop(alts, 'volume')
        ref_pol   = _prop(refs, 'polarity')
        alt_pol   = _prop(alts, 'polarity')

        df['Delta_Hydrophobicity'] = np.where(valid, alt_hydro - ref_hydro, np.nan)
        df['Delta_MW']             = np.where(valid, alt_mw    - ref_mw,    np.nan)
        df['Delta_pI']             = np.where(valid, alt_pi    - ref_pi,    np.nan)
        df['Delta_Volume']         = np.where(valid, alt_vol   - ref_vol,   np.nan)
        df['Polarity_Change']      = np.where(
            valid,
            (ref_pol != alt_pol).astype(int),
            np.nan
        )
        return df

    # ------------------------------------------------------------------ #
    # fit — eğitim setinden istatistik öğren
    # ------------------------------------------------------------------ #

    def fit(self, X: pd.DataFrame, y=None) -> 'VariantFeatureExtractor':
        """
        EK_ medyanlarını ve CAT_ kategorilerini SADECE eğitim setinden öğren.
        Test setine bu istatistikler uygulanır (data leakage yok).
        """
        df = X.copy()

        # EK_ medyanları
        ek_cols = [c for c in df.columns if c.startswith('EK_')]
        self.ek_medians_ = {}
        for col in ek_cols:
            median = df[col].median()
            self.ek_medians_[col] = median if not pd.isna(median) else 0.0

        # CAT_ kategorileri (one-hot sütunlarını sabitlemek için)
        cat_cols = [c for c in df.columns if c.startswith('CAT_')]
        self.cat_categories_ = {}
        for col in cat_cols:
            self.cat_categories_[col] = sorted(df[col].dropna().unique().tolist())

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # transform — fit istatistiklerini uygula
    # ------------------------------------------------------------------ #

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        # AL_: eksik = 0 (nadir varyant, biyolojik gerekçe: şartname Bölüm 3.2)
        al_cols = [c for c in df.columns if c.startswith('AL_')]
        for col in al_cols:
            df[col] = df[col].fillna(0.0)

        # EK_: eğitim seti medyanı (fit'te öğrenildi)
        for col, median in self.ek_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median)

        # AA delta'ları: geçersiz varyant → 0 (tarafsız/sıfır değişim)
        aa_delta_cols = ['Delta_Hydrophobicity', 'Delta_MW',
                         'Delta_pI', 'Delta_Volume', 'Polarity_Change']
        for col in aa_delta_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fit'te öğrenilen kategorilere göre one-hot encoding.
        Test setinde görülmeyen kategoriler görmezden gelinir.
        Eğitimde görülen ama testte eksik olan kategoriler 0 ile doldurulur.
        """
        for col, categories in self.cat_categories_.items():
            if col not in df.columns:
                continue
            # Sadece bilinen kategoriler için dummy oluştur
            dummies = pd.get_dummies(df[col], prefix=col)
            expected_cols = [f"{col}_{cat}" for cat in categories[1:]]  # drop_first=True
            for ec in expected_cols:
                if ec not in dummies.columns:
                    dummies[ec] = 0
            # Sadece beklenen sütunları al
            dummies = dummies[[c for c in expected_cols if c in dummies.columns]]
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(
                "Transformer henüz fit edilmedi. Önce extractor.fit(X) çağırın."
            )

        df = X.copy()

        # 1. AA öznitelikleri (vektörize)
        df = self._extract_aa_features(df, aa_col='AA_change')

        # 2. Eksik değer doldurma (fit istatistikleriyle)
        df = self._impute(df)

        # 3. Kategorik kodlama (fit kategorileriyle)
        df = self._encode_categorical(df)

        # 4. Ham AA sütununu sil
        if 'AA_change' in df.columns:
            df.drop(columns=['AA_change'], inplace=True)
            
        # 4.5. Scikit-Learn (VarianceThreshold/Scaler) çökmemesi için tüm string/kategorik objeleri düşür
        non_numeric = df.select_dtypes(exclude=['number', 'bool']).columns
        if len(non_numeric) > 0:
            df.drop(columns=non_numeric, inplace=True)

        # 5. Kolon sırasını sabitle (ilk transform'da öğren, sonrakilerde uygula)
        if not self.feature_columns_:
            self.feature_columns_ = df.columns.tolist()
        else:
            # Eksik kolon varsa 0 ekle, fazla kolon varsa at
            for col in self.feature_columns_:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns_]

        # Booleans (One-Hot'tan gelen) -> int (0/1) dönüşümü
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Kolaylık metodu: fit + transform birlikte."""
        return self.fit(X, y).transform(X)

if __name__ == '__main__':
    # Test
    extractor = VariantFeatureExtractor()
    test_df = pd.DataFrame({
        'AA_change': ['A>V', 'R>D', 'invalid', np.nan],
        'CAT_type': ['missense', 'nonsense', 'missense', 'frameshift'],
        'EK_score': [0.5, np.nan, 0.8, 0.2]
    })
    res = extractor.fit_transform(test_df)
    print(res)
