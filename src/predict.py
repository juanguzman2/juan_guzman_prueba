import os
import pickle
import pandas as pd
from src.data_engineer import FeatureSelector

class Predictor:
    def __init__(self, modelo_nombre: str, base_dir: str = None):
        print("🔧 Inicializando Predictor...")

        self.modelo_nombre = modelo_nombre
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_path = os.path.join(self.base_dir, "models")
        self.tr_path = os.path.join(self.base_dir, "data", "raw", "prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv")
        self.features_path = os.path.join(self.base_dir, "data", "procesed", "features.csv")

        self.modelo = None
        self.columnas_modelo = None

        print("✅ Predictor inicializado correctamente.")

    def cargar_modelo(self):
        print(f"📦 Cargando modelo: {self.modelo_nombre}.pkl...")
        modelo_path = os.path.join(self.models_path, f"{self.modelo_nombre}.pkl")

        if not os.path.exists(modelo_path):
            modelos_disponibles = [f.replace(".pkl", "") for f in os.listdir(self.models_path) if f.endswith(".pkl")]
            raise FileNotFoundError(
                f"\n❌ Modelo '{self.modelo_nombre}' no encontrado en:\n  {modelo_path}\n"
                f"📁 Modelos disponibles: {modelos_disponibles}"
            )

        with open(modelo_path, 'rb') as archivo:
            self.modelo = pickle.load(archivo)

        self.columnas_modelo = self.modelo.feature_names_in_
        print("✅ Modelo cargado con éxito.")

    def cargar_datos(self, oot_path: str):
        print(f"📄 Cargando datos desde: {oot_path}")
        if not os.path.exists(oot_path):
            raise FileNotFoundError(f"Archivo OOT no encontrado en: {oot_path}")

        df_oot = pd.read_csv(oot_path)
        df_tr = pd.read_csv(self.tr_path)
        features = pd.read_csv(self.features_path)

        print("🔗 Generando ID para merge y unificando datos...")
        df_oot['id'] = df_oot['nit_enmascarado'].astype(str) + '#' + \
                       df_oot['num_oblig_orig_enmascarado'].astype(str) + '#' + \
                       df_oot['num_oblig_enmascarado'].astype(str)

        df_tr['id'] = df_tr['nit_enmascarado'].astype(str) + '#' + \
                      df_tr['num_oblig_orig_enmascarado'].astype(str) + '#' + \
                      df_tr['num_oblig_enmascarado'].astype(str)

        df_tr = df_tr.drop_duplicates(subset='id', keep='first')
        df = pd.merge(df_oot, df_tr, on='id', how='left')

        print("🧼 Aplicando FeatureSelector...")
        limpieza = FeatureSelector(df, features)
        df_clean = limpieza.fit_transform()

        print("✅ Datos cargados y limpiados.")
        return df_clean

    def predecir(self, oot_path: str, guardar_csv: bool = False):
        print("🚀 Iniciando proceso de predicción...")

        if self.modelo is None:
            self.cargar_modelo()

        df = self.cargar_datos(oot_path)
        X = df[self.columnas_modelo]

        print("🔮 Generando predicciones...")
        predicciones = self.modelo.predict(X)
        probabilidades = self.modelo.predict_proba(X)[:, 1]

        print("📊 Armando DataFrame de resultados...")
        df_resultado = pd.DataFrame({
            'id': df['id'],
            'var_rpta_alt': predicciones,
            'var_rpta_alt_prob': probabilidades
        })

        if guardar_csv:
            output_path = os.path.join(self.base_dir, "submission.csv")
            df_resultado.to_csv(output_path, index=False)
            print(f"💾 Archivo de salida guardado en: {output_path}")

        print("✅ Predicción completada.")
        return df_resultado
