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
        self.tr_path = os.path.join(self.base_dir, "data", "procesed", "df_train.csv")
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

        # Cargar datasets base
        df_oot = pd.read_csv(oot_path)
        df_hist_scores = pd.read_csv(os.path.join(self.base_dir, "data", "raw", "prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv"))
        df_clientes = pd.read_csv(os.path.join(self.base_dir, "data", "raw", "prueba_op_master_customer_data_enmascarado_completa.csv"))
        df_pagos = pd.read_csv(os.path.join(self.base_dir, "data", "raw", "prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv"))
        features = pd.read_csv(self.features_path)

        print("🧹 Limpiando y formateando datos históricos...")

        # Ajuste fecha corte
        df_pagos['fecha_corte'] = df_pagos['fecha_corte'].astype(str).str[:6].astype(int)
        df_clientes['fecha_corte'] = df_clientes['year'].astype(str) + df_clientes['month'].astype(str).str.zfill(2)

        # Drop duplicates manteniendo el más reciente
        for df_name, df_data in [('Clientes', df_clientes), ('Pagos', df_pagos), ('Hist Scores', df_hist_scores)]:
            df_data.sort_values(by='fecha_corte', ascending=False, inplace=True)
            df_data.drop_duplicates(subset='nit_enmascarado', keep='first', inplace=True)
            print(f"✔️ {df_name} limpio y único por nit.")

        print("🔗 Realizando merges con información de cliente, pagos e histórico de scores...")

        df_test = pd.merge(df_oot, df_clientes, on='nit_enmascarado', how='left')
        df_test = pd.merge(df_test, df_pagos, on='nit_enmascarado', how='left')
        df_test = pd.merge(df_test, df_hist_scores, on='nit_enmascarado', how='left')

        df_test['id'] = df_test['nit_enmascarado'].astype(str) + '#' + \
                        df_test['num_oblig_orig_enmascarado'].astype(str) + '#' + \
                        df_test['num_oblig_enmascarado'].astype(str)

        

        print("🧼 Aplicando FeatureSelector...")
        limpieza = FeatureSelector(df_test, features)
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
