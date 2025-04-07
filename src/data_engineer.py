import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class FeatureSelector:
    def __init__(self, df, features_csv, correlation_threshold=0.9, drop_first=True, max_cardinality=30):
        self.df = df.copy()
        self.features_csv = list(features_csv.iloc[:, 0])
        self.correlation_threshold = correlation_threshold
        self.drop_first = drop_first
        self.max_cardinality = max_cardinality

        if "id" not in self.df.columns:
            required_cols = ['nit_enmascarado', 'num_oblig_orig_enmascarado', 'num_oblig_enmascarado']
            missing_cols = [col for col in required_cols if col not in self.df.columns]

            if missing_cols:
                raise ValueError(f"Faltan las siguientes columnas para construir el ID: {missing_cols}")

            self.df["id"] = self.df["nit_enmascarado"].astype(str) + "#" + \
                            self.df["num_oblig_orig_enmascarado"].astype(str) + "#" + \
                            self.df["num_oblig_enmascarado"].astype(str)

        self.preprocessor = None
        self.selected_features = None

    def _tratamiento_numericas_basico_auto(self, verbose=False):
        df = self.df.copy()
        num_vars = df.select_dtypes(include=['int64', 'float64']).columns

        for col in num_vars:
            if df[col].isnull().sum() > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                if verbose:
                    print(f"[Imputación] '{col}': nulos imputados con mediana = {mediana:.2f}")

            skew = df[col].skew()
            if skew > 1 and (df[col] >= 0).all():
                df[f"{col}_log"] = np.log1p(df[col])
                if verbose:
                    print(f"[Log1p] '{col}': skew={skew:.2f} → transformación log aplicada")

            p50 = df[col].quantile(0.5)
            p99 = df[col].quantile(0.99)
            if p50 != 0 and ((p99 - p50) / abs(p50)) > 3:
                p01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=p01, upper=p99)
                if verbose:
                    print(f"[Winsor] '{col}': winsorizada entre P01={p01:.2f} y P99={p99:.2f}")
        return df

    def fit_transform(self, verbose=False):
        df = self.df.copy()
        
        if 'id' not in df.columns:
            raise ValueError("La columna 'id' es requerida en el DataFrame.")

        tiene_y = 'var_rpta_alt' in df.columns

        print("Limpiando variables numéricas...")
        df_clean = self._tratamiento_numericas_basico_auto(verbose=verbose)

        if tiene_y:
            print("Separando X e y...")
            X = df_clean.drop(columns=['var_rpta_alt', 'id'])
            y = df_clean['var_rpta_alt']

            # Eliminar filas con NaN en y
            valid_idx = y.notna()
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        else:
            print("Modo inferencia: No se encontró 'var_rpta_alt'")
            X = df_clean.drop(columns=['id'])
            y = None
            valid_idx = df_clean.index  # usar todos los índices

        print("Detectando tipos de variables...")
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        categorical_features = [col for col in categorical_features if X[col].nunique() <= self.max_cardinality]

        print("Preprocesando variables...")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first' if self.drop_first else None))
        ])
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        print("Transformando variables...")
        X_preprocessed = self.preprocessor.fit_transform(X)

        print("Reconstruyendo nombres de columnas...")
        cat_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_names)

        X_preprocessed_df = pd.DataFrame(
            X_preprocessed.toarray() if hasattr(X_preprocessed, "toarray") else X_preprocessed,
            columns=all_feature_names,
            index=X.index
        )

        print("Seleccionando variables...")
        self.selected_features = self.features_csv

        print("Ajustando columnas faltantes...")
        for col in self.selected_features:
            if col not in X_preprocessed_df.columns:
                X_preprocessed_df[col] = 0
                if verbose:
                    print(f"[Padding] '{col}': columna faltante rellenada con ceros.")

        print("Ordenando columnas...")
        X_final = X_preprocessed_df[self.selected_features]

        print("Finalizando DataFrame...")
        if tiene_y:
            df_final = pd.concat([df_clean.loc[valid_idx, ['id']], y, X_final], axis=1)
        else:
            df_final = pd.concat([df_clean.loc[valid_idx, ['id']], X_final], axis=1)

        return df_final


    def get_selected_features(self):
        return list(self.selected_features)
