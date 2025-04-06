import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Ruta al modelo (ajústala si usas otro nombre o modelo)
MODEL_PATH = "models/random_forest.pkl"

# Carga de modelo
print("🔍 Cargando modelo...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Cargar datos de prueba
print("📄 Cargando datos de test...")
df = pd.read_csv("data/train_test/test.csv")  # usa tus propios datos

X = df.drop(columns=["var_rpta_alt", "id"], errors="ignore")
y = df["var_rpta_alt"]

print("🔮 Generando predicciones...")
y_pred = model.predict(X)

# Calcular accuracy
acc = accuracy_score(y, y_pred)
print(f"✅ Accuracy: {acc}")

# Guardar score en archivo (para la pipeline)
with open("testing/accuracy.txt", "w") as out:
    out.write(str(acc))
