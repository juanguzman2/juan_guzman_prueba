from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import FileResponse
from typing import Literal, Optional
from src.predict import Predictor
import os
import tempfile

app = FastAPI(
    title="API de Predicción Bancolombia",
    description="Servicio para generar predicciones usando modelos entrenados",
    version="1.1"
)

# Ruta base del proyecto (donde están las carpetas /models, /data, etc.)
BASE_DIR = os.getcwd()


@app.post("/predecir")
def predecir(
    modelo_nombre: Literal[
        "Best_Model_RF","RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"
    ] = Query(..., description="Nombre del modelo a usar (.pkl sin extensión)"),
    oot_path: Optional[str] = Query(None, description="Ruta completa del archivo CSV OOT"),
    file: Optional[UploadFile] = File(None)
):
    """
    Ejecuta la predicción usando un archivo OOT cargado o desde una ruta local,
    y retorna el archivo de salida submission.csv.
    """

    # Validación: se debe enviar una ruta o un archivo
    if file is None and oot_path is None:
        return {"error": "Debes proporcionar una ruta de archivo (oot_path) o subir un archivo .csv"}

    try:
        # Si se sube un archivo, lo guardamos temporalmente
        if file is not None:
            if not file.filename.endswith(".csv"):
                return {"error": "El archivo subido debe tener extensión .csv"}

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(file.file.read())
                tmp_path = tmp.name
            path_final = tmp_path
        else:
            path_final = oot_path

        # Ejecutar la predicción
        predictor = Predictor(modelo_nombre=modelo_nombre, base_dir=BASE_DIR)
        predictor.predecir(oot_path=path_final, guardar_csv=True)

        output_path = os.path.join(BASE_DIR, "submission.csv")

        # Limpiar archivo temporal si aplica
        if file is not None:
            os.remove(tmp_path)

        return FileResponse(output_path, media_type="text/csv", filename="submission.csv")

    except FileNotFoundError as e:
        return {"error": str(e)}

    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}
