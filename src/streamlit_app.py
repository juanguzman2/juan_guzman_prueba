import streamlit as st
import requests

# URL base de la API
API_URL = "http://127.0.0.1:8000/predecir"

st.set_page_config(page_title="Predicción Bancolombia", layout="centered")
st.title("🔍 Predicción Bancolombia")
st.markdown("Sube un archivo OOT (.csv), selecciona un modelo y descarga el resultado.")

# Selección del modelo
modelo = st.selectbox("📌 Selecciona el modelo a usar", [
        "Best_Model_RF","RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier"])

# Upload del archivo OOT
archivo = st.file_uploader("📄 Carga tu archivo CSV para predicción", type="csv")

# Botón de predicción
if st.button("🔮 Generar predicción"):
    if archivo is None:
        st.warning("⚠️ Debes subir un archivo .csv para continuar.")
    else:
        # Mostrar spinner mientras se envía la solicitud
        with st.spinner("Enviando archivo al backend y generando predicción..."):

            try:
                # Enviar solicitud POST con archivo
                files = {"file": (archivo.name, archivo, "text/csv")}
                params = {"modelo_nombre": modelo}
                response = requests.post(API_URL, files=files, params=params)

                if response.status_code == 200:
                    st.success("✅ Predicción generada exitosamente.")
                    st.download_button(
                        label="📥 Descargar submission.csv",
                        data=response.content,
                        file_name="submission.csv",
                        mime="text/csv"
                    )
                else:
                    error_msg = response.json().get("error", "Error desconocido")
                    st.error(f"❌ Error en la predicción:\n\n{error_msg}")

            except Exception as e:
                st.error(f"❌ Error al conectar con la API:\n\n{e}")
