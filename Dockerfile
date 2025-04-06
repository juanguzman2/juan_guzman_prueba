# Imagen base oficial de Python
FROM python:3.10

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de requerimientos y código
COPY requirements.txt requirements.txt
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Exponer el puerto 8000
EXPOSE 8000

# Comando por defecto al correr el contenedor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
