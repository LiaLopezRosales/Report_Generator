import os
import json
import pandas as pd
from datetime import datetime
from pandas.api.types import is_scalar
from pathlib import Path
import re

# Carpeta donde están los JSON originales
INPUT_DIR = "/home/lia/Escritorio/4to Año/PL/Proyecto/Datos procesados/Data_articles2/"
OUTPUT_DIR = "Data/teleSUR_tv/agrupados2/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def limpiar_nombre(nombre):
    """
    Limpia el título para usarlo como nombre de archivo.
    """
    nombre = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ ]', '', nombre)
    nombre = "_".join(nombre.strip().split())
    return nombre[:80]  # limitar tamaño del nombre

def limpiar_valores_json(d):
    """
    Convierte valores problemáticos (NaT, NaN, Timestamp) a strings o None.
    Deja listas y diccionarios tal cual.
    """
    limpio = {}
    for k, v in d.items():
        if isinstance(v, pd.Timestamp):
            limpio[k] = v.strftime("%Y-%m-%dT%H:%M:%S")
        elif is_scalar(v) and pd.isna(v):  # solo revisar si es escalar
            limpio[k] = None
        else:
            limpio[k] = v
    return limpio

# --- Cargar todos los JSONs ---
data = []
for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".json"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    item = json.load(f)
                    if isinstance(item, dict):
                        data.append(item)
                    elif isinstance(item, list):
                        data.extend(item)  # si es lista, agregar todos
                except json.JSONDecodeError as e:
                    print(f"Error leyendo {filepath}: {e}")

# --- Pasar a DataFrame ---
df = pd.DataFrame(data)

# --- Extraer fecha de forma segura ---
def extraer_fecha(meta):
    if isinstance(meta, dict) and "date" in meta:
        return meta["date"]
    return None

df["date"] = pd.to_datetime(
    df["source_metadata"].apply(extraer_fecha),
    errors="coerce"
)

# Crear columna semana (YYYY-Wxx)
df["week"] = df["date"].dt.strftime("%Y-W%U")

# --- Guardar cada noticia en carpeta semanal ---
for _, row in df.iterrows():
    fecha = row["date"]

    # Si no tiene fecha, guardarlo en carpeta "sin_fecha"
    if pd.isna(fecha):
        semana = "sin_fecha"
        fecha_str = "sin_fecha"
    else:
        semana = row["week"]
        fecha_str = fecha.strftime("%Y-%m-%d")

    titulo = limpiar_nombre(row["title"])

    semana_dir = Path(OUTPUT_DIR) / semana
    semana_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{titulo}_{fecha_str}.json"
    filepath = semana_dir / filename

    # Convertir fila a dict limpio
    row_dict = limpiar_valores_json(row.to_dict())

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(row_dict, f, ensure_ascii=False, indent=2)

print("✅ Organización completada.")
