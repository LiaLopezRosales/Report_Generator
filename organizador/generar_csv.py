import os
import json
import pandas as pd
from pathlib import Path

# Carpeta donde quedaron organizados los archivos
ORG_DIR = "Data/teleSUR_tv/agrupados/"
OUTPUT_CSV = "noticias_tracking.csv"

rows = []

# Recorrer todas las subcarpetas y archivos
for root, _, files in os.walk(ORG_DIR):
    for file in files:
        if file.endswith(".json"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    item = json.load(f)

                # Validar que es un diccionario
                if not isinstance(item, dict):
                    print(f"⚠️ Archivo inválido {filepath}")
                    continue

                # Extraer campos principales
                titulo = item.get("title", "sin_titulo")
                fecha = item.get("date", "sin_fecha")
                url = item.get("url") or item.get("source_metadata", {}).get("url", "sin_url")

                rows.append({
                    "titulo": titulo,
                    "fecha": fecha,
                    "url": url,
                    "archivo": filepath
                })

            except Exception as e:
                print(f"⚠️ Error leyendo {filepath}: {e}")

# Crear DataFrame y exportar a CSV
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ CSV generado: {OUTPUT_CSV}")
