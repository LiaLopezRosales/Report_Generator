"""
Script para descargar recursos de NLTK necesarios
"""
import nltk
import os

def download_nltk_resources():
    """Descarga todos los recursos necesarios de NLTK"""
    print("Descargando recursos de NLTK...")
    
    resources = [
        'punkt',
        'stopwords',
        'vader_lexicon',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    
    for resource in resources:
        try:
            print(f"Descargando {resource}...")
            nltk.download(resource, quiet=False)
            print(f"✓ {resource} descargado correctamente")
        except Exception as e:
            print(f"✗ Error descargando {resource}: {e}")
    
    print("\nRecursos de NLTK descargados correctamente!")

if __name__ == "__main__":
    download_nltk_resources()



