from bs4 import BeautifulSoup
import os 
import requests
import re
import json
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class Scraper:
    def __init__(self, data_path="Data/teleSUR_tv", max_workers=5):
        self.data_path = data_path
        self.max_workers = max_workers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.lock = threading.Lock()
    
    def extract_urls_from_json(self, json_file_path):
        """
        Extrae todas las URLs del archivo JSON de mensajes de Telegram
        manteniendo la referencia al archivo y posición original
        """
        urls_with_metadata = []
        try:
            for i in os.listdir(json_file_path):
                year_path = os.path.join(json_file_path, i)
                if os.path.isdir(year_path):
                    for j in os.listdir(year_path):
                        json_file = os.path.join(year_path, j)
                        if j.endswith('.json'):
                            with open(json_file, 'r', encoding='utf-8') as file:
                                messages = json.load(file)
                            
                            for index, message in enumerate(messages):
                                text = message.get('text', '')
                                url_pattern = r'https?://[^\s\n]+'
                                found_urls = re.findall(url_pattern, text)
                                
                                for url in found_urls:
                                    url_metadata = {
                                        'url': url,
                                        'json_file': json_file,
                                        'message_index': index,
                                        'message_id': message.get('message_id'),
                                        'date': message.get('date'),
                                        'text': text,
                                        'photo_path': message.get('photo_path'),
                                        'views': message.get('views'),
                                        'reactions': message.get('reactions', {}),
                                        'total_reactions': message.get('total_reactions', 0)
                                    }
                                    urls_with_metadata.append(url_metadata)
                
        except Exception as e:
            print(f"Error al leer el archivo JSON: {e}")
            
        return urls_with_metadata  
    
    def scrape_article_content(self, url, url_metadata=None):
        """
        Hace scraping de una URL y extrae el contenido de las etiquetas <article>
        """
        try:
            with self.lock:
                print(f"Haciendo scraping de: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metas = soup.find_all("meta", property=lambda x:x and x.startswith("article:tag"))
            paragraphs = soup.find('div', class_="content-area__text__full")
            
            content = {
                'url': url,
                'title': '',
                'section': '',
                'tags': [],
                'text': '',
                'source_metadata': url_metadata 
            }
            
            title_tag = soup.find('title')
            if title_tag:
                content['title'] = title_tag.get_text().strip()

            section_tag = soup.find('meta', {'property': 'article:section'})
            if section_tag:
                content['section'] = section_tag.get('content', '').strip()
            
            for meta in metas:
                tag_content = meta.get('content', '').strip()
                if tag_content:
                    content['tags'].append(tag_content)
            
            paragraph_texts = []
            if paragraphs:
                p_tags = paragraphs.find_all('p', recursive=False)
                for i, p_tag in enumerate(p_tags):
                    if not p_tag.attrs:
                        paragraph_texts.append(p_tag.get_text().strip())
            text = '\n'.join(paragraph_texts)
            
            text = re.sub(r'\n*LEA TAMBIÉN:\n.*?(\n|$)', '\n', text)

            content['text'] = text

            return content
            
        except requests.exceptions.RequestException as e:
            with self.lock:
                print(f"Error al hacer request a {url}: {e}")
            return None
        except Exception as e:
            with self.lock:
                print(f"Error inesperado al procesar {url}: {e}")
            return None
    
    def process_single_url(self, url_data_with_index):
        """
        Procesa una sola URL con su metadata y índice para uso en concurrencia
        """
        url_data, index, total = url_data_with_index
        url = url_data['url']
        
        with self.lock:
            print(f"Procesando {index}/{total}: {url}")
            print(f"  Origen: {url_data['json_file']} (mensaje #{url_data['message_index']})")
        
        article_data = self.scrape_article_content(url, url_data)
        
        if article_data and article_data['text']:
            return article_data, url_data, index
        else:
            with self.lock:
                print(f"No se encontró texto en: {url}")
            return None, url_data, index
    
    def scrape_urls_from_data(self, json_file_path, output_dir="scraped_articles"):
        """
        Extrae URLs del JSON y hace scraping de cada una manteniendo referencias
        Ahora usa procesamiento concurrente para mayor eficiencia
        """
        os.makedirs(output_dir, exist_ok=True)
        
        urls_with_metadata = self.extract_urls_from_json(json_file_path)
        print(f"Encontradas {len(urls_with_metadata)} URLs con metadata")
        
        seen_urls = set()
        telesur_urls_metadata = []
        for url_data in urls_with_metadata:
            if 'telesurtv.net' in url_data['url'] and url_data['url'] not in seen_urls:
                telesur_urls_metadata.append(url_data)
                seen_urls.add(url_data['url'])
        
        print(f"URLs únicas de teleSUR: {len(telesur_urls_metadata)}")
        print(f"Usando {self.max_workers} workers concurrentes")
        
        scraped_data = []
        
        url_data_with_index = [
            (url_data, i+1, len(telesur_urls_metadata)) 
            for i, url_data in enumerate(telesur_urls_metadata)
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.process_single_url, url_item): url_item[0]['url'] 
                for url_item in url_data_with_index
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article_data, url_data, index = future.result()
                    
                    if article_data:
                        scraped_data.append(article_data)
                        
                        safe_filename = re.sub(r'[^\w\-_.]', '_', urlparse(url).path.split('/')[-1])
                        if not safe_filename:
                            safe_filename = f"article_{index}"
                            
                        output_file = os.path.join(output_dir, f"{safe_filename}.json")
                        
                        try:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(article_data, f, ensure_ascii=False, indent=2)
                            with self.lock:
                                print(f"✓ Guardado: {output_file}")
                        except Exception as e:
                            with self.lock:
                                print(f"Error al guardar {output_file}: {e}")
                        
                except Exception as exc:
                    with self.lock:
                        print(f"Error procesando {url}: {exc}")

        print(f"\n📊 Resumen: {len(scraped_data)} artículos procesados exitosamente")

        consolidated_file = os.path.join(output_dir, "all_articles_with_metadata.json")
        try:
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Archivo consolidado guardado: {consolidated_file}")
        except Exception as e:
            print(f"Error al guardar archivo consolidado: {e}")
        
        index_data = []
        for i, article_data in enumerate(scraped_data):
            metadata = article_data.get('source_metadata', {})
            index_entry = {
                'article_id': f"{i}",
                'url': article_data['url'],
                'title': article_data['title'],
                'section': article_data['section'],
                'tags': article_data['tags'],
                'text': article_data['text'],
                'original_json_file': metadata.get('json_file'),
                'message_index': metadata.get('message_index'),
                'message_id': metadata.get('message_id'),
                'date': metadata.get('date'),
                'original_text': metadata.get('text'),
                'photo_path': metadata.get('photo_path'),
                'views': metadata.get('views'),
                'reactions': metadata.get('reactions'),
                'total_reactions': metadata.get('total_reactions')
            }
            index_data.append(index_entry)
    
        index_file = os.path.join(output_dir, "articles_index.json")
        try:
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            print(f"✓ Índice de artículos guardado: {index_file}")
        except Exception as e:
            print(f"Error al guardar índice: {e}")
        
        return scraped_data
    
    def find_article_by_message(self, index_file_path, message_id=None, json_file=None, message_index=None):
        """
        Busca artículos basándose en la referencia del mensaje original
        """
        try:
            with open(index_file_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            results = []
            for entry in index_data:
                match = False
                
                if message_id and entry.get('message_id') == message_id:
                    match = True
                elif json_file and message_index is not None:
                    if (entry.get('original_json_file', '').endswith(json_file) and 
                        entry.get('message_index') == message_index):
                        match = True
                
                if match:
                    results.append(entry)
            
            return results
            
        except Exception as e:
            print(f"Error al buscar en el índice: {e}")
            return []
    
    def get_original_message(self, json_file_path, message_index):
        """
        Obtiene el mensaje original completo del archivo JSON
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            
            if 0 <= message_index < len(messages):
                return messages[message_index]
            else:
                print(f"Índice {message_index} fuera de rango en {json_file_path}")
                return None
                
        except Exception as e:
            print(f"Error al obtener mensaje original: {e}")
            return None