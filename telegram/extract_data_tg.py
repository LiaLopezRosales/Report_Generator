import os
import json
import asyncio
from collections import defaultdict
from telethon import TelegramClient
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str | None = None):
    """Lee variables de entorno en minúsculas o mayúsculas."""
    return os.environ.get(key) or os.environ.get(key.upper()) or default


class ScraperT:
    def __init__(self, group_username: str | None = None, api_id: str | None = None, api_hash: str | None = None, max_workers: int = 5):
        # Compatibilidad con .env: claves esperadas api_id / api_hash
        self.api_id = api_id or _get_env("api_id")
        self.api_hash = api_hash or _get_env("api_hash")
        if not self.api_id or not self.api_hash:
            raise ValueError("Configura api_id y api_hash en .env")

        self.group_username = group_username or _get_env("tg_group_username", "teleSURtv")
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

    def check_last_date(self):
        pass

    async def process_message_batch(self, messages_batch, start_of_two_days_ago, end_of_today, extract_all):
        """Procesa un lote de mensajes en paralelo"""
        async with self.semaphore:
            batch_data = defaultdict(list)
            
            for message in messages_batch:
                try:
                    if not extract_all:
                        if not (start_of_two_days_ago <= message.date <= end_of_today):
                            continue
                    
                    year_month = message.date.strftime("%Y-%m")
                    
                    if not message.text:
                        continue
                    
                    sender_id = str(message.sender_id)

                    reactions = {}
                    total_reactions = 0
                    if hasattr(message, 'reactions') and message.reactions:
                        if hasattr(message.reactions, 'results') and message.reactions.results:
                            for reaction in message.reactions.results:
                                if hasattr(reaction, 'reaction'):
                                    if hasattr(reaction.reaction, 'emoticon'):
                                        reaction_type = reaction.reaction.emoticon
                                    elif hasattr(reaction.reaction, 'document_id'):
                                        reaction_type = f"custom_{reaction.reaction.document_id}"
                                    else:
                                        reaction_type = str(reaction.reaction)
                                else:
                                    reaction_type = "unknown"
                                
                                reactions[reaction_type] = reaction.count
                                total_reactions += reaction.count
                    
                    photo_path = None
                    #if message.photo:
                     #   photo_path = await self.download_photo(message)

                    batch_data[year_month].append(
                        {
                            "text": message.text,
                            "sender_id": sender_id,
                            "date": message.date.isoformat(),
                            "bot": message.via_bot_id,
                            "views": message.views,
                            "message_id": message.id,
                            "is_reply": message.reply_to_msg_id is not None,
                            "reply_to": message.reply_to_msg_id if message.reply_to_msg_id else None,
                            "reactions": reactions,
                            "total_reactions": total_reactions,
                            "photo_path": photo_path
                        }
                    )
                        
                except Exception as e:
                    print(f"Error procesando mensaje {message.id}: {e}")
                    continue
            
            return batch_data

    async def download_photo(self, message):
        """Descarga una foto de manera asíncrona y devuelve la ruta relativa"""
        try:
            dt = message.date
            year = dt.strftime('%Y')
            month = dt.strftime('%m')
            day = dt.strftime('%d')
            photo_dir = os.path.join("Data", "photos", self.group_username, year, month, day)
            os.makedirs(photo_dir, exist_ok=True)
            ts_compact = dt.strftime('%Y%m%dT%H%M%S')
            base_name = f"foto_{ts_compact}_msg{message.id}_from{message.sender_id}"
            filename = f"{base_name}.jpg"
            full_path = os.path.join(photo_dir, filename)
            
            await message.download_media(file=full_path)
            return os.path.relpath(full_path, start=os.getcwd())
        except Exception as e:
            print(f"Error descargando foto del mensaje {message.id}: {e}")
            return None

    async def extract_group_sms(self, limit: int = None, extract_all: bool = False, n: int = 2, batch_size: int = 100) -> None:
        os.makedirs("Data", exist_ok=True)
        async with TelegramClient("session_name", self.api_id, self.api_hash) as client:
            print(f"Descargando mensajes de {self.group_username}...")
            
            try:
                entity = await client.get_entity(self.group_username)
               
                entity_type = "Canal" if hasattr(entity, 'broadcast') and entity.broadcast else "Grupo"
                print(f"Conectado a: {entity_type} - {entity.title}")
                
                if hasattr(entity, 'restriction_reason') and entity.restriction_reason:
                    print(f"⚠️ Advertencia: {entity.restriction_reason}")
                    
            except Exception as e:
                print(f"❌ Error al acceder a {self.group_username}: {e}")
                return
            
            monthly_data = defaultdict(list)
            
            if not extract_all:
                today = datetime.now(pytz.UTC).date()
                two_days_ago = today - timedelta(days=n)
                
                start_of_two_days_ago = datetime.combine(two_days_ago, datetime.min.time()).replace(tzinfo=pytz.UTC)
                end_of_today = datetime.combine(today, datetime.max.time()).replace(tzinfo=pytz.UTC)

                print(f"Extrayendo mensajes desde hace {n} días ({two_days_ago.strftime('%Y-%m-%d')}) hasta hoy ({today.strftime('%Y-%m-%d')})")
            else:
                start_of_two_days_ago = None
                end_of_today = None
                print("Extrayendo todos los mensajes del grupo")
            
            message_count = 0
            filtered_count = 0
            messages_batch = []
            processing_tasks = []
            
            print(f"Procesando mensajes en lotes de {batch_size} con máximo {self.max_workers} hilos concurrentes...")
            
            async for message in tqdm_asyncio(
                client.iter_messages(
                    entity, 
                    limit=limit,
                    offset_date=end_of_today if not extract_all else None
                ),
                desc="Recolectando mensajes",
                unit="msg",
            ):
                try:
                    if not extract_all:
                        if not (start_of_two_days_ago <= message.date <= end_of_today):
                            break
                    
                    message_count += 1
                    messages_batch.append(message)
                    
                    
                    if len(messages_batch) >= batch_size:
                        task = self.process_message_batch(
                            messages_batch.copy(), 
                            start_of_two_days_ago, 
                            end_of_today, 
                            extract_all
                        )
                        processing_tasks.append(task)
                        messages_batch.clear()
                        
                        
                        if len(processing_tasks) >= self.max_workers:
                            completed_batches = await asyncio.gather(*processing_tasks[:self.max_workers])
                            processing_tasks = processing_tasks[self.max_workers:]
                            
                            
                            for batch_data in completed_batches:
                                for year_month, msgs in batch_data.items():
                                    monthly_data[year_month].extend(msgs)
                                    filtered_count += len(msgs)
                    
                except Exception as e:
                    print(f"Error recolectando mensaje {message_count}: {e}")
                    continue
            
            
            if messages_batch:
                task = self.process_message_batch(
                    messages_batch, 
                    start_of_two_days_ago, 
                    end_of_today, 
                    extract_all
                )
                processing_tasks.append(task)
            
            if processing_tasks:
                print("Procesando lotes restantes...")
                completed_batches = await asyncio.gather(*processing_tasks)
                
                for batch_data in completed_batches:
                    for year_month, msgs in batch_data.items():
                        monthly_data[year_month].extend(msgs)
                        filtered_count += len(msgs)
            
            print(f"\nResumen:")
            print(f"Mensajes procesados: {message_count}")
            print(f"Mensajes en rango de fechas: {filtered_count}")
            print(f"Mensajes guardados por mes: {len(monthly_data.keys())}")
            
            if filtered_count == 0 and not extract_all:
                print(f"\n No se encontraron mensajes en el rango de fechas especificado.")
                print(f"   El grupo podría tener mensajes de fechas diferentes.")
                print(f"   Usa extract_all=True para extraer todos los mensajes disponibles.")
            
            await self.save_data_parallel(monthly_data)

    async def save_data_parallel(self, monthly_data):
        """Guarda los datos en archivos usando hilos para operaciones I/O"""
        def save_file(year_month, msgs):
            try:
        
                year, month = year_month.split('-')
                year_dir = os.path.join("Data", self.group_username, year)
                os.makedirs(year_dir, exist_ok=True)
                filename = os.path.join(
                    year_dir,
                    f"mensajes_{self.group_username}_{year_month}.json"
                )
                
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(msgs, f, ensure_ascii=False, indent=4)
                return f"Mensajes de {year_month} guardados en {filename}."
            except Exception as e:
                return f"Error en el archivo: {year_month} de {filename} - {e}"

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            save_tasks = []
            for year_month, msgs in monthly_data.items():
                task = loop.run_in_executor(executor, save_file, year_month, msgs)
                save_tasks.append(task)
            
            if save_tasks:
                results = await asyncio.gather(*save_tasks)
                for result in results:
                    print(result)
