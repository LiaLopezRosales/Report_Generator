from telegram.extract_data_tg import ScraperT
from src.summarization.summarizer import PersonalizedSummarizer, TextRankSummarizer
from src.recommendation.vectorizer import NewsVectorizer, UserProfileVectorizer
from src.recommendation.matcher import NewsMatcher
from src.recommendation.user_profile import UserProfileManager
from src.recommendation.report_generator import ReportGenerator
from src.nlp.preprocessing import TextPreprocessor
from src.nlp.regex_annotator import RegexAnnotator
import os 
import json


path = 'Data_articles'
data_dirs = [x for x in os.listdir(path) if not x.startswith(".")]

def load_raw_data(limit=None):
    """Carga datos crudos de artículos"""
    all_data = []
    count = 0
    for data_dir in data_dirs:
        dir_path = os.path.join(path, data_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(dir_path, filename)) as f:
                        article = json.load(f)
                        all_data.append(article)
                        count += 1
                        if limit and count >= limit:
                            return all_data
                except:
                    continue
    return all_data


def prepare_articles(raw_data, text_processor, annotator, news_vectorizer):
    """
    Prepara artículos: extrae texto, categoriza con regex, limpia y vectoriza
    
    Returns:
        Lista de artículos procesados con vectores y metadatos
    """
    articles = []
    clean_texts = []
    
    print(f"\n📰 Procesando {len(raw_data)} artículos...")
    import tqdm 
    for i, article_data in enumerate(tqdm.tqdm(raw_data)):
        try:
            # Extraer texto
            text = article_data.get('text', '')
            if not text:
                continue
            
            # Anotar con regex para extraer categorías
            annotations = annotator.annotate(text)
            
            # Preprocesar texto
            clean_tokens = text_processor.preprocess(text)
            clean_text = ' '.join(clean_tokens)
            clean_texts.append(clean_text)
            
            # Guardar artículo procesado (sin vector aún)
            articles.append({
                'id': i,
                'title': article_data.get('title', 'Sin título'),
                'text': text,
                'clean_text': clean_text,
                'categories': annotations['categories'],
                'section': article_data.get('section', 'General'),
                'tags': article_data.get('tags', []),
                'url': article_data.get('url', ''),
                'source_metadata': article_data.get('source_metadata', {}),
            })
            
        except Exception as e:
            continue
    
    print(f"✅ {len(articles)} artículos procesados exitosamente")
    
    # Vectorizar todos los textos limpios
    print(f"\n🔢 Vectorizando artículos con TF-IDF...")
    article_matrix = news_vectorizer.fit_transform0(clean_texts)
    print(f"✅ Matriz de artículos: {article_matrix.shape}")
    
    # Agregar vectores a los artículos
    for i, article in enumerate(articles):
        article['vector'] = article_matrix[i].tolist()
    
    return articles


def save_processed_articles(articles, filepath='processed_articles.json'):
    """Guarda los artículos procesados en un archivo JSON"""
    print(f"\n💾 Guardando artículos procesados en {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"✅ Artículos guardados exitosamente")


def load_processed_articles(filepath='processed_articles.json'):
    """Carga los artículos procesados desde un archivo JSON"""
    if os.path.exists(filepath):
        print(f"\n📂 Cargando artículos procesados desde {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"✅ {len(articles)} artículos cargados desde cache")
        return articles
    return None


def create_simulated_users():
    """Crea perfiles de usuarios simulados con diferentes intereses basados en categorías regex"""
    users = [
        {
            'name': 'Sofía - Crítica de Arte',
            'profile_text': (
                'Soy una apasionada del arte contemporáneo, las exposiciones y las galerías. '
                'Me interesan las obras de artistas emergentes, el muralismo, la escultura y '
                'la fotografía artística. Sigo festivales culturales, bienales de arte, '
                'inauguraciones de museos y eventos de patrimonio cultural. Me fascina el '
                'teatro, la danza, el cine de autor y las manifestaciones artísticas urbanas. '
                'Disfruto la música clásica, jazz, y expresiones folclóricas tradicionales.'
            )
        },
        {
            'name': 'Diego - Ambientalista',
            'profile_text': (
                'Me dedico a la conservación ambiental y protección de ecosistemas. '
                'Sigo temas de biodiversidad, especies en peligro de extinción, reservas naturales '
                'y parques nacionales. Me preocupan los desastres naturales como terremotos, '
                'inundaciones y huracanes. Denuncio la deforestación, contaminación de ríos, '
                'derrames de petróleo y el cambio climático. Apoyo energías renovables, '
                'reciclaje y desarrollo sostenible. Me interesan proyectos de reforestación '
                'y la protección de océanos y recursos hídricos.'
            )
        },
        {
            'name': 'Laura - Educadora Cultural',
            'profile_text': (
                'Me apasiona la educación, la literatura y la promoción cultural. '
                'Sigo lanzamientos de libros, ferias literarias, conciertos y recitales de poesía. '
                'Me interesan programas educativos, becas, talleres artísticos y actividades '
                'para niños y jóvenes. Apoyo bibliotecas comunitarias, centros culturales '
                'y espacios de creación artística. Me gusta el teatro comunitario, '
                'la música folclórica y las tradiciones ancestrales. Valoro la preservación '
                'del patrimonio inmaterial y las lenguas indígenas.'
            )
        },
        {
            'name': 'Martín - Fotógrafo de Naturaleza',
            'profile_text': (
                'Soy fotógrafo especializado en naturaleza, paisajes y vida silvestre. '
                'Me apasionan los parques naturales, santuarios de fauna, volcanes y montañas. '
                'Documento especies animales, aves migratorias, flora endémica y ecosistemas únicos. '
                'Me interesan expediciones científicas, descubrimientos de nuevas especies '
                'y proyectos de conservación de hábitats. Sigo fenómenos naturales, auroras, '
                'eclipses y eventos astronómicos. Apoyo el turismo ecológico y responsable.'
            )
        },
        {
            'name': 'Carmen - Historiadora del Arte',
            'profile_text': (
                'Investigo historia del arte latinoamericano, arquitectura colonial y '
                'patrimonio histórico. Me fascinan las restauraciones de monumentos, '
                'excavaciones arqueológicas y descubrimientos de sitios históricos. '
                'Estudio arte prehispánico, culturas indígenas y tradiciones artesanales. '
                'Me interesan museos, archivos históricos, documentales culturales '
                'y la preservación de arte sacro. Valoro el arte popular, textiles tradicionales '
                'y técnicas ancestrales de pintura y cerámica.'
            )
        },
    ]
    return users


def main():

    print("=" * 80)
    print("SISTEMA DE RECOMENDACIÓN DE NOTICIAS PERSONALIZADO")
    print("=" * 80)
    
    # Inicializar componentes
    text_processor = TextPreprocessor(use_spacy=False)
    annotator = RegexAnnotator()
    
    # Intentar cargar artículos procesados desde cache
    processed_cache_file = 'processed_articles.json'
    articles = load_processed_articles(processed_cache_file)
    
    if articles is None:
        # No existe cache, procesar artículos desde cero
        print("\n📂 Cargando artículos crudos...")
        raw_data = load_raw_data()  # Cambia el limit o quítalo para cargar todos
        print(f"✅ {len(raw_data)} artículos crudos cargados")
        
        # Inicializar vectorizador de noticias
        news_vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
        
        # Preparar artículos: categorizar, limpiar y vectorizar
        articles = prepare_articles(raw_data, text_processor, annotator, news_vectorizer)
        
        # Guardar en cache para futuras ejecuciones
        save_processed_articles(articles, processed_cache_file)
    
    
    # Crear perfiles de usuarios simulados
    print("\n👥 Creando usuarios simulados...")
    simulated_users = create_simulated_users()
    
    # Inicializar componentes de recomendación
    profile_vectorizer = UserProfileVectorizer(news_vectorizer)
    profile_manager = UserProfileManager(profile_vectorizer)
    matcher = NewsMatcher()
    
    # Inicializar resumidores
    base_summarizer = TextRankSummarizer(language="spanish")
    personalized_summarizer = PersonalizedSummarizer(base_summarizer)
    
    # Inicializar generador de reportes
    report_generator = ReportGenerator(personalized_summarizer)
    
    # Procesar cada usuario
    print("\n" + "=" * 80)
    print("GENERANDO RECOMENDACIONES PERSONALIZADAS")
    print("=" * 80)
    
    all_reports = []
    
    # Crear directorio para PDFs
    pdf_output_dir = "reportes_pdf"
    os.makedirs(pdf_output_dir, exist_ok=True)
    
    for user in simulated_users:
        print(f"\n{'='*80}")
        print(f"👤 Usuario: {user['name']}")
        print(f"{'='*80}")
        print(f"📝 Perfil: {user['profile_text'][:100]}...")
        
        # Crear perfil del usuario
        user_profile = profile_manager.create_profile(user['profile_text'])
        
        print(f"\n🏷️  Categorías de interés detectadas: {user_profile['categories'][:8]}")
        print(f"📊 Dimensión del vector de perfil: {len(user_profile['vector'])}")
        
        # Encontrar artículos relevantes
        matches = matcher.match_articles(user_profile, articles, top_k=10)
        
        # Generar reporte personalizado
        report = report_generator.generate_report(matches, user_profile, max_articles=5)
        all_reports.append({
            'user_name': user['name'],
            'report': report
        })
        
        # Generar PDF
        # Crear nombre de archivo seguro
        safe_name = user['name'].replace(' ', '_').replace('-', '_').replace('/', '_')
        pdf_filename = f"{pdf_output_dir}/reporte_{safe_name}.pdf"
        
        print(f"\n📄 Generando PDF...")
        if report_generator.generate_pdf(report, pdf_filename, user['name']):
            print(f"✅ PDF guardado en: {pdf_filename}")
        else:
            print(f"⚠️  No se pudo generar el PDF (instala reportlab: pip install reportlab)")
        
        print(f"\n{'='*80}\n")
    
    # Estadísticas generales
    print("\n" + "=" * 80)
    print("📊 ESTADÍSTICAS GENERALES")
    print("=" * 80)
    
    # Categorías más comunes en artículos
    category_counts = {}
    for article in articles:
        for cat in article['categories']:
            category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n🏆 Top 10 categorías más frecuentes en artículos:")
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat, count in sorted_cats:
        print(f"   {cat}: {count} artículos")
    
    print(f"\n📁 Reportes PDF guardados en: {pdf_output_dir}/")
    print("\n✅ Sistema completado exitosamente!")


if __name__ == "__main__":
    main()
