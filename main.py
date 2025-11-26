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


def create_simulated_users():
    """Crea perfiles de usuarios simulados con diferentes intereses basados en categorías regex"""
    users = [
        {
            'name': 'Ana - Activista',
            'profile_text': (
                'Me dedico a defender los derechos humanos y seguir las crisis humanitarias '
                'en Gaza, Palestina y Líbano. Denuncio el genocidio, crímenes de guerra y '
                'violaciones a la libertad de prensa. Me preocupan los periodistas asesinados, '
                'la discriminación racial, tortura y desapariciones forzadas. Sigo protestas '
                'contra represión policial y bloqueos humanitarios que afectan civiles.'
            )
        },
        {
            'name': 'Carlos - Analista Político',
            'profile_text': (
                'Analizo elecciones, campañas electorales y procesos políticos en América Latina. '
                'Sigo reformas legislativas, aprobación de leyes y decisiones judiciales. '
                'Me interesan las relaciones internacionales, ALBA, CELAC, UNASUR, cumbres '
                'presidenciales, tratados bilaterales y cooperación multilateral. Estudio el '
                'antiimperialismo, soberanía nacional y declaraciones políticas.'
            )
        },
        {
            'name': 'María - Economista',
            'profile_text': (
                'Analizo crisis económicas, inflación, desempleo, recesión y deuda externa. '
                'Sigo tensiones comerciales, aranceles, sanciones económicas y el FMI. '
                'Me interesan ajustes fiscales, privatización, poder adquisitivo y crecimiento '
                'económico en países en desarrollo. Monitoreo mercados financieros, inversión '
                'extranjera y políticas de redistribución económica.'
            )
        },
        {
            'name': 'Pedro - Ambientalista',
            'profile_text': (
                'Me dedico a conservación ambiental y sigo desastres naturales: terremotos, '
                'inundaciones, huracanes, erupciones volcánicas. Denuncio deforestación, '
                'contaminación, derrames petroleros y cambio climático. Me preocupan emergencias '
                'sanitarias, epidemias, escasez de agua. Apoyo objetivos de desarrollo sostenible '
                'y protección de biodiversidad y recursos naturales.'
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
    
    # Cargar artículos 
    print("\n📂 Cargando artículos...")
    raw_data = load_raw_data()  # Cambia el limit o quítalo para cargar todos
    print(f"✅ {len(raw_data)} artículos cargados")
    
    # Inicializar vectorizador de noticias
    news_vectorizer = NewsVectorizer(max_features=3000, ngram_range=(1, 2))
    
    # Preparar artículos: categorizar, limpiar y vectorizar
    articles = prepare_articles(raw_data, text_processor, annotator, news_vectorizer)
    
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
