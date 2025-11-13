from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import logging
import os
import time
import hashlib
import glob
import shutil

from .utils import load_and_split_pdf

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Habilitar caché para los LLMs
set_llm_cache(InMemoryCache())

# Caché de respuestas para consultas idénticas
RESPONSE_CACHE = {}
# Tamaño máximo de caché para evitar consumo excesivo de memoria
MAX_CACHE_SIZE = 100

# Directorios locales que no se trackean en git
DATA_DIR = "local_data"
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")
EMBEDDING_CACHE_DIR = os.path.join(DATA_DIR, "embedding_cache")

# Parámetros de búsqueda simplificados para garantizar compatibilidad
RETRIEVER_SEARCH_KWARGS = {
    "k": 5  # Número de documentos a recuperar
}

def ingest(pdf_paths=None):
    """
    Ingiere uno o varios documentos PDF para construir la base de conocimientos.
    
    Args:
        pdf_paths: Puede ser una ruta a un archivo PDF específico, una lista de rutas, 
                  o None para procesar todos los PDFs en el directorio data/reglamento/
    
    Returns:
        str: Mensaje con el resultado de la ingestión
    """
    try:
        start_time = time.time()
        
        # Si no se especifican rutas, procesar todos los PDFs en el directorio reglamento
        if pdf_paths is None:
            pdf_paths = glob.glob("data/reglamento/*.pdf")
            if not pdf_paths:
                logger.error("No se encontraron archivos PDF en data/reglamento/")
                return "Error: No se encontraron archivos PDF para ingestar"
        
        # Si es una sola ruta, convertirla a lista
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        
        # Verificar que los archivos existan
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                logger.error(f"El archivo {pdf_path} no existe")
                return f"Error: El archivo {pdf_path} no existe"
        
        # Asegurar que los directorios existan
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
        
        # Reiniciar la base de datos vectorial para asegurar una ingestión limpia
        if os.path.exists(PERSIST_DIR):
            logger.info(f"Eliminando base de datos vectorial existente: {PERSIST_DIR}")
            try:
                shutil.rmtree(PERSIST_DIR)
                logger.info("Base de datos vectorial eliminada correctamente")
            except Exception as e:
                logger.error(f"Error al eliminar base de datos: {str(e)}")
                # Continuar con la operación
                
        # Crear el directorio vacío
        os.makedirs(PERSIST_DIR, exist_ok=True)
        
        # Procesar cada PDF y recolectar todos los chunks
        all_chunks = []
        total_chunks = 0
        
        for pdf_path in pdf_paths:
            logger.info(f"Procesando documento: {pdf_path}")
            chunks = load_and_split_pdf(pdf_path)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
            logger.info(f"Documento {pdf_path} dividido en {len(chunks)} chunks")
        
        logger.info(f"Total de documentos procesados: {len(pdf_paths)}")
        logger.info(f"Total de chunks: {total_chunks}")
        
        # Configurar embeddings
        embedding = FastEmbedEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir=EMBEDDING_CACHE_DIR
        )
        
        # Crear la base de vectores
        Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Limpiar caché para reflejar nueva base de conocimiento
        RESPONSE_CACHE.clear()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Ingestión completada en {elapsed_time:.2f} segundos")
        
        return f"Ingestión completada en {elapsed_time:.2f} segundos. Se procesaron {total_chunks} fragmentos de {len(pdf_paths)} documentos."
    except Exception as e:
        logger.error(f"Error durante la ingestión: {str(e)}", exc_info=True)
        return f"Error durante la ingestión: {str(e)}"

def get_query_hash(query):
    """Genera un hash único para la consulta para usar como clave de caché"""
    return hashlib.md5(query.lower().strip().encode('utf-8')).hexdigest()

def ask(query: str):
    """
    Responde preguntas usando la base de conocimiento.
    """
    try:
        logger.info(f"🔍 Procesando consulta: {query}")
        
        # Verificar caché
        query_hash = get_query_hash(query)
        if query_hash in RESPONSE_CACHE:
            logger.info("✅ Respuesta encontrada en caché")
            return RESPONSE_CACHE[query_hash]
        
        # Verificar que exista la base de datos
        if not os.path.exists(PERSIST_DIR):
            logger.error("❌ Directorio de persistencia no existe")
            return "❌ Base de conocimiento no encontrada. Por favor, ejecuta primero la ingestión de documentos."
        
        logger.info("📥 Configurando embeddings...")
        # Configurar embeddings y vectorstore
        embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir=EMBEDDING_CACHE_DIR
        )
        
        logger.info("📥 Cargando vectorstore...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        
        # Verificar que hay documentos
        logger.info("🔍 Verificando documentos en vectorstore...")
        try:
            test_docs = vectorstore.similarity_search("universidad", k=1)
            logger.info(f"📄 Documentos encontrados en prueba: {len(test_docs)}")
            if not test_docs:
                return "❌ La base de conocimiento está vacía. Por favor, reingesta los documentos."
        except Exception as e:
            logger.error(f"❌ Error en similarity_search: {str(e)}")
            return f"❌ Error accediendo a la base de conocimiento: {str(e)}"
        
        # Configurar retriever
        logger.info("🔧 Configurando retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Probamos el retriever
        logger.info("🔍 Probando retriever...")
        retrieved_docs = retriever.get_relevant_documents(query)
        logger.info(f"📄 Documentos recuperados: {len(retrieved_docs)}")
        
        if not retrieved_docs:
            return "❌ No se encontró información relevante para tu pregunta en los documentos."
        
        # Prompt
        prompt = ChatPromptTemplate.from_template("""
        Eres Ardy, un asistente de la Universidad de Ibagué. Responde amablemente 
        basándote SOLO en la información proporcionada:

        {context}

        Pregunta: {input}

        Respuesta (sé claro y conciso):
        """)
        
        # Modelo
        logger.info("🤖 Configurando modelo Ollama...")
        model = ChatOllama(
            model="llama3",
            temperature=0.1,
            num_predict=512,
        )
        
        # Crear cadena
        logger.info("⛓️ Creando cadenas...")
        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Ejecutar
        logger.info("🚀 Invocando cadena...")
        response = retrieval_chain.invoke({"input": query})
        
        logger.info(f"📝 Respuesta recibida: {len(response)} elementos")
        logger.info(f"📝 Keys en respuesta: {list(response.keys())}")
        
        answer = response.get("answer", "No se pudo generar respuesta")
        
        # Guardar en caché
        RESPONSE_CACHE[query_hash] = answer
        
        logger.info("✅ Consulta procesada exitosamente")
        return answer
        
    except Exception as e:
        error_msg = f"❌ Error procesando consulta: {str(e)}"
        logger.error(error_msg, exc_info=True)
        import traceback
        logger.error(traceback.format_exc())
        return f"Lo siento, ha ocurrido un error: {str(e)}"