# ingest.py
import os
import sys
import logging
from pathlib import Path

# Agregar el directorio actual al path de Python
sys.path.insert(0, str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=== INGESTIÓN DE DOCUMENTOS ===")
    
    # Verificar PDFs
    pdf_dir = Path("data/reglamento")
    if not pdf_dir.exists():
        print("❌ La carpeta 'data/reglamento' no existe")
        print("Creando estructura de directorios...")
        os.makedirs("data/reglamento", exist_ok=True)
        print("✅ Directorio 'data/reglamento' creado")
        print("Por favor, coloca tus archivos PDF en 'data/reglamento/' y ejecuta de nuevo")
        return
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No hay archivos PDF en 'data/reglamento/'")
        print("Archivos encontrados en data/reglamento/:")
        for item in pdf_dir.iterdir():
            print(f"   - {item.name}")
        return
    
    print(f"📁 PDFs encontrados: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"   - {pdf.name} ({pdf.stat().st_size} bytes)")
    
    try:
        # Importar desde la carpeta chatbot
        from chatbot.rag_engine import ingest
        
        print("\n🔄 Iniciando ingestión...")
        result = ingest([str(pdf) for pdf in pdf_files])
        
        print(f"\n{result}")
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("Verificando estructura de archivos...")
        
        # Verificar qué archivos existen en chatbot/
        chatbot_dir = Path("chatbot")
        if chatbot_dir.exists():
            print("Archivos en carpeta chatbot/:")
            for item in chatbot_dir.iterdir():
                print(f"   - {item.name}")
        else:
            print("❌ La carpeta 'chatbot' no existe")
            
    except Exception as e:
        print(f"❌ Error durante la ingestión: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()