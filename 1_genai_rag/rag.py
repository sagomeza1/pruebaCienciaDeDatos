import os
import gradio as gr

# --- 1. IMPORTACIONES MODERNAS (LangChain v0.2 + Integraciones Oficiales) ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# CAMBIO CRÍTICO 1: Usamos la librería dedicada para eliminar el Warning
# Si esto falla, asegúrate de haber ejecutado: pip install langchain-huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback por si no se instaló la nueva librería
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuración e inicialización
def inicializar_base_conocimiento():
    print("Inicializando base de conocimientos...")
    
    texto_base = """
Los clientes tienen 30 días calendario para solicitar devolución.
El producto debe estar en condiciones comerciales.
Garantía por defectos de fabricación durante 12 meses.
El cliente debe presentar evidencia de compra.
El uso de la plataforma implica la aceptación de términos.
Las responsabilidades y limitaciones están descritas."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=x) for x in text_splitter.split_text(texto_base)]

    # Modelo estándar
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(documents=docs, embedding=embedding_function)
    
    print("Base de conocimientos lista.")
    return db

# Inicialización Global
vector_db = inicializar_base_conocimiento()

# RAG
def consultar_sistema_rag(pregunta):
    if not pregunta:
        return "Por favor, escribe una pregunta."

    docs_recuperados = vector_db.similarity_search(pregunta, k=2)
    
    contexto_texto = "\n\n".join([f"- {doc.page_content}" for doc in docs_recuperados])
    
    respuesta_final = (
        f"🤖 **Contexto recuperado:**\n\n"
        f"{contexto_texto}\n\n"
        f"--- \n"
        f"ℹ️ *Nota técnica: Fragmentos recuperados por similitud vectorial.*"
    )
    return respuesta_final

# INTERFAZ GRÁFICA
tema_visual = gr.themes.Soft()

interfaz = gr.Interface(
    fn=consultar_sistema_rag,
    inputs=gr.Textbox(lines=2, placeholder="Ej: ¿Qué implica el uso de la plataforma?", label="Tu Pregunta"),
    outputs=gr.Markdown(label="Respuesta del Sistema"),
    title="🔬 Demo RAG: Proyecto Omega",
    description="Interfaz de prueba para recuperación de información semántica.",
    theme=tema_visual,  # Pasamos el objeto tema aquí
    examples=[
        ["¿Cuál es la duración de la garantía?"],
        ["¿Cuál es el tiempo para solicitar una devolución?"]
    ]
)

if __name__ == "__main__":
    interfaz.launch()