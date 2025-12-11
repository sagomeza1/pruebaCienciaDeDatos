import os
from typing import List, Optional
import gradio as gr

# --- LangChain & Vector Store ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Manejo robusto de importaciones para Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURACIÓN ---
# Usamos un modelo MULTILINGÜE. Crucial para que entienda español semántico.
# 'paraphrase-multilingual-MiniLM-L12-v2' es el estándar "bueno, bonito y barato" para CPU.
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Umbral de similitud (Distancia L2 o Coseno). Si la distancia es alta, el doc no es relevante.
# En Chroma (L2), menor distancia = mayor similitud. Ajustar según pruebas.
UMBRAL_CORTE = 1.5 * 40

class MotorRAG:
    """Clase que encapsula la lógica del RAG para mantener el estado y la limpieza."""
    
    def __init__(self):
        self.vector_db = self._inicializar_base_conocimiento()

    def _inicializar_base_conocimiento(self) -> Chroma:
        """Carga, procesa y vectoriza los documentos."""
        print(f"🔄 Inicializando embeddings con: {MODELO_EMBEDDING}...")
        
        texto_base = """
Los clientes tienen 30 días calendario para solicitar devolución.
El producto debe estar en condiciones comerciales.
Garantía por defectos de fabricación durante 12 meses.
El cliente debe presentar evidencia de compra.
El uso de la plataforma implica la aceptación de términos.
Las responsabilidades y limitaciones están descritas."""

        # OPTIMIZACIÓN 1: Chunking Semántico
        # Usamos separadores específicos para no cortar oraciones legales a la mitad.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=70, 
            chunk_overlap=0,
            separators=["\n", ".", ";"] 
        )
        
        docs = [Document(page_content=x.strip()) for x in text_splitter.split_text(texto_base) if x.strip()]

        # OPTIMIZACIÓN 2: Modelo Multilingüe
        embedding_function = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)

        # Crear DB en memoria
        db = Chroma.from_documents(documents=docs, embedding=embedding_function)
        print("✅ Base de conocimientos vectorizada y lista.")
        return db

    def consultar(self, pregunta: str) -> str:
        """Realiza la búsqueda semántica y formatea la respuesta."""
        if not pregunta.strip():
            return "⚠️ Por favor, escribe una pregunta válida."

        # OPTIMIZACIÓN 3: Búsqueda con Score (Distancia)
        # k=3 para tener más contexto, pero filtraremos por calidad.
        resultados = self.vector_db.similarity_search_with_score(pregunta, k=3)

        contextos_validos = []
        
        for doc, score in resultados:
            # En Chroma (default L2), score bajo = mejor coincidencia.
            # Un score > 1.0 o 1.5 suele ser una coincidencia pobre para oraciones cortas.
            if score < UMBRAL_CORTE: 
                contextos_validos.append(f"• {doc.page_content} (Confianza: {1/score:.2f})")

        if not contextos_validos:
            return (
                "❌ **Información no encontrada.**\n\n"
                "El sistema no encontró reglas relevantes en la base de conocimiento para tu consulta. "
                "Para garantizar la consistencia, no intentaré inventar una respuesta."
            )

        # Construcción del Prompt para el usuario (o para un futuro LLM)
        texto_contexto = "\n".join(contextos_validos)
        
        respuesta = (
            f"✅ **Información Recuperada (Base de Conocimiento):**\n\n"
            f"{texto_contexto}\n\n"
            f"---\n"
            f"💡 *Respuesta sugerida basada estrictamente en lo anterior:*\n"
            f"Según las políticas: {contextos_validos[0].split('(')[0]}"
        )
        return respuesta

# Instancia global del motor
motor_rag = MotorRAG()

# --- INTERFAZ GRÁFICA ---
def interfaz_fn(pregunta):
    return motor_rag.consultar(pregunta)

tema_visual = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

with gr.Interface(
    fn=interfaz_fn,
    inputs=gr.Textbox(lines=2, placeholder="Ej: ¿Cuánto tiempo tengo para devolver algo?", label="Consulta al Manual"),
    outputs=gr.Markdown(label="Respuesta Consistente"),
    title="🧬 RAG Optimizer: Contexto Legal",
    description="Sistema de recuperación semántica optimizado para consistencia en español.",
    theme=tema_visual,
    examples=[
        ["¿Cuánto dura la garantía?"],
        ["¿Puedo devolver el producto si ya lo usé?"], # Pregunta capciosa para probar consistencia
        ["¿Cómo contacto a soporte técnico?"] # Pregunta fuera de dominio
    ]
) as demo:
    pass

if __name__ == "__main__":
    demo.launch()