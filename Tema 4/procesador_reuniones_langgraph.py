from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import os
from tkinter import Tk, filedialog
import openai

# Configuración
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Definición del Estado
class State(TypedDict):
    notes: str
    participants: List[str]
    topics: List[str]
    action_items: List[str]
    minutes: str
    summary: str

# ============= NODOS DEL WORKFLOW =============

def extract_participants(state: State) -> State:
    """Extrae los participantes de la reunión."""
    prompt = f"""
    De las siguientes notas de reunión, extrae SOLO los nombres de los participantes.
    
    Notas: {state['notes']}
    
    Responde ÚNICAMENTE con una lista de nombres separados por comas, sin explicaciones adicionales.
    Ejemplo: Juan García, María López, Carlos Ruiz
    """
    
    response = llm.invoke(prompt)
    participants = [p.strip() for p in response.content.split(',') if p.strip()]
    
    print(f"✓ Participantes extraídos: {len(participants)} personas")
    
    return {
        'participants': participants
    }

def identify_topics(state: State) -> State:
    """Identifica los temas principales discutidos."""
    prompt = f"""
    Identifica los 3-5 temas principales discutidos en esta reunión.
    
    Notas: {state['notes']}
    
    Responde SOLO con los temas separados por punto y coma (;).
    Ejemplo: Arquitectura del sistema; Plazos de entrega; Asignación de tareas
    """
    
    response = llm.invoke(prompt)
    topics = [t.strip() for t in response.content.split(';') if t.strip()]
    
    print(f"✓ Temas identificados: {len(topics)} temas")
    
    return {
        'topics': topics
    }

def extract_actions(state: State) -> State:
    """Extrae las acciones acordadas y sus responsables."""
    prompt = f"""
    Extrae las acciones específicas acordadas en la reunión, incluyendo el responsable si se menciona.
    
    Notas: {state['notes']}
    
    Formato de respuesta: Una acción por línea, separadas por |
    Ejemplo: María se encargará del backend | Carlos preparará el plan de testing | Próxima reunión el lunes
    
    Si no hay acciones claras, responde con: "No se identificaron acciones específicas"
    """
    
    response = llm.invoke(prompt)
    
    if "No se identificaron" in response.content:
        action_items = []
    else:
        action_items = [a.strip() for a in response.content.split('|') if a.strip()]
    
    print(f"✓ Acciones extraídas: {len(action_items)} items")
    
    return {
        'action_items': action_items
    }

def generate_minutes(state: State) -> State:
    """Genera una minuta formal de la reunión."""
    participants_str = ", ".join(state['participants'])
    topics_str = "\n• ".join(state['topics'])
    actions_str = "\n• ".join(state['action_items']) if state['action_items'] else "No se definieron acciones específicas"
    
    prompt = f"""
    Genera una minuta formal y profesional basándote en la siguiente información:
    
    PARTICIPANTES: {participants_str}
    
    TEMAS DISCUTIDOS:
    • {topics_str}
    
    ACCIONES ACORDADAS:
    • {actions_str}
    
    NOTAS ORIGINALES: {state['notes']}
    
    Genera una minuta profesional de máximo 150 palabras que incluya:
    1. Encabezado con tipo de reunión
    2. Lista de asistentes
    3. Puntos principales discutidos
    4. Acuerdos y próximos pasos
    
    Usa un tono formal y estructura clara.
    """
    
    response = llm.invoke(prompt)
    
    print(f"✓ Minuta generada: {len(response.content.split())} palabras")
    
    return {
        'minutes': response.content
    }

def create_summary(state: State) -> State:
    """Crea un resumen ejecutivo ultra-breve."""
    prompt = f"""
    Crea un resumen ejecutivo de MÁXIMO 2 líneas (30 palabras) que capture la esencia de esta reunión.
    
    Participantes: {', '.join(state['participants'][:3])}{'...' if len(state['participants']) > 3 else ''}
    Tema principal: {state['topics'][0] if state['topics'] else 'General'}
    Acciones clave: {len(state['action_items'])} acciones definidas
    
    El resumen debe ser conciso y directo al punto.
    """
    
    response = llm.invoke(prompt)
    
    print(f"✓ Resumen creado")
    
    return {
        'summary': response.content
    }

# ============= CONSTRUCCIÓN DEL GRAFO =============

def create_workflow():
    """Crea y configura el workflow de LangGraph."""
    workflow = StateGraph(State)
    
    # Agregar todos los nodos
    workflow.add_node("extract_participants", extract_participants)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("extract_actions", extract_actions)
    workflow.add_node("generate_minutes", generate_minutes)
    workflow.add_node("create_summary", create_summary)
    
    # Configurar flujo secuencial
    workflow.add_edge(START, "extract_participants")
    workflow.add_edge("extract_participants", "identify_topics")
    workflow.add_edge("identify_topics", "extract_actions")
    workflow.add_edge("extract_actions", "generate_minutes")
    workflow.add_edge("generate_minutes", "create_summary")
    workflow.add_edge("create_summary", END)
    
    # LangGraph: pipeline lineal, sin thread_id ni checkpointer (ver LANGGRAPH_NOTAS.md#grafos-sin-persistencia)
    return workflow.compile()

# ============= FUNCIONES DE PROCESAMIENTO =============

def transcribe_media_direct(file_path: str) -> str:
    """Transcribe usando directamente la API de OpenAI Whisper."""
    try:
        print("🎙️ Transcribiendo con OpenAI Whisper API directa...")
        
        client = openai.OpenAI()  # Usa OPENAI_API_KEY del entorno
        
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es",  # Español
                prompt="Esta es una reunión de trabajo en español con múltiples participantes.",
                response_format="text"
            )
        
        print(f"✓ Transcripción completada: {len(transcript)} caracteres")
        return transcript
        
    except Exception as e:
        print(f"❌ Error en transcripción: {e}")
        return f"Error: {str(e)}"

def process_meeting_notes(notes: str, app):
    """Procesa una nota de reunión individual."""
    initial_state = {
        'notes': notes,
        'participants': [],
        'topics': [],
        'action_items': [],
        'minutes': '',
        'summary': ''
    }
    
    print("\n" + "="*60)
    print("🔄 Procesando nota de reunión...")
    print("="*60)
    
    result = app.invoke(initial_state)
    return result

def display_results(result: State, meeting_num: int):
    """Muestra los resultados de forma estructurada."""
    print(f"\n📋 RESULTADOS - REUNIÓN #{meeting_num}")
    print("-"*60)
    
    print(f"\n👥 Participantes ({len(result['participants'])}):")
    for p in result['participants']:
        print(f"   • {p}")
    
    print(f"\n📍 Temas tratados ({len(result['topics'])}):")
    for t in result['topics']:
        print(f"   • {t}")
    
    print(f"\n✅ Acciones acordadas ({len(result['action_items'])}):")
    if result['action_items']:
        for a in result['action_items']:
            print(f"   • {a}")
    else:
        print("   • No se definieron acciones específicas")
    
    print(f"\n📄 MINUTA FORMAL:")
    print("-"*40)
    print(result['minutes'])
    print("-"*40)
    
    print(f"\n💡 RESUMEN EJECUTIVO:")
    print(f"   {result['summary']}")
    
    print("\n" + "="*60)

# ============= DEMOSTRACIÓN =============

if __name__ == "__main__":
    app = create_workflow()

    # Pequeña interfaz gráfica: selector de archivo
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecciona un vídeo o transcripción",
        filetypes=[
            ("Vídeo/Audio", "*.mp4 *.mov *.m4a *.mp3 *.wav *.mkv *.webm"),
            ("Texto", "*.txt *.md")
        ]
    )

    if not file_path:
        print("No se seleccionó archivo.")
        raise SystemExit(0)

    ext = os.path.splitext(file_path)[1].lower()
    media_exts = {".mp4", ".mov", ".m4a", ".mp3", ".wav", ".mkv", ".webm"}

    if ext in media_exts:
        notes = transcribe_media_direct(file_path) 
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            notes = f.read()

    result = process_meeting_notes(notes, app)
    display_results(result, 1)