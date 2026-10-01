# Igual que memoria_persistente_langgraph, pero PostgresSaver persiste en PostgreSQL.
# Ver Tema 5/MEMORIA_NOTAS.md · pip install langgraph-checkpoint-postgres "psycopg[binary,pool]"
import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from psycopg import Connection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

workflow = StateGraph(state_schema=MessagesState)

def chatbot_node(state):
    """Nodo que procesa mensajes y genera respuestas."""
    system_prompt = "Eres un asistente amigable que recuerda conversaciones previas."
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")

# PostgresSaver: checkpoints sobreviven al cerrar el script.
# Doc: https://docs.langchain.com/oss/python/langgraph/add-memory
# autocommit + dict_row son obligatorios si pasas la conexión a mano.
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
conn = Connection.connect(DB_URI, autocommit=True, row_factory=dict_row)
memory = PostgresSaver(conn)
memory.setup()
app = workflow.compile(checkpointer=memory)

def chat(message, thread_id="sesion_terminal"):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("Chat en terminal (escribe 'salir' para terminar)\n")
    session_id = "sesion_terminal2"

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego!")
            break

        respuesta = chat(user_input, session_id)
        print("Asistente:", respuesta)
