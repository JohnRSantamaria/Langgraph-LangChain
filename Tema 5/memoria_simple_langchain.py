"""
? RunnableWithMessageHistory: es un runnable que se encarga de la memoria de la conversacion. 
LangGraph nos dice que utilicemos RunnableHistory unicamente para casos simples, que no requieren 
persistir la memoria, o que lo unico que implementen sean cadenas con el lenguaje natural.  

InMemoryChatMessageHistory: es un memory que se encarga de la memoria de la conversacion en memoria. (Almacena en memoria RAM)

! Las siguientes clases aunque siguen funcionando, estan deprecadas y se deben usar las nuevas clases de LangChain.
- ConversationBufferMemory: es un memory que se encarga de la memoria de la conversacion.
- ConversationChain: es una cadena que se encarga de la conversacion.

"""
import os 
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Prompt: no hace falta usar tuplas ("system", ...) / ("human", ...); es solo un atajo equivalente.
# Con plantillas defines cada rol con from_template (útil si luego añades variables {nombre} en el texto).
# MessagesPlaceholder inyecta el historial (variable_name debe coincidir con history_messages_key="history").
# HumanMessagePromptTemplate debe incluir {input}: es la clave que usa RunnableWithMessageHistory.
# Atajo equivalente: ("system", "..."), MessagesPlaceholder(...), ("human", "{input}")
system_template = SystemMessagePromptTemplate.from_template("Eres un asistente útil.")
human_template = HumanMessagePromptTemplate.from_template("{input}")
prompt = ChatPromptTemplate.from_messages([
    system_template,
    MessagesPlaceholder(variable_name="history"),
    human_template,
])

chain = prompt | llm

store = {}

# session_id elige qué historial usar en store — https://reference.langchain.com/python/langchain-core/runnables/history/RunnableWithMessageHistory
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Envuelve la chain: lee/escribe historial según session_id en config (ver MEMORIA_NOTAS.md)
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history" 
)
print("\n")
print("."*100)
print("Chat en terminal (escribe 'salir' para terminar)".center(100))
print("."*100)
print("\n")

session_id = "sesion_terminal"

while True:
    try:
        user_input = input("\nTú: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nHasta luego!")
        break

    if not user_input: 
        continue
    if user_input.lower() in {"salir", "exit", "quit"}:
        print("Hasta luego!")
        break

    # session_id va en config (no en input): memoria por conversación — ver MEMORIA_NOTAS.md
    respuesta = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    print("Asistente:", respuesta.content)