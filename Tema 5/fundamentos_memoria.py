"""
! Este demostracion es un forma simple que no se debe usar en produccion, ya que no es eficiente y no es segura.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil."),
    ("human", "{input}")
])

chain = prompt | llm

print("Chat en terminal (escribe 'salir' para terminar)\n")

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

    # Sin memoria: cada invoke es independiente — ver MEMORIA_NOTAS.md
    respuesta = chain.invoke({"input": user_input})
    print("Asistente:", respuesta.content)