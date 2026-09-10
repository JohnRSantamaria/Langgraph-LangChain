from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario con su nombre.\nNombre del usuario: {nombre}\nAsistente:"
)

chain = LLMChain(llm=chat, prompt=plantilla)

resultado = chain.run(nombre="Carlos")
print(resultado)