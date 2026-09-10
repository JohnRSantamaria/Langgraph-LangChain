import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


load_dotenv()   

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7 , api_key=os.getenv("OPENAI_API_KEY"))

pregunta = "¿En qué año llegó el ser humano a la Luna por primera vez?"
print("Pregunta: ", pregunta)

respueta = llm.invoke(pregunta)
print("Respuesta del modelo: ", respueta.content)