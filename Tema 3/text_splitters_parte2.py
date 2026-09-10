from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Cargar el documento PDF
loader = PyPDFLoader("C:\\Users\\santiago\\curso_langchain\\Tema 3\\quijote.pdf")
pages = loader.load()

# Dividir el texto en chunks mas pequeños
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(pages)

# 3. Pasar el texto al LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
summaries = []

i = 0
for chunk in chunks:
    if i > 10:
        break
    response = llm.invoke(f"Haz un resumen de los puntos mas importantes del siguiente texto: {chunk.page_content}")
    summaries.append(response.content)
    i += 1

print(summaries)

final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y completo: {" ".join(summaries)}")
print(final_summary.content)