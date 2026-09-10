from langchain_community.document_loaders import GoogleDriveLoader

credentials_path = "C:\\Users\\santiago\\curso_langchain\\Tema 3\\credentials.json"
token_path = "C:\\Users\\santiago\\curso_langchain\\Tema 3\\token.json"

loader = GoogleDriveLoader(
    folder_id="17DDwGPRjhjZhR6NRpegkFEmA4Wo6M_l3",
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True
)

documents = loader.load()

print(f"Metadatos: {documents[0].metadata}")
print(f"Contenido: {documents[0].page_content}")