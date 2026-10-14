from langchain_core.tools import tool, StructuredTool

@tool("Herramienta acceso base de datos usuarios", return_direct=True)
def herramienta_personalizada(query: str) -> str:
    """Consulta la base de datos de usuarios de la empresa."""
    # Codigo que accede a la base de datos
    return f"Respuesta a la consulta: {query}"

output = herramienta_personalizada.run("Consulta de prueba")
print(output)

print(f"Nombre de la herramienta: {herramienta_personalizada.name}")
print(f"Descripcion de la herramienta: {herramienta_personalizada.description}")

# Dice que esto es redundante y que no se usa tanto. 

def herramienta_personalizada_estructurada(query: str) -> str:
    """Consulta la base de datos de usuarios de la empresa."""
    
    # Codigo que accede a la base de datos
    return f"Respuesta a la consulta: {query}"

mi_tool = StructuredTool.from_function(herramienta_personalizada_estructurada)

print(mi_tool.run("Consulta de prueba"))

