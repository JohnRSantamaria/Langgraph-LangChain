# Lectura: MessagesPlaceholder y Few-Shot Examples

## ¿Qué es el In-Context Learning (ICL)?

El In-Context Learning o Aprendizaje en Contexto es la capacidad de los modelos de lenguaje de aprender nuevas tareas o patrones simplemente proporcionándoles ejemplos dentro del mismo prompt, sin necesidad de entrenar o ajustar el modelo.

Es como mostrarle a alguien cómo hacer algo dándole ejemplos directos:

> "Mira, así se hace esto... y esto otro... ¿ahora puedes hacer tú algo similar?"

---

## ¿Qué son los Few-Shot Examples?

Los Few-Shot Examples (ejemplos de pocos intentos) son una técnica de ICL donde proporcionamos al modelo entre 2-10 ejemplos de la tarea que queremos que realice. Estos ejemplos sirven como "entrenamiento instantáneo".

**Estructura típica:**

```mermaid
flowchart LR
    SYS["Sistema:<br/>Instrucciones generales"]
    E1["Ejemplo 1:<br/>Input → Output esperado"]
    E2["Ejemplo 2:<br/>Input → Output esperado"]
    E3["Ejemplo 3:<br/>Input → Output esperado"]
    Q["Pregunta real:<br/>Input actual → ?"]

    SYS --> E1 --> E2 --> E3 --> Q

    classDef sys fill:#4F46E5,stroke:#312E81,color:#fff
    classDef ex fill:#0284C7,stroke:#075985,color:#fff
    classDef q fill:#059669,stroke:#065F46,color:#fff

    class SYS sys
    class E1,E2,E3 ex
    class Q q
```

- Sistema: Instrucciones generales
- Ejemplo 1: Input → Output esperado
- Ejemplo 2: Input → Output esperado
- Ejemplo 3: Input → Output esperado
- Pregunta real: Input actual → ?

---

## MessagesPlaceholder: La Herramienta Perfecta

`MessagesPlaceholder` es ideal para few-shot examples porque:

- ✅ Mantiene la estructura de mensajes (Human/AI)
- ✅ Permite insertar múltiples ejemplos dinámicamente
- ✅ El modelo entiende mejor el formato de conversación
- ✅ Fácil de reutilizar para diferentes tareas

---

## Evolución del Código: De Historial a Few-Shot

### Código Base (Historial de Conversación)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que mantiene el contexto de la conversación."),
    MessagesPlaceholder(variable_name="historial"),
    ("human", "Usuario: {pregunta_actual}")
])

# Simulamos un historial de conversación
historial_conversacion = [
    HumanMessage(content="Usuario: ¿Cuál es la capital de Francia?"),
    AIMessage(content="IA: La capital de Francia es París."),
    HumanMessage(content="Usuario: ¿Y cuántos habitantes tiene?"),
    AIMessage(content="IA: París tiene aproximadamente 2.2 millones de habitantes en la ciudad propiamente dicha.")
]
```

### Código Evolucionado (Few-Shot Examples)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template para clasificación de sentimientos con few-shot examples
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO."),
    MessagesPlaceholder(variable_name="ejemplos"),
    ("human", "Texto a analizar: {texto_usuario}")
])

# Few-shot examples para análisis de sentimientos
ejemplos_sentimientos = [
    HumanMessage(content="Texto a analizar: Me encanta este producto, es increíble"),
    AIMessage(content="POSITIVO"),
    HumanMessage(content="Texto a analizar: El servicio fue terrible, muy decepcionante"),
    AIMessage(content="NEGATIVO"),
    HumanMessage(content="Texto a analizar: El clima está nublado hoy"),
    AIMessage(content="NEUTRO")
]

# Generar el prompt con los ejemplos
mensajes = chat_prompt.format_messages(
    ejemplos=ejemplos_sentimientos,
    texto_usuario="¡Qué día tan maravilloso!"
)

# Ver el resultado
for i, m in enumerate(mensajes):
    print(f"Mensaje {i+1} ({m.__class__.__name__}):")
    print(m.content)
    print("-" * 40)
```

### Salida del Código:

```
Mensaje 1 (SystemMessage):
Eres un experto en análisis de sentimientos. Clasifica cada texto como: POSITIVO, NEGATIVO o NEUTRO.
----------------------------------------
Mensaje 2 (HumanMessage):
Texto a analizar: Me encanta este producto, es increíble
----------------------------------------
Mensaje 3 (AIMessage):
POSITIVO
----------------------------------------
Mensaje 4 (HumanMessage):
Texto a analizar: El servicio fue terrible, muy decepcionante
----------------------------------------
Mensaje 5 (AIMessage):
NEGATIVO
----------------------------------------
Mensaje 6 (HumanMessage):
Texto a analizar: El clima está nublado hoy
----------------------------------------
Mensaje 7 (AIMessage):
NEUTRO
----------------------------------------
Mensaje 8 (HumanMessage):
Texto a analizar: ¡Qué día tan maravilloso!
----------------------------------------
```

---

## Ventajas de Este Enfoque

**Con MessagesPlaceholder:**

- ✅ Estructura clara: Cada ejemplo mantiene su rol (Human/AI)
- ✅ Escalable: Fácil añadir/quitar ejemplos
- ✅ Reutilizable: Cambiar ejemplos = Nueva tarea
- ✅ Natural: El modelo entiende el formato conversacional

**Sin MessagesPlaceholder (texto plano):**

- ❌ Menos claro: Todo mezclado en un string
- ❌ Más errores: El modelo puede confundirse
- ❌ Difícil mantenimiento: Cambios requieren reescribir todo
- ❌ Menos efectivo: Pierde la estructura conversacional

---

## Otros Ejemplos de Uso

### 1. Extracción de Información

```python
ejemplos_extraccion = [
    HumanMessage(content="Texto: Juan Pérez trabaja en Google como ingeniero desde 2020"),
    AIMessage(content="Nombre: Juan Pérez, Empresa: Google, Puesto: ingeniero, Año: 2020"),
    HumanMessage(content="Texto: María Silva es doctora en el Hospital Central"),
    AIMessage(content="Nombre: María Silva, Empresa: Hospital Central, Puesto: doctora, Año: N/A")
]
```

### 2. Traducción con Estilo

```python
ejemplos_traduccion = [
    HumanMessage(content="Formal en inglés: Good morning, how are you today?"),
    AIMessage(content="Casual en español: ¡Hola! ¿Qué tal?"),
    HumanMessage(content="Formal en inglés: I would like to schedule a meeting"),
    AIMessage(content="Casual en español: ¿Podemos quedar?")
]
```

---

## Mejores Prácticas

- **2-5 ejemplos:** Suficiente para mostrar el patrón, no demasiados
- **Ejemplos diversos:** Cubrir diferentes casos/variaciones
- **Formato consistente:** Mismo patrón en todos los ejemplos
- **Ejemplos de calidad:** Outputs perfectos como referencia

---

## Conclusión

MessagesPlaceholder transforma los few-shot examples de una técnica básica a una herramienta poderosa y flexible. La estructura clara de mensajes Human/AI hace que el modelo comprenda mejor qué esperamos, resultando en respuestas más precisas y consistentes.

Recuerda: No es solo insertar ejemplos, es enseñarle al modelo a través de la conversación. ¡Esa es la magia del In-Context Learning! ✨
