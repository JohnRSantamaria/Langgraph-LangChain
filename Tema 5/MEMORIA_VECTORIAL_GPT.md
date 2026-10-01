# Memoria vectorial en un chat con GPT

GPT no consulta por sí solo una base de datos ni recuerda automáticamente todas las conversaciones. En una aplicación, la **memoria la gestiona nuestro código**: guarda información útil, recupera la que parece relacionada con la pregunta actual y se la entrega al modelo como contexto.

> **Idea clave:** GPT responde usando el texto que recibe en ese momento. La memoria vectorial es un sistema externo que decide qué recuerdos añadir a ese texto.

---

## Memoria del chat y memoria vectorial

- **Memoria del chat:** conserva los mensajes de una conversación concreta. Sirve para continuar el tema actual.
- **Memoria vectorial:** conserva hechos o fragmentos útiles a largo plazo, incluso entre conversaciones diferentes.

Por ejemplo, el historial puede contener todo lo hablado hoy. La memoria vectorial puede guardar que el usuario trabaja con Python, vive en Colombia o prefiere respuestas breves.

---

## Cómo funciona la recuperación

```mermaid
flowchart LR
    U["Pregunta actual"] --> E["Convertir su significado<br/>en una representación numérica"]
    E --> B["Buscar recuerdos<br/>con significado parecido"]
    B --> C["Añadir los recuerdos<br/>al contexto del chat"]
    C --> G["GPT genera<br/>la respuesta"]

    classDef input fill:#64748B,stroke:#334155,color:#fff
    classDef process fill:#0284C7,stroke:#075985,color:#fff
    classDef vector fill:#DB2777,stroke:#9D174D,color:#fff
    classDef output fill:#059669,stroke:#065F46,color:#fff

    class U input
    class E,C process
    class B vector
    class G output
```

1. **Se guarda un recuerdo:** por ejemplo, “El usuario está aprendiendo LangGraph”.
2. **Se representa por significado:** un modelo de *embeddings* convierte el texto en números que permiten comparar ideas.
3. **Llega una pregunta nueva:** “¿Qué debería estudiar después de dominar los grafos?”
4. **Se buscan recuerdos similares:** la base vectorial encuentra el recuerdo sobre LangGraph aunque no se repitan exactamente las mismas palabras.
5. **Se inyectan en el contexto:** la aplicación añade los mejores resultados al mensaje de sistema o al prompt.
6. **GPT responde:** el modelo usa la pregunta, el historial reciente y los recuerdos recuperados.

### Pseudocódigo de la recuperación

```text
función responder(pregunta, usuario):
    recuerdos = buscar_en_memoria_vectorial(
        texto = pregunta,
        filtro = usuario,
        cantidad = 3
    )

    contexto = "Eres un asistente útil."

    si recuerdos no está vacío:
        contexto = contexto + "Información recordada:"

        para cada recuerdo en recuerdos:
            contexto = contexto + recuerdo

    mensajes = [
        mensaje_sistema(contexto),
        historial_reciente_del_chat,
        mensaje_usuario(pregunta)
    ]

    respuesta = GPT.generar(mensajes)
    devolver respuesta
```

La función `buscar_en_memoria_vectorial` compara el **significado** de la pregunta con los recuerdos guardados. El filtro de usuario evita mezclar recuerdos pertenecientes a personas diferentes.

La búsqueda no garantiza que un recuerdo sea verdadero o útil. Por eso conviene recuperar pocos resultados, filtrar por usuario y permitir actualizar o borrar información.

---

## ¿Qué conviene guardar?

Información estable y reutilizable:

- Preferencias del usuario.
- Nombre, idioma o forma de comunicación preferida.
- Objetivos de aprendizaje o de trabajo.
- Decisiones importantes de un proyecto.
- Datos que serán útiles en conversaciones futuras.

No suele ser buena idea guardar cada mensaje. Eso introduce ruido, aumenta el coste y puede recuperar información irrelevante o desactualizada. Tampoco se deberían almacenar secretos, contraseñas ni datos sensibles sin una política clara de privacidad.

---

## ¿Cuándo vale la pena implementarla?

**Sí puede ser necesaria si:**

- El asistente debe recordar información entre chats distintos.
- Hay muchos documentos o conocimientos que no caben completos en el prompt.
- Se busca personalizar respuestas durante semanas o meses.
- Los usuarios esperan poder retomar proyectos antiguos.

**Probablemente no sea necesaria si:**

- Cada conversación empieza desde cero.
- El historial del chat es corto y suficiente.
- La aplicación solo responde preguntas aisladas.
- Un prompt fijo contiene toda la información necesaria.

> **Regla práctica:** empieza solo con historial de conversación. Añade memoria vectorial cuando exista información útil que deba sobrevivir entre sesiones o cuando el volumen de conocimiento sea demasiado grande para enviarlo completo al modelo.

---

## Aplicado al ejemplo del curso

En [`memoria_vectorial_langgraph.py`](memoria_vectorial_langgraph.py):

- `MemorySaver` mantiene los mensajes del hilo actual.
- Chroma guarda hechos reutilizables.
- `buscar_memoria()` recupera recuerdos parecidos al último mensaje.
- El nodo añade esos recuerdos a un `SystemMessage`.
- GPT recibe ese contexto y genera la respuesta.

Así, GPT no “abre” directamente Chroma: el programa realiza la búsqueda y le entrega únicamente los recuerdos seleccionados.
