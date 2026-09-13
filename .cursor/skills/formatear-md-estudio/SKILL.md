---
name: formatear-md-estudio
description: >-
  Formats course Markdown into a study notebook: headings, --- separators,
  fenced code, relative links, study notes, and colored Mermaid diagrams.
  Never deletes original text or code. Use when the user asks to formatear
  un md, formatear markdown, limpiar un .md, convertirlo en cuaderno de
  estudio, notes del curso, h.md, "como los demás .md", add Mermaid, or
  apply the color palette to diagrams.
---

# Formatear Markdown de estudio

Cuadernos de este repo (Temas 1–5). Conservar el contenido original; añadir estructura, notas y diagramas.

## Workflow

1. Leer el `.md` objetivo **y 1–2 `.md` vecinos del mismo tema** (mismo `Tema N/`) como referencia de estilo.
2. Formatear **sin borrar** texto ni código originales. Sí se puede titular, envolver en Markdown y reordenar secciones.
3. Añadir explicación, Mermaid, tablas o notas **solo si ayudan** a estudiar.
4. Plantilla Mermaid y snippet de nota: [reference.md](reference.md).

Referencias de estilo típicas:

- [`Tema 4/Componentes fundamentales de LangGraph.md`](../../../Tema 4/Componentes fundamentales de LangGraph.md)
- [`Tema 5/Tipos de memoria principales.md`](../../../Tema 5/Tipos de memoria principales.md)
- [`Tema 5/MEMORIA_NOTAS.md`](../../../Tema 5/MEMORIA_NOTAS.md)

## Conservar contenido

- No eliminar párrafos, listas ni bloques de código del original.
- Código que venía como texto plano → bloque con lenguaje (` ```python `).
- Restaurar títulos de sección si el original los tenía (p. ej. “Estrategias de Gestión de Memoria”).
- Preservar **Cuándo usar**, conclusiones y avisos del curso (p. ej. “en los próximos videos”).

## Estructura Markdown

- Título `# Lectura: …` **o** el patrón del archivo vecino (`# Notas …`).
- Secciones `##` / `###`; separadores `---` entre bloques principales.
- Listas con **negrita** en la etiqueta: `- **Limitaciones de contexto:** …`
- `**Cuándo usar:**` en negrita, no como heading extra.
- Enlaces relativos a scripts y notas del mismo tema.
- Índice o tabla de mapa si el artículo es largo.
- Código siempre en fences con lenguaje.

## Añadir si aporta

- `> **Para aclarar:**` — analogía o distingo conceptual.
- `> **Nota de implementación:**` — cómo encaja en LangGraph / scripts del repo.
- Tablas comparativas, checklist de práctica, enlaces oficiales.
- Leyenda de colores bajo el índice si hay varios diagramas.

## Paleta Mermaid (obligatoria)

Todo diagrama usa `classDef` + `class` y `color:#fff`. No dejar nodos sin clase.

| Rol | fill | stroke |
| --- | --- | --- |
| Estado / memoria persistida | `#4F46E5` | `#312E81` |
| Procesamiento / nodos | `#0284C7` | `#075985` |
| Salida / LLM / START-END | `#059669` | `#065F46` |
| Decisión / routing | `#D97706` | `#92400E` |
| Entrada / restricciones | `#64748B` | `#334155` |
| Vectorial / semántica | `#DB2777` | `#9D174D` |

Índigo = checkpoint / system / resumen. Azul = nodos de trabajo. Verde = LLM y extremos del grafo. Naranja = `should_*` y rombos. Gris = límites o entrada cruda. Rosa = Chroma / retrieval.

## Fuera de alcance

No reescribir el artículo desde cero. No inventar código que no estuviera (salvo snippets mínimos en notas, claramente marcados). No tocar otros `.md` salvo que el usuario lo pida.
