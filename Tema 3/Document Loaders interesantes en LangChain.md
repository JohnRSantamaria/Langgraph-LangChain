# Lectura: Document Loaders interesantes en LangChain

Los Document Loaders son uno de los componentes más versátiles y útiles del ecosistema LangChain. Permiten convertir prácticamente cualquier fuente de información en documentos estructurados que pueden ser procesados por modelos de lenguaje. En este artículo exploraremos los loaders más interesantes y útiles, con ejemplos prácticos para cada uno.

> **Para aclarar:** el loader **no llama al LLM**. Solo trae datos de fuera (web, PDF, CSV…) y los deja en un formato común: una lista de objetos `Document`. Después puedes trocearlos, indexarlos o meterlos en un prompt.

```mermaid
flowchart LR
    S["Fuente<br/>web · PDF · CSV · Git…"]
    L["Document Loader"]
    D["Lista de Document"]
    U["Split · embed · RAG · LLM"]

    S --> L --> D --> U

    classDef s fill:#64748B,stroke:#334155,color:#fff
    classDef l fill:#4F46E5,stroke:#312E81,color:#fff
    classDef d fill:#059669,stroke:#065F46,color:#fff
    classDef u fill:#0284C7,stroke:#075985,color:#fff

    class S s
    class L l
    class D d
    class U u
```

---

## ¿Qué son los Document Loaders?

Los Document Loaders son clases especializadas que:

- Extraen contenido de diferentes fuentes (web, archivos, APIs, etc.)
- Convierten el contenido en objetos Document estandarizados
- Preservan metadatos importantes sobre la fuente original
- Facilitan el preprocesamiento y la indexación de datos

Cada documento cargado contiene:

- `page_content`: El texto extraído
- `metadata`: Información sobre la fuente (URL, fecha, tipo de archivo, etc.)

> **Para aclarar el objeto `Document`:** da igual si vino de YouTube o de un CSV. Siempre es el mismo tipo:

```python
Document(
    page_content="el texto...",
    metadata={"source": "...", "page": 1}  # las claves cambian según el loader
)
```

`loader.load()` devuelve `list[Document]`. El LLM nunca ve el PDF ni el HTML crudo: ve `page_content`.

Casi todos estos loaders están en `langchain_community`. Hay que instalarlo (y a veces una dependencia extra: `beautifulsoup4`, `pypdf`, `selenium`, etc.).

```mermaid
flowchart TB
    subgraph WEB["Web"]
        W["WebBaseLoader"]
        SE["SeleniumURLLoader"]
        H["UnstructuredHTMLLoader"]
    end
    subgraph FILES["Archivos"]
        PDF["PyPDFLoader"]
        DIR["DirectoryLoader"]
        CSV["CSVLoader"]
    end
    subgraph OTROS["Otras fuentes"]
        YT["YoutubeLoader"]
        GIT["GitLoader"]
    end

    classDef web fill:#0284C7,stroke:#075985,color:#fff
    classDef files fill:#059669,stroke:#065F46,color:#fff
    classDef otros fill:#D97706,stroke:#92400E,color:#fff

    class W,SE,H web
    class PDF,DIR,CSV files
    class YT,GIT otros
```

---

## 1. WebBaseLoader - El Poder de la Web

El WebBaseLoader es perfecto para extraer contenido de páginas web de forma sencilla.

```python
from langchain_community.document_loaders import WebBaseLoader

# Ejemplo básico: cargar documentación
loader = WebBaseLoader("https://docs.langchain.com/docs/")
docs = loader.load()

print(f"Título: {docs[0].metadata.get('title', 'Sin título')}")
print(f"URL: {docs[0].metadata['source']}")
print(f"Contenido: {docs[0].page_content[:300]}...")

# Ejemplo avanzado: múltiples URLs con configuración personalizada
urls = [
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/how_to/"
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div", {"class": ["main-content", "article-content"]}
        )
    )
)
docs = loader.load()

print(f"Páginas cargadas: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"Página {i+1}: {doc.metadata['source']}")
    print(f"Longitud: {len(doc.page_content)} caracteres")
```

**Casos de uso ideales:**

- Documentación técnica
- Artículos de blog
- Noticias y contenido web estático
- Scraping de contenido público

> **Para aclarar:** `WebBaseLoader` descarga el HTML y lo parsea con BeautifulSoup. **No ejecuta JavaScript.** Si la página es una SPA y el contenido aparece después en el navegador, aquí saldrá vacío o casi; para eso está `SeleniumURLLoader` (sección 7).
>
> `SoupStrainer` no “limpia magia”: solo extrae los `div` con esas clases, así evitas menús, pies de página y ruido.
>
> Extra: `pip install langchain-community beautifulsoup4`

---

## 2. PyPDFLoader - Documentos PDF Inteligentes

Los PDFs contienen información valiosa que este loader puede extraer página por página.

```python
from langchain_community.document_loaders import PyPDFLoader

# Cargar PDF con información detallada por página
loader = PyPDFLoader("manual_usuario.pdf")
pages = loader.load_and_split()

print(f"Total de páginas: {len(pages)}")

# Analizar contenido por página
for i, page in enumerate(pages[:3]):  # Primeras 3 páginas
    print(f"\n=== PÁGINA {i+1} ===")
    print(f"Número de página: {page.metadata['page']}")
    print(f"Archivo fuente: {page.metadata['source']}")
    print(f"Contenido: {page.page_content[:200]}...")

    # Estadísticas de la página
    words = len(page.page_content.split())
    chars = len(page.page_content)
    print(f"Palabras: {words}, Caracteres: {chars}")

# Buscar páginas con contenido específico
keyword = "configuración"
relevant_pages = [
    page for page in pages
    if keyword.lower() in page.page_content.lower()
]
print(f"\nPáginas que mencionan '{keyword}': {len(relevant_pages)}")
```

**Casos de uso ideales:**

- Manuales técnicos
- Contratos y documentos legales
- Papers académicos
- Reportes empresariales

> **Para aclarar:** `load()` vs `load_and_split()`:
>
> | Método | Resultado típico |
> | --- | --- |
> | `load()` | Un `Document` por página (o el PDF entero, según el loader) |
> | `load_and_split()` | Trocea además el texto (útil para embeddings / RAG) |
>
> `metadata['page']` te permite citar “página 12” después. Extra: `pip install pypdf`

---

## 3. DirectoryLoader - Procesamiento Masivo

Para procesar múltiples archivos de forma eficiente.

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os

# Cargar todos los archivos markdown de un proyecto
loader = DirectoryLoader(
    'docs/',
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    recursive=True,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()
print(f"Documentos cargados: {len(docs)}")

# Análisis del contenido cargado
total_chars = sum(len(doc.page_content) for doc in docs)
file_stats = {}

for doc in docs:
    filename = os.path.basename(doc.metadata['source'])
    file_stats[filename] = {
        'chars': len(doc.page_content),
        'words': len(doc.page_content.split()),
        'lines': doc.page_content.count('\n') + 1
    }

# Mostrar estadísticas
print(f"\nTotal de caracteres procesados: {total_chars:,}")
print("\nTop 5 archivos más largos:")
sorted_files = sorted(file_stats.items(), key=lambda x: x[1]['chars'], reverse=True)
for filename, stats in sorted_files[:5]:
    print(f"  {filename}: {stats['chars']:,} chars, {stats['words']:,} words")
```

**Casos de uso ideales:**

- Bases de conocimiento empresariales
- Repositorios de código con documentación
- Colecciones de artículos
- Archivos de proyectos completos

> **Para aclarar:** `DirectoryLoader` no entiende Markdown por sí solo. Recorre la carpeta y, **por cada archivo**, llama a `loader_cls` (aquí `UnstructuredMarkdownLoader`). Si cambiaras a `TextLoader`, leería `.md` como texto plano.
>
> | Parámetro | Qué hace |
> | --- | --- |
> | `glob="**/*.md"` | Solo archivos Markdown |
> | `recursive=True` | Incluye subcarpetas |
> | `use_multithreading=True` | Carga en paralelo (más rápido en carpetas grandes) |

```mermaid
flowchart LR
    DIR["docs/"]
    G["glob **/*.md"]
    M["UnstructuredMarkdownLoader<br/>por cada archivo"]
    OUT["lista de Document"]

    DIR --> G --> M --> OUT

    classDef a fill:#4F46E5,stroke:#312E81,color:#fff
    classDef b fill:#0284C7,stroke:#075985,color:#fff
    classDef c fill:#059669,stroke:#065F46,color:#fff

    class DIR,G a
    class M b
    class OUT c
```

---

## 4. YoutubeLoader - Contenido Multimedia

Extrae transcripciones automáticas de videos de YouTube.

```python
from langchain_community.document_loaders import YoutubeLoader
import re

# Cargar transcripción de un video educativo
video_url = "https://www.youtube.com/watch?v=ejemplo123"
loader = YoutubeLoader.from_youtube_url(
    video_url,
    add_video_info=True,
    language=['es', 'en'],  # Priorizar idiomas
    translation='es'        # Traducir si es necesario
)

try:
    docs = loader.load()
    video_info = docs[0].metadata
    transcript = docs[0].page_content

    print("=== INFORMACIÓN DEL VIDEO ===")
    print(f"Título: {video_info.get('title', 'N/A')}")
    print(f"Autor: {video_info.get('author', 'N/A')}")
    print(f"Duración: {video_info.get('length', 'N/A')} segundos")
    print(f"Fecha de publicación: {video_info.get('publish_date', 'N/A')}")
    print(f"Vistas: {video_info.get('view_count', 'N/A')}")

    print(f"\n=== ANÁLISIS DE TRANSCRIPCIÓN ===")
    print(f"Longitud de transcripción: {len(transcript):,} caracteres")
    print(f"Palabras aproximadas: {len(transcript.split()):,}")

    # Extraer temas principales (palabras frecuentes)
    words = re.findall(r'\b[a-záéíóúñ]{4,}\b', transcript.lower())
    from collections import Counter
    common_words = Counter(words).most_common(10)

    print("\nPalabras más frecuentes:")
    for word, count in common_words:
        print(f"  {word}: {count} veces")

    print(f"\nPrimeros 500 caracteres:")
    print(transcript[:500] + "...")

except Exception as e:
    print(f"Error al cargar el video: {e}")
```

**Casos de uso ideales:**

- Análisis de contenido educativo
- Transcripción de conferencias
- Análisis de tendencias en videos
- Creación de resúmenes automáticos

> **Para aclarar:** no descarga el vídeo. Pide la **transcripción** (la que YouTube ya generó o subió). Sin subtítulos, `load()` falla; por eso el `try/except`.
>
> `page_content` = texto hablado. Título, autor, vistas van en `metadata` si `add_video_info=True`.
>
> Extra habitual: `pip install youtube-transcript-api pytube` (o el paquete que indique tu versión de `langchain-community`).

---

## 5. UnstructuredHTMLLoader - HTML Avanzado

Procesa archivos HTML locales o remotos con análisis inteligente de estructura.

```python
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Procesar archivo HTML local
loader = UnstructuredHTMLLoader(
    "reporte_anual.html",
    mode="elements",  # Preservar estructura de elementos
    strategy="fast"   # Estrategia de procesamiento
)

docs = loader.load()

print(f"Elementos HTML procesados: {len(docs)}")

# Analizar tipos de elementos encontrados
element_types = {}
for doc in docs:
    element_type = doc.metadata.get('category', 'unknown')
    element_types[element_type] = element_types.get(element_type, 0) + 1

print("\nTipos de elementos encontrados:")
for element_type, count in sorted(element_types.items()):
    print(f"  {element_type}: {count}")

# Mostrar elementos de texto más largos
text_elements = [doc for doc in docs if doc.metadata.get('category') == 'NarrativeText']
text_elements.sort(key=lambda x: len(x.page_content), reverse=True)

print(f"\nTop 3 elementos de texto más largos:")
for i, element in enumerate(text_elements[:3]):
    print(f"{i+1}. {len(element.page_content)} caracteres:")
    print(f"   {element.page_content[:150]}...")
```

**Casos de uso ideales:**

- Procesamiento de reportes web
- Análisis de documentación HTML
- Extracción de contenido de emails HTML
- Procesamiento de archivos exportados

> **Para aclarar vs `WebBaseLoader`:**
>
> | | `WebBaseLoader` | `UnstructuredHTMLLoader` |
> | --- | --- | --- |
> | Fuente típica | URL en vivo | Archivo HTML local |
> | Salida | Suele ser 1 documento con casi todo el texto | Con `mode="elements"`, **un Document por elemento** (título, párrafo, tabla…) |
> | `metadata['category']` | No (o muy poco) | Sí: `NarrativeText`, `Title`, etc. |
>
> `mode="elements"` sirve si luego quieres filtrar “solo párrafos” y tirar menús o botones.

---

## 6. CSVLoader - Datos Tabulares

Convierte datos estructurados en documentos procesables.

```python
from langchain_community.document_loaders import CSVLoader

# Configuración avanzada para CSV
loader = CSVLoader(
    file_path="ventas_2024.csv",
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['fecha', 'producto', 'cantidad', 'precio', 'cliente']
    },
    encoding='utf-8',
    source_column='producto',  # Usar una columna como identificador
    metadata_columns=['fecha', 'cliente']  # Incluir en metadatos
)

docs = loader.load()
print(f"Registros de ventas cargados: {len(docs)}")

# Análisis de los datos cargados
productos = set()
clientes = set()
ventas_por_fecha = {}

for doc in docs[:10]:  # Mostrar primeros 10 registros
    print(f"\nRegistro: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")

    # Recopilar estadísticas
    if 'fecha' in doc.metadata:
        fecha = doc.metadata['fecha']
        ventas_por_fecha[fecha] = ventas_por_fecha.get(fecha, 0) + 1

    if 'cliente' in doc.metadata:
        clientes.add(doc.metadata['cliente'])

print(f"\nResumen de datos:")
print(f"  Clientes únicos: {len(clientes)}")
print(f"  Fechas con ventas: {len(ventas_por_fecha)}")
print(f"  Promedio de ventas por día: {len(docs) / len(ventas_por_fecha):.1f}")
```

**Casos de uso ideales:**

- Análisis de datos de ventas
- Logs de sistema
- Datos de encuestas
- Registros de transacciones

> **Para aclarar:** **una fila del CSV = un `Document`**. Un archivo de 10 000 ventas son 10 000 documentos, no uno solo.
>
> - `source_column='producto'` → `metadata['source']` toma el valor de esa columna (útil para citar “este producto”).
> - `metadata_columns` saca campos del texto y los deja en `metadata` (filtros: “solo ventas de marzo”).
> - El resto de columnas suele ir concatenado en `page_content` tipo `fecha: ...\nproducto: ...`.

```mermaid
flowchart LR
    CSV["Fila CSV"]
    DOC["Document"]
    PC["page_content<br/>todas las columnas en texto"]
    MD["metadata<br/>fecha, cliente, source"]

    CSV --> DOC
    DOC --> PC
    DOC --> MD

    classDef a fill:#D97706,stroke:#92400E,color:#fff
    classDef b fill:#4F46E5,stroke:#312E81,color:#fff
    classDef c fill:#059669,stroke:#065F46,color:#fff

    class CSV a
    class DOC b
    class PC,MD c
```

---

## 7. SeleniumURLLoader - JavaScript y Contenido Dinámico

Para páginas web que requieren renderizado de JavaScript.

```python
from langchain_community.document_loaders import SeleniumURLLoader
from selenium.webdriver.chrome.options import Options

# Configurar opciones de navegador
chrome_options = Options()
chrome_options.add_argument("--headless")  # Sin interfaz gráfica
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# URLs con contenido dinámico
urls = [
    "https://example.com/dashboard-dinamico",
    "https://spa-application.com/data",
    "https://chart-website.com/interactive"
]

loader = SeleniumURLLoader(
    urls=urls,
    browser="chrome",
    executable_path="/path/to/chromedriver",  # Opcional si está en PATH
    chrome_options=chrome_options
)

docs = loader.load()

print(f"Páginas procesadas: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n=== PÁGINA {i+1} ===")
    print(f"URL: {doc.metadata['source']}")
    print(f"Título: {doc.metadata.get('title', 'Sin título')}")
    print(f"Contenido renderizado: {len(doc.page_content)} caracteres")

    # Buscar elementos que indiquen contenido dinámico
    dynamic_indicators = ['data-', 'ng-', 'v-', 'react-', 'vue-']
    has_dynamic = any(indicator in doc.page_content for indicator in dynamic_indicators)
    print(f"Contiene elementos dinámicos: {'Sí' if has_dynamic else 'No'}")

    print(f"Vista previa: {doc.page_content[:200]}...")
```

**Casos de uso ideales:**

- Single Page Applications (SPAs)
- Dashboards interactivos
- Sitios con autenticación
- Contenido generado por JavaScript

> **Para aclarar:** Selenium **abre un Chrome real** (aunque sea `--headless`), espera a que corra el JS y luego extrae el HTML ya pintado. Es más lento y pesado que `WebBaseLoader`.
>
> | Loader | JS | Cuándo |
> | --- | --- | --- |
> | `WebBaseLoader` | No | HTML estático, docs, blogs |
> | `SeleniumURLLoader` | Sí | React/Vue, dashboards, contenido que “aparece después” |
>
> Extra: `pip install selenium` y ChromeDriver (o Selenium Manager lo descarga solo en versiones recientes).

---

## 8. GitLoader - Repositorios de Código

Carga archivos directamente desde repositorios Git.

```python
from langchain_community.document_loaders import GitLoader

# Clonar y procesar repositorio
loader = GitLoader(
    clone_url="https://github.com/usuario/proyecto-ejemplo.git",
    repo_path="./temp_repo",
    branch="main",
    file_filter=lambda file_path: file_path.endswith(('.py', '.md', '.txt'))
)

docs = loader.load()
print(f"Archivos cargados del repositorio: {len(docs)}")

# Análisis por tipo de archivo
file_types = {}
total_lines = 0

for doc in docs:
    file_path = doc.metadata['source']
    file_ext = file_path.split('.')[-1] if '.' in file_path else 'sin_extension'

    file_types[file_ext] = file_types.get(file_ext, 0) + 1
    lines = doc.page_content.count('\n') + 1
    total_lines += lines

    print(f"Archivo: {file_path}")
    print(f"  Líneas: {lines}")
    print(f"  Caracteres: {len(doc.page_content)}")
    print(f"  Vista previa: {doc.page_content[:100]}...")
    print()

print(f"\nEstadísticas del repositorio:")
print(f"  Total de líneas de código: {total_lines:,}")
print(f"  Tipos de archivo:")
for ext, count in sorted(file_types.items()):
    print(f"    .{ext}: {count} archivos")
```

**Casos de uso ideales:**

- Análisis de código fuente
- Documentación de proyectos
- Auditorías de repositorios
- Generación de documentación automática

> **Para aclarar:** clona el repo en `repo_path` y luego lee archivos. `file_filter` evita meter `node_modules`, binarios o imágenes: aquí solo `.py`, `.md` y `.txt`.
>
> Cada archivo que pasa el filtro = un `Document`. Extra: `pip install GitPython` y Git instalado en el sistema.

```mermaid
flowchart LR
    URL["clone_url"]
    DISK["repo_path"]
    F["file_filter"]
    DOCS["Document por archivo"]

    URL --> DISK --> F --> DOCS

    classDef a fill:#4F46E5,stroke:#312E81,color:#fff
    classDef b fill:#D97706,stroke:#92400E,color:#fff
    classDef c fill:#059669,stroke:#065F46,color:#fff

    class URL,DISK a
    class F b
    class DOCS c
```
