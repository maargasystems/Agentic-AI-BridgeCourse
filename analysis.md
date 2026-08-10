# Repository Analysis — Agentic AI Bridge Course

Overview of the notebooks and scripts in this repo, grouped by topic.

## Pydantic Basics

- **`pydantic.ipynb`** — `BaseModel` vs. `dataclass`, optional fields, list fields, nested models, `Field()` constraints/defaults, and JSON schema generation.

## LangChain Fundamentals (`LangChainBasics/`)

- **`openai/basics.ipynb`** — LangChain + Groq basics: `ChatGroq` setup, `ChatPromptTemplate`, chaining (`prompt | llm | output_parser`) with `StrOutputParser`.
- **`openai/genaiapp.ipynb`** — End-to-end RAG pipeline: `WebBaseLoader` → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → `FAISS` → `create_retrieval_chain`, queried over LangSmith docs.
- **`ollama/genaiapp.ipynb`** — Same RAG pipeline pattern applied to Wikipedia ("Chennai") content, using `bge-large-en-v1.5` embeddings and `llama-3.1-8b-instant`.
- **`ollama/app.py`** — Streamlit chat demo wrapping a `ChatPromptTemplate | ChatGroq | StrOutputParser` chain.

### Document Loading & Splitting

- **`data_ingestion/dataingestion.ipynb`** — Document loaders: `TextLoader`, `PyPDFLoader`, `WebBaseLoader`, `ArxivLoader`, `WikipediaLoader`.
- **`data_transformer/charactertextsplitter.ipynb`** — `CharacterTextSplitter` (fixed-separator splitting).
- **`data_transformer/recursivecharactertextsplitter.ipynb`** — `RecursiveCharacterTextSplitter` (more even chunk sizes).
- **`data_transformer/htmltextsplitter.ipynb`** — `HTMLHeaderTextSplitter` (structure-aware HTML splitting by header tags).
- **`data_transformer/recursivejsonsplitter.ipynb`** — `RecursiveJsonSplitter` over a live OpenAPI spec.

### Embeddings & Vector Stores

- **`embeddings/embedding.ipynb`** — General embedding techniques: converting text into vectors.
- **`embeddings/huggingface.ipynb`** — Embedding techniques using HuggingFace models.
- **`vectorstore/faiss.ipynb`** / **`vectorstore/chroma.ipynb`** — FAISS and Chroma vector store usage, including retriever conversion for use in downstream LangChain chains. (Note: as of this writing, the markdown content inside these two notebooks is swapped relative to their filenames — `faiss.ipynb`'s intro text describes Chroma and vice versa.)

## LCEL (`LCEL/`)

- **`simplellmlcel.ipynb`** — Quickstart building a simple LLM translation app with LangChain Expression Language (LCEL).
- **`app.py` / `client.py`** — Companion server/client scripts (likely a LangServe deployment of the LCEL chain).

## Conversation History (`conversation_history/`)

- **`chatbot.ipynb`** — Building a stateful chatbot: wrapping a model with `MessageHistory` for conversational memory, and strategies for managing/trimming conversation history so it doesn't grow unbounded.

## Supporting Files

- **`main.py`** — Entry-point script.
- **`metrics.json`** — Session activity log and repo file listing.
- **`pyproject.toml` / `uv.lock`** — Project dependencies (managed via `uv`).
