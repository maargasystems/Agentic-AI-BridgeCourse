# Agentic AI Bridge Course

Notes and hands-on exercises from the Agentic AI bridge course, covering Pydantic data validation and LangChain fundamentals.

## `pydantic.ipynb` — Pydantic Basics

- **Models vs. dataclasses**: Compared a plain Python `@dataclass` (no runtime validation) with a Pydantic `BaseModel` (raises `ValidationError` on type mismatches).
- **Optional fields**: Used `Optional[type]` with default values (`None`, `True`) to make fields non-required, while still validating provided values.
- **List fields**: Modeled a `Classroom` with a `List[str]` field and observed validation errors for wrong item/field types.
- **Nested models**: Built a `Customer` model containing a nested `Address` model, passing a dict that Pydantic auto-parses into the nested type.
- **`Field()` customization**: Applied constraints (`min_length`, `max_length`, `gt`, `le`, `ge`) and defaults (`default`, `default_factory`) with field descriptions.
- **JSON schema generation**: Generated a model's JSON schema via `model_json_schema()`.

## `LangChainBasics/` — LangChain Fundamentals

### `openai/basics.ipynb`
- Set up environment variables for Groq and LangSmith (LangChain tracing/observability).
- Connected to `ChatGroq` models (`gpt-oss-120b`) and ran basic `.invoke()` calls.
- Built a `ChatPromptTemplate` (system + user messages) and chained it with the LLM (`prompt | llm`).
- Added `StrOutputParser` to the chain to get plain string responses (`prompt | llm | output_parser`).

### `openai/genaiapp.ipynb` — Retrieval-Augmented Generation (RAG)
- Loaded a web page with `WebBaseLoader` (LangSmith docs).
- Split documents into chunks using `RecursiveCharacterTextSplitter`.
- Generated embeddings with `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`) and stored them in a `FAISS` vector store.
- Ran similarity search queries against the vector store.
- Built a `create_stuff_documents_chain` document chain with a context-based prompt and `ChatGroq` (`gpt-oss-20b`).
- Combined the retriever and document chain into a full `create_retrieval_chain` RAG pipeline and queried it end-to-end.

### `ollama/genaiapp.ipynb` — RAG on Wikipedia content
- Repeated the RAG pipeline (load → split → embed → FAISS → retrieve → answer) using the Wikipedia "Chennai" page as the source.
- Used `BAAI/bge-large-en-v1.5` embeddings and the `llama-3.1-8b-instant` Groq model for answer generation.

### `ollama/app.py` — Streamlit Chat Demo
- A small Streamlit app (`Langchain Demo With Ollama Model`) that takes a user question via a text input, runs it through a `ChatPromptTemplate | ChatGroq | StrOutputParser` chain, and displays the response.

### `data_ingestion/dataingestion.ipynb` — Document Loaders
- Explored LangChain's document loader integrations:
  - `TextLoader` for plain text files (`speech.txt`).
  - `PyPDFLoader` for PDF files (`attention.pdf`, 15 pages).
  - `WebBaseLoader` for web pages (Wikipedia "Bengaluru").
  - `ArxivLoader` for fetching papers by arXiv ID (e.g., `1706.03762`, the "Attention Is All You Need" paper).
  - `WikipediaLoader` for querying Wikipedia articles directly (e.g., "chennai").
- Sample data files used: `speech.txt`, `attention.pdf`, `records.xml` (in `data_ingestion/`).

### `data_transformer/` — Text Splitters

- **`charactertextsplitter.ipynb`** — `CharacterTextSplitter`: splits on a single character sequence (default `"\n\n"`), measuring chunk size by character count. Demonstrated on `speech.txt` via both `split_documents` (on loaded `Document`s) and `create_documents` (on raw text), noting that chunks larger than `chunk_size` are still produced when no separator occurs within the limit.
- **`recursivecharactertextsplitter.ipynb`** — `RecursiveCharacterTextSplitter`: recursively tries a list of separators to split text while measuring chunk size by character count, producing more evenly-sized chunks than `CharacterTextSplitter`. Applied to `speech.txt` with `chunk_size=100`, `chunk_overlap=20`.
- **`htmltextsplitter.ipynb`** — `HTMLHeaderTextSplitter`: a structure-aware splitter that splits HTML at header elements (`h1`/`h2`/`h3`/`h4`) and attaches each header's text as chunk metadata, preserving document hierarchy. Demonstrated on an inline HTML string and on a live page (`split_text_from_url`).
- **`recursivejsonsplitter.ipynb`** — `RecursiveJsonSplitter`: splits JSON data depth-first into chunks bounded by `max_chunk_size`, keeping nested objects whole where possible. Applied to the LangSmith OpenAPI spec (fetched via `requests`), producing dict chunks (`split_json`), `Document` objects (`create_documents`), and JSON strings (`split_text`).
- Sample data files reused: `speech.txt`, `attention.pdf` (in `data_transformer/`).

## Project Structure

```
.
├── pydantic.ipynb              # Pydantic models, validation, Field constraints
├── LangChainBasics/
│   ├── openai/
│   │   ├── basics.ipynb        # LangChain + Groq basics (prompts, chains, parsers)
│   │   └── genaiapp.ipynb      # RAG pipeline over LangSmith docs
│   ├── ollama/
│   │   ├── genaiapp.ipynb      # RAG pipeline over Wikipedia (Chennai)
│   │   └── app.py              # Streamlit chat demo
│   ├── data_ingestion/
│   │   ├── dataingestion.ipynb # Document loaders (text, PDF, web, arXiv, Wikipedia)
│   │   ├── speech.txt
│   │   ├── attention.pdf
│   │   └── records.xml
│   └── data_transformer/
│       ├── charactertextsplitter.ipynb        # CharacterTextSplitter basics
│       ├── recursivecharactertextsplitter.ipynb # RecursiveCharacterTextSplitter
│       ├── htmltextsplitter.ipynb             # HTMLHeaderTextSplitter
│       ├── recursivejsonsplitter.ipynb        # RecursiveJsonSplitter
│       ├── speech.txt
│       └── attention.pdf
```
