# Agentic AI Bridge Course

Notes and hands-on exercises from the Agentic AI bridge course, covering Pydantic data validation, LangChain fundamentals, retrieval-augmented generation (RAG), LangChain Expression Language (LCEL), conversational memory, LangChain v1 agents/tools/middleware, LangGraph state graphs/chatbots/tool-calling/ReAct agents, common agentic workflow patterns (prompt chaining, routing, parallelization, evaluator-optimizer, orchestrator-worker), human-in-the-loop graph interrupts, and LangGraph-based agentic/corrective/adaptive RAG.

## Tech Stack

- **Language / runtime**: Python 3.12
- **Package management**: [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`)
- **Data validation**: [Pydantic](https://docs.pydantic.dev/) v2
- **LLM orchestration**: [LangChain](https://python.langchain.com/) v1 (`langchain`, `langchain-classic`, `langchain-community`, `langchain-core`) — including `create_agent`, `langchain.agents.middleware`, and the `@tool` decorator
- **Agent runtime**: [LangGraph](https://langchain-ai.github.io/langgraph/) — used both directly (`StateGraph`, `add_messages` reducer, `ToolNode`/`tools_condition` prebuilts, `TypedDict`/dataclass/Pydantic state schemas, ReAct-style tool loops, `MemorySaver`/`InMemorySaver` checkpointing, `.stream()`/`.astream_events()` streaming) and under the hood by `create_agent`/middleware for interrupts, checkpointing, and human-in-the-loop resumption
- **LLM providers**: [Groq](https://groq.com/) via `langchain-groq` / `init_chat_model` (models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.1-8b-instant`, `qwen/qwen3.6-27b`)
- **Observability / tracing**: [LangSmith](https://www.langchain.com/langsmith) via `langsmith`
- **Embeddings**: `langchain-huggingface` + `sentence-transformers` (`BAAI/bge-small-en-v1.5`, `BAAI/bge-large-en-v1.5`)
- **Vector stores**: [FAISS](https://faiss.ai/) (`faiss-cpu`) and [Chroma](https://www.trychroma.com/) (`chromadb`, `langchain-chroma`)
- **Document loading / search tools**: `pypdf`, `pymupdf` (PDF), `beautifulsoup4` (web/HTML), `arxiv`, `wikipedia`, and [Tavily](https://tavily.com/) (`TAVILY_API_KEY`) for web search via `langchain_community.tools.tavily_search`
- **Serving / APIs**: [FastAPI](https://fastapi.tiangolo.com/), [LangServe](https://python.langchain.com/docs/langserve/), `uvicorn`, `sse-starlette`
- **UI**: [Streamlit](https://streamlit.io/)
- **Environment config**: `python-dotenv` (secrets kept in a git-ignored `.env`)

## Project Structure

```
.
├── pydantic/
│   └── pydantic.ipynb             # Pydantic models, validation, Field constraints
├── main.py                        # Placeholder entry-point script
├── LCEL/                          # LangChain Expression Language examples
│   ├── simplellmlcel.ipynb        # LCEL quickstart: simple LLM translation app
│   ├── app.py                     # FastAPI + LangServe server exposing the LCEL chain
│   └── client.py                  # Streamlit client that calls the LangServe endpoint
├── Chatbot_con_history/
│   └── chatbot.ipynb              # Stateful chatbot with message history & trimming
├── LangChainRAG/                  # LangChain fundamentals & RAG pipelines
│   ├── openai/
│   │   ├── basics.ipynb           # LangChain + Groq basics (prompts, chains, parsers)
│   │   └── genaiapp.ipynb         # RAG pipeline over LangSmith docs
│   ├── ollama/
│   │   ├── genaiapp.ipynb         # RAG pipeline over Wikipedia (Chennai)
│   │   └── app.py                 # Streamlit chat demo
│   ├── data_ingestion/
│   │   ├── dataingestion.ipynb    # Document loaders (text, PDF, web, arXiv, Wikipedia)
│   │   ├── speech.txt
│   │   ├── attention.pdf
│   │   └── records.xml
│   ├── data_transformer/
│   │   ├── charactertextsplitter.ipynb          # CharacterTextSplitter basics
│   │   ├── recursivecharactertextsplitter.ipynb # RecursiveCharacterTextSplitter
│   │   ├── htmltextsplitter.ipynb               # HTMLHeaderTextSplitter
│   │   ├── recursivejsonsplitter.ipynb          # RecursiveJsonSplitter
│   │   ├── speech.txt
│   │   └── attention.pdf
│   ├── embeddings/
│   │   ├── embedding.ipynb        # General embedding techniques
│   │   └── huggingface.ipynb      # HuggingFace embedding models
│   └── vectorstore/
│       ├── faiss.ipynb            # FAISS vector store usage & retrievers
│       └── chroma.ipynb           # Chroma vector store usage & retrievers
├── LangChain/                      # LangChain v1 agents, messages, tools, middleware
│   ├── lanchainintro.ipynb        # create_agent quickstart (weatherman tool-calling agent)
│   ├── messages.ipynb             # Unified message model (System/Human/AI/Tool)
│   ├── middleware.ipynb           # SummarizationMiddleware & HumanInTheLoopMiddleware
│   ├── modelintegration.ipynb     # Groq model integration, streaming, batching
│   ├── structuredoutput.ipynb     # with_structured_output (Pydantic/TypedDict/dataclass)
│   └── tools.ipynb                # @tool decorator, bind_tools, manual tool-call loop
├── LangGraph/                      # LangGraph state graphs, chatbots, tool-calling & ReAct agents
│   ├── simplegraph.ipynb          # StateGraph basics: nodes, conditional edges, compile/invoke
│   ├── chainslanggraph.ipynb      # Messages as state, add_messages reducer, bind_tools, ToolNode/tools_condition
│   ├── chatbot.ipynb              # Minimal single-node chatbot graph backed by ChatGroq
│   ├── chatbotwithmultipletools.ipynb # Chatbot wired to Arxiv, Wikipedia & Tavily search tools
│   ├── dataclassstateschema.ipynb # State schemas via TypedDict vs. Python @dataclass
│   ├── pydantic.ipynb             # State schema validation via a Pydantic BaseModel
│   ├── ReActagents.ipynb          # ReAct agent loop (act/observe/reason) with math + search tools & MemorySaver
│   └── streaming.ipynb            # Streaming graph output via .stream() (values/updates) and .astream_events()
├── LangGraphWorkflow/               # Agentic workflow patterns (Anthropic "building effective agents" style)
│   ├── prompt_chaining.ipynb      # Sequential prompts with a conditional gate/quality check between steps
│   ├── routing.ipynb              # Structured-output classifier routes input to a specialized prompt/node
│   ├── parallelization.ipynb      # Independent nodes fan out from START and fan into a combiner node
│   ├── evaluator.ipynb            # Generator/evaluator loop: one LLM drafts, another grades & feeds back
│   └── orchestrator-worker.ipynb  # Orchestrator plans dynamic subtasks; Send() API fans out workers, synthesizer combines results
├── HumanInTheLoop/
│   └── humanintheloop.ipynb       # Interrupt/resume graphs for approval, debugging & editing agent state mid-run
├── RAGLangGraph/                    # RAG patterns built as LangGraph graphs
│   ├── agenticrag.ipynb           # Tool-calling agent decides whether to retrieve, grades relevance, rewrites query & retries
│   ├── correctiverag.ipynb        # Retrieve → grade → (generate | rewrite + web search fallback) self-correcting RAG graph
│   └── adaptiverag.ipynb          # Routes each question to vectorstore or web search, grades docs, hallucinations & answers
├── Debugging/
│   └── graph.py                    # Standalone tool-calling StateGraph script for debugging outside a notebook
├── pyproject.toml / uv.lock        # Project dependencies (managed via uv)
├── analysis.md                    # Standalone repo analysis notes
└── metrics.json                   # Session activity log
```

## `pydantic/pydantic.ipynb` — Pydantic Basics

- **Models vs. dataclasses**: Compared a plain Python `@dataclass` (no runtime validation) with a Pydantic `BaseModel` (raises `ValidationError` on type mismatches).
- **Optional fields**: Used `Optional[type]` with default values (`None`, `True`) to make fields non-required, while still validating provided values.
- **List fields**: Modeled a `Classroom` with a `List[str]` field and observed validation errors for wrong item/field types.
- **Nested models**: Built a `Customer` model containing a nested `Address` model, passing a dict that Pydantic auto-parses into the nested type.
- **`Field()` customization**: Applied constraints (`min_length`, `max_length`, `gt`, `le`, `ge`) and defaults (`default`, `default_factory`) with field descriptions.
- **JSON schema generation**: Generated a model's JSON schema via `model_json_schema()`.

## `LangChainRAG/` — LangChain Fundamentals

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

### `embeddings/` — Embedding Techniques
- **`embedding.ipynb`** — General embedding techniques: converting text into vector representations.
- **`huggingface.ipynb`** — Generating embeddings using HuggingFace models (`langchain-huggingface`, `sentence-transformers`).

### `vectorstore/` — Vector Stores
- **`faiss.ipynb`** — FAISS vector store usage: indexing embedded documents and converting the store into a retriever for downstream LangChain chains.
- **`chroma.ipynb`** — Chroma vector store usage, following the same indexing/retriever pattern. Persisted data is written to `chroma_db/` and `faiss_db/`, which are git-ignored since they're generated artifacts.

## `LCEL/` — LangChain Expression Language

- **`simplellmlcel.ipynb`** — Quickstart building a simple LLM translation app using LCEL (`prompt | model | parser`).
- **`app.py`** — FastAPI server that exposes the LCEL translation chain as a REST endpoint via `add_routes` (LangServe), backed by `ChatGroq` (`gpt-oss-20b`).
- **`client.py`** — Streamlit client that posts user input to the LangServe `/chain/invoke` endpoint and displays the translated response.

## `Chatbot_con_history/chatbot.ipynb` — Conversational Memory

- Building a stateful chatbot: wrapping a `ChatGroq` model with `ChatMessageHistory` / `RunnableWithMessageHistory` to make it remember prior turns per `session_id`.
- Using `MessagesPlaceholder` in a `ChatPromptTemplate` to inject conversation history alongside a `language` variable for multilingual responses.
- **Managing conversation history**: using `trim_messages` to cap how many tokens/messages are sent to the model (keeping the system message, controlling partial-message handling) so history doesn't grow unbounded and overflow the context window.

## `LangChain/` — LangChain v1 Agents, Messages, Middleware & Model Integration (Groq)

### `lanchainintro.ipynb`
- Pinned LangChain v1 (`langchain.__version__` = `1.3.14`) and configured Groq + LangSmith tracing via environment variables (`GROQ_API_KEY`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_TRACING_V2`).
- Built a weatherman agent with `create_agent` from `langchain.agents`, using `ChatGroq(model="openai/gpt-oss-20b")` and a custom `get_weather(city)` Python function as a tool, plus a `system_prompt`.
- Invoked the agent with both structured (`{"messages":[{"role":"user", ...}]}`) and shorthand (`{"messages": "..."}`) input formats.
- Demonstrated the full ReAct-style loop: `HumanMessage` → `AIMessage` with `tool_calls` → `ToolMessage` result → final `AIMessage` answer, inspecting `response["messages"]`.

### `messages.ipynb`
- Introduced LangChain's unified message model (System/Human/AI/Tool) and used `init_chat_model(model="groq:openai/gpt-oss-120b")` for provider-agnostic model construction.
- Compared text-prompt `.invoke("string")` calls versus message-list prompts built from `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` (`langchain.messages` / `langchain_core.messages`).
- Showed priming model behavior with a `SystemMessage` (e.g., "act as a senior python developer") and attaching `name`/`id` metadata to a `HumanMessage`.
- Manually constructed conversation history by inserting a pre-made `AIMessage` into a message list to simulate multi-turn context.
- Demonstrated manual tool-call round-tripping by constructing an `AIMessage` with a `tool_calls` payload and a matching `ToolMessage(tool_call_id=...)`, then feeding both back into `model.invoke(...)`.

### `middleware.ipynb`
- Covered `langchain.agents.middleware` classes on top of `create_agent(model="groq:openai/gpt-oss-120b", ...)`.
- `SummarizationMiddleware`: compresses conversation history using `trigger`/`keep` thresholds expressed as message count (`("messages", 10)`/`("messages", 4)`), token count (`("tokens", 400)`/`("tokens", 100)`), and context-window fraction (`("fraction", 0.005)`/`("fraction", 0.002)`), tested across multi-turn arithmetic Q&A and a `search_hotel`/`search_hotels` `@tool` across multiple cities, using `InMemorySaver` (from `langgraph.checkpoint.memory`) for thread-scoped state (`thread_id` in config).
- `HumanInTheLoopMiddleware`: gated a `send_email_tool` (mock) behind human approval via `interrupt_on={"send_email_tool": {"allowed_decisions": [...]}, "read_email_tool": False}`, while leaving `read_email_tool` un-interrupted.
- Demonstrated all three human decisions using `langgraph.types.Command(resume={"decisions": [...]})`: `approve`, `reject` (tool call cancelled, agent asks user how to proceed), and `edit` (human rewrites `recipient`/`subject`/`body` before the tool actually executes).
- Illustrated LangGraph's `__interrupt__` mechanism for pausing/resuming agent execution mid-run.

### `modelintegration.ipynb`
- Focused on connecting to Groq-hosted models via two equivalent paths: `init_chat_model(model="groq:openai/gpt-oss-20b")` and `ChatGroq(model=...)` directly (`from langchain_groq.chat_models import ChatGroq`), also trying `qwen/qwen3.6-27b`.
- Inspected model metadata via `model.profile` (context window, input/output token limits, modality support, tool-calling/structured-output/reasoning flags).
- Used `.invoke()` for single calls and `.stream()` for token-by-token streaming (printing `chunk.text`), observing visible `<think>...</think>` reasoning traces from the Qwen model.
- Used `.batch()` for parallel prompt execution, including a `config={"max_concurrency": 3}` example across multiple unrelated questions.

### `structuredoutput.ipynb`
- Demonstrated `model.with_structured_output(...)` on `ChatGroq`/`init_chat_model("groq:openai/gpt-oss-120b")` across three schema styles: Pydantic `BaseModel` (`Movie`, nested `MovieDetails`/`Actor`), `typing_extensions.TypedDict` (`MovieDict`, nested `MovieDetails`), and Python `@dataclass` (`ContactInfo`).
- Showed `include_raw=True` to get both the raw `AIMessage` (with tool-call args) and the `parsed` Pydantic object plus `parsing_error` in one call.
- Illustrated nested/list fields (`cast: list[Actor]`, `genres: list[str]`, optional `budget: float | None`) being correctly populated by the model.
- Used `create_agent(model="groq:openai/gpt-oss-120b", response_format=ContactInfo)` to get a `structured_response` key on the agent's invoke result for Pydantic, TypedDict, and dataclass response formats alike, extracting contact info (name/email/phone) from free text.

### `tools.ipynb`
- Defined a tool with the `@tool` decorator from `langchain.tools` (`get_weather(city:str)->str`) and bound it to `init_chat_model(model="groq:openai/gpt-oss-120b")` via `model.bind_tools([...])`.
- Inspected the resulting `AIMessage.tool_calls` list (name, args, id) produced when the model decides to call the tool.
- Walked through a manual tool-execution loop: append the model's tool-call `AIMessage` to the message list, call `get_weather.invoke(tool_call)` to produce a `ToolMessage`, append it, then call `model_with_tool.invoke(messages)` again to get the final natural-language answer via `.text`.

## `LangGraph/` — State Graphs, Chatbots & Tool-Calling Agents

### `simplegraph.ipynb`
- Defined a `State` as a `TypedDict` and built plain-Python node functions (`start_play`, `cricket`, `badminton`) that read/return state keys.
- Used a conditional-edge routing function (`random_play`, returning a `Literal["cricket", "badminton"]`) to branch the graph at random via `add_conditional_edges`.
- Assembled the graph with `StateGraph`, `START`/`END`, `add_node`/`add_edge`, compiled it, rendered it as a Mermaid diagram (`draw_mermaid_png`), and ran it with `graph.invoke(...)`.

### `chainslanggraph.ipynb`
- Modeled conversation state as a list of LangChain messages (`HumanMessage`/`AIMessage`/`ToolMessage`) and introduced the **reducer** concept: without one, a node's return value overwrites the state key; with the prebuilt `add_messages` reducer (via `Annotated[list[AnyMessage], add_messages]`), new messages are appended instead.
- Bound a simple `add(a, b)` Python tool to `ChatGroq(model="qwen/qwen3.6-27b")` via `bind_tools([add])` and inspected the resulting `AIMessage.tool_calls`.
- Built a one-node graph (`llm_tool`) that calls the tool-bound model, then upgraded it to a full tool-calling loop using the prebuilt `ToolNode` and `tools_condition` (from `langgraph.prebuilt`) to route between the LLM node and tool execution.

### `chatbot.ipynb`
- Minimal chatbot graph: a `State` with an `add_messages`-annotated `messages` key and a single `chatbot` node (`superbot`) that calls `ChatGroq(model="llama-3.1-8b-instant")` on the running message history.
- Compiled with just `START → chatbot → END` (no tools) and invoked it with a `("user", "...")` tuple message.

### `chatbotwithmultipletools.ipynb`
- Wired up three external tools: `ArxivQueryRun` (via `ArxivAPIWrapper`), `WikipediaQueryRun` (via `WikipediaAPIWrapper`), and `TavilySearchResults` (web search, requires `TAVILY_API_KEY`).
- Bound all three tools to `ChatGroq(model="qwen/qwen3.6-27b")` and inspected tool-selection behavior for arXiv IDs, general-knowledge, and current-events queries.
- Built the same `ToolNode` + `tools_condition` graph pattern as `chainslanggraph.ipynb` (`tool_calling_node` → conditional routing → `tools` → `END`) so the agent can autonomously pick and call arXiv, Wikipedia, or Tavily based on the user's question.

### `dataclassstateschema.ipynb`
- Compared two ways of defining a graph's state schema beyond `TypedDict`: a `TypedDict` with a `Literal["cricket", "badminton"]` field (type hints only, not enforced at runtime) and a Python `@dataclass` (`DataClassState`) with the same shape.
- Rebuilt the same play/cricket/badminton branching graph from `simplegraph.ipynb` against the dataclass schema, invoking it with a `DataClassState(...)` instance instead of a plain dict.

### `pydantic.ipynb`
- Defined a graph state as a Pydantic `BaseModel` (`class State(BaseModel): name: str`) to get runtime type validation on inputs, unlike `TypedDict`/dataclass schemas.
- Showed a valid invocation (`graph.invoke({"name": "saran"})`) succeeding and an invalid one (`graph.invoke({"name": 123})`) raising a Pydantic validation error at the graph boundary.

### `ReActagents.ipynb` — ReAct Agent Architecture
- Explained the ReAct loop: **act** (model calls a tool) → **observe** (tool output returned to the model) → **reason** (model decides whether to call another tool or answer).
- Assembled a six-tool toolkit — `ArxivQueryRun`, `WikipediaQueryRun`, `TavilySearchResults`, and custom `add`/`multiply`/`divide` functions — bound to `ChatGroq(model="qwen/qwen3.6-27b")` via `bind_tools`.
- Built the same `State`/`ToolNode`/`tools_condition` graph pattern as `chainslanggraph.ipynb` and ran multi-tool queries (e.g., "give me recent AI news, add 5 plus 5, then multiply by 10") that chain several tool calls before the final answer.
- **Agent memory**: introduced `MemorySaver` as a checkpointer (`builder.compile(checkpointer=memory)`) so the graph persists state across turns keyed by `thread_id`, enabling follow-up questions like "divide that by 5?" or "can you add that with two plus two?" that reference prior results.

### `streaming.ipynb` — Streaming Graph Output
- Built a minimal single-node chatbot graph (`SuperBot`) compiled with a `MemorySaver` checkpointer.
- Compared `.stream()` modes: `"updates"` (only the delta produced by each node) vs. `"values"` (the full graph state after each node).
- Introduced `.astream_events(..., version="v2")` for streaming lower-level events (including token-by-token model output), as the async alternative to `.stream()` for finer-grained observability.

## `LangGraphWorkflow/` — Agentic Workflow Patterns

Notebooks modeling the common workflow archetypes (prompt chaining, routing, parallelization, evaluator-optimizer, orchestrator-worker), each built as a small `StateGraph` over `ChatGroq(model="qwen/qwen3.6-27b")` (orchestrator-worker uses `llama-3.1-8b-instant`).

### `prompt_chaining.ipynb`
- Chained sequential LLM calls — `generate_story` → `improved_story` → `polished_story` — where each node's output feeds the next node's prompt.
- Inserted a conditional gate (`add_conditional_edges` with a `check_conflict` routing function) after the first step to short-circuit the chain when a quality check fails.

### `routing.ipynb`
- Defined a `Route` Pydantic schema (`step: Literal["poem", "story", "joke"]`) and used `with_structured_output` so the LLM classifies the input into one of several paths.
- Used `add_conditional_edges` to dispatch to a dedicated node/prompt per route (poem vs. story vs. joke generation) instead of a single generic prompt handling every case.

### `parallelization.ipynb`
- Fanned out three independent nodes (`generate_character`, `generate_settings`, `generate_premises`) directly from `START`, each calling the LLM with an unrelated sub-prompt over the same `topic`.
- Fanned all three back into a single `combine_elements` node that assembles their outputs into one `story_intro`, demonstrating concurrent execution for independent sub-tasks vs. the sequential chaining pattern.

### `evaluator.ipynb` — Evaluator-Optimizer
- Modeled a generator/evaluator loop: `llm_call_generator` writes a joke about a topic (optionally incorporating prior `feedback`), and `llm_call_evaluator` grades it via a structured `Feedback` schema (`grade: Literal["funny", "not funny"]`, `feedback: str`).
- Routed with a conditional edge back to the generator when the grade is "not funny" (feeding the critique back in) and to `END` once the evaluator approves — useful when there's a clear evaluation criterion and iterative refinement improves the result.

### `orchestrator-worker.ipynb` — Orchestrator-Worker
- Used for tasks where the subtasks can't be predicted up front (unlike parallelization's fixed, pre-defined nodes): a central LLM plans the subtasks, then delegates each one dynamically.
- `orchestrator` calls `llm.with_structured_output(Sections)` to plan a report as a list of `Section(topic, description)` objects.
- `assign_worker` uses LangGraph's `Send` API (`from langgraph.types import Send`) inside `add_conditional_edges` to fan out one `llm_call` worker per planned section at runtime, each with its own `worker_state`.
- Each `llm_call` worker writes one report section; all worker outputs accumulate into a shared `completed_sections` key via the `operator.add` reducer (`Annotated[List, operator.add]`).
- `synthesizer` joins the completed sections into a single `final_report` string, rendered as Markdown.

## `HumanInTheLoop/humanintheloop.ipynb` — Interrupts & State Editing

- Covered three human-in-the-loop motivations: **approval** (pause and let a user accept an action), **debugging** (rewind the graph to reproduce/avoid issues), and **editing** (modify state mid-run).
- Built an arithmetic assistant (`add`/`multiply`/`divide` tools bound to `ChatGroq`) as a `MessagesState` graph with `ToolNode`/`tools_condition`, compiled with `interrupt_before=["assistant"]` and a `MemorySaver` checkpointer.
- Used `graph.stream(initial_input, thread, stream_mode="values")` followed by `graph.stream(None, thread, ...)` to pause before the `assistant` node and then resume execution from the checkpoint.
- Inspected paused state via `graph.get_state(thread)` / `state.next` to see which node is queued to run next.
- **Editing feedback**: called `graph.update_state(thread, {"messages": [...]})` to inject a corrected human message (e.g., change "multiply 2 and 3" to "multiply 15 and 6") into the checkpointed state before resuming.
- **Waiting for user input**: added a no-op `human_feedback` node before `assistant` with `interrupt_before=["human_feedback"]`, then resumed with `graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")` to inject fresh input captured via `input(...)` and continue the run.

## `RAGLangGraph/` — RAG as LangGraph Graphs

### `agenticrag.ipynb` — Agentic RAG
- Built two separate `FAISS` retriever tools via `create_retriever_tool` — `search_documents` and `retriever_vector_langchain_blog` — each backed by its own `WebBaseLoader` → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` pipeline, and bound both to a `ChatGroq(model="openai/gpt-oss-120b")` agent.
- `agent` node: the tool-bound LLM decides whether to call a retriever tool or answer directly, based on `AgentState.messages` (`Annotated[Sequence[BaseMessage], add_messages]`).
- `grade_documents` conditional edge (`Literal["generate", "rewrite"]`): grades the retrieved tool output for relevance to the question and routes to `generate` or `rewrite`.
- `rewrite` node: reformulates the question to better capture semantic intent when retrieved documents were graded irrelevant, looping back through the agent.
- `generate` node: pulls the `rlm/rag-prompt` from the LangSmith prompt hub and answers using the retrieved context via `prompt | llm | StrOutputParser()`.
- Assembled with `StateGraph` + `ToolNode([retriever_tool, retriever_tool_langchain])` + `tools_condition`, tested against LangGraph/LangChain/off-topic ("What is Machine learning?") questions to show routing behavior.

### `correctiverag.ipynb` — Corrective RAG (CRAG)
- Indexed LangChain docs into a `FAISS` store (`HuggingFaceEmbeddings`, `BAAI/bge-small-en-v1.5`) and defined a `GraphState` (`question`, `generation`, `web_search`, `documents`).
- `retrieve` → `grade_documents`: a `GradeDocuments` structured-output grader (`ChatGroq(model="openai/gpt-oss-120b")`) scores each retrieved doc as relevant (`binary_score: "yes"/"no"`); any irrelevant doc flips a `web_search` flag.
- `decide_to_generate` conditional edge routes to `generate` when all docs are relevant, or to `transform_query` (a `question_rewriter` chain optimizing the query for web search) when any are not.
- `web_search` node falls back to `TavilySearchResults` to fetch supplementary documents before generation when local retrieval was graded insufficient.
- `generate` node answers via `rag_chain = prompt | llm | StrOutputParser()` (LangSmith `rlm/rag-prompt`) over the final (possibly web-augmented) document set — self-correcting the retrieval step instead of trusting it outright, unlike the plain RAG pipelines in `LangChainRAG/`.

### `adaptiverag.ipynb` — Adaptive RAG
- Combined **query routing** with corrective self-checks: each question is first routed to the best retrieval strategy, then the pipeline grades its own retrieval and generation before returning an answer.
- Indexed the LangChain tutorials page (`WebBaseLoader` → `RecursiveCharacterTextSplitter` with `chunk_size=500`/`chunk_overlap=50` → `FAISS` + `HuggingFaceEmbeddings`).
- **Router**: a `RouteQuery` structured-output chain on `ChatGroq(model="openai/gpt-oss-120b")` (`datasource: Literal["vectorstore", "websearch"]`) sends in-domain questions to the vector store and everything else to web search (`TavilySearchResults(max_results=3)`).
- **Three binary graders** built with `with_structured_output`: `GradeDocuments` (is each retrieved doc relevant?), `GradeHallucinations` (is the generation grounded in the docs?), and an answer grader (does the generation address the question?), plus a question re-writer chain that reformulates queries for better retrieval.
- Defined a `GraphState` TypedDict (`question`, `generation`, `documents`, `loop_count`) with a `MAX_RETRIES = 3` cap on rewrite/generate loops.
- **Graph wiring**: `START` conditionally routes (`route_question`) to `web_search` or `retrieve`; `retrieve` → `grade_documents` → `decide_to_generate` picks `generate` or `transform_query` (which loops back to `retrieve`); after `generate`, `grade_generation_v_documents_and_question` routes "not supported" back to `generate`, "not useful" to `transform_query`, and "useful" to `END` — answers via the LangSmith `rlm/rag-prompt` RAG chain.

## `Debugging/graph.py`

- A standalone (non-notebook) script version of the `chainslanggraph.ipynb` tool-calling pattern: a `State` TypedDict with an `add_messages`-reduced `messages` key, an `add` tool, and a `ToolNode`-based agent/tools loop.
- Exposes `make_default_graph()` (plain LLM, no tools) and `make_alternate_graph()` (tool-calling agent) factory functions, useful for debugging/inspecting a compiled graph outside a Jupyter kernel (e.g., via `python Debugging/graph.py` or LangGraph Studio/CLI tooling that expects a `.py` graph entry point).

## Supporting Files

- **`main.py`** — Placeholder entry-point script (`uv run main.py`).
- **`analysis.md`** — Standalone analysis notes covering the same notebooks/scripts, grouped by topic.
- **`metrics.json`** — Session activity log and repo file listing.
- **`pyproject.toml` / `uv.lock`** — Project dependencies, managed via `uv`.

## Getting Started

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Create a .env with your API keys (git-ignored)
# GROQ_API_KEY=...
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=...
# TAVILY_API_KEY=...

# Run a notebook
uv run jupyter notebook

# Run a Streamlit demo, e.g.
uv run streamlit run LangChainRAG/ollama/app.py

# Run the LCEL LangServe API + Streamlit client
uv run LCEL/app.py
uv run streamlit run LCEL/client.py
```
