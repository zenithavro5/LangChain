# 🦜🔗 LangChain Mastery Roadmap

> A structured, progressive guide to mastering the LangChain framework — from zero to production-ready AI applications.

---

## Prerequisites

Before diving into LangChain, make sure you're comfortable with the following:

- **Python** — intermediate level (functions, classes, decorators, async/await)
- **APIs** — how REST APIs work, using `requests` or `httpx`
- **Basic ML/AI concepts** — what LLMs are, tokens, temperature, prompts
- **OpenAI API** — at least basic usage of `openai` Python SDK
- **Virtual environments** — `venv`, `conda`, or `poetry`
- **Git** — version control basics

---

## Phase 1 — Foundations (Week 1–2)

### 1.1 Understanding the LangChain Ecosystem

- What is LangChain and why it exists
- The LangChain package structure: `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`
- LangChain vs LlamaIndex vs raw API calls — when to use what
- Setting up your environment and API keys

**Resources:**
- Official docs: https://python.langchain.com/docs/get_started/introduction
- LangChain GitHub: https://github.com/langchain-ai/langchain

### 1.2 Core Concepts

- **Models** — LLMs vs Chat Models vs Embedding Models
- **Prompts** — `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`
- **Output Parsers** — `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`
- **Messages** — `HumanMessage`, `AIMessage`, `SystemMessage`

**Practice Project:** Build a CLI chatbot using `ChatOpenAI` + `ChatPromptTemplate` + `StrOutputParser`

---

## Phase 2 — LangChain Expression Language (LCEL) (Week 3)

### 2.1 The Pipe Operator & Chains

- Understanding `|` (pipe) syntax for chaining components
- Building basic chains: `prompt | model | parser`
- Runnable interface: `.invoke()`, `.stream()`, `.batch()`
- `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`

### 2.2 Advanced LCEL Patterns

- Branching logic with `RunnableBranch`
- Dynamic routing between chains
- Fallbacks with `.with_fallbacks()`
- Retries with `.with_retry()`
- Binding model parameters with `.bind()`

**Practice Project:** Build a multi-step content generation pipeline (topic → outline → full article) using LCEL chains.

---

## Phase 3 — Memory & State (Week 4)

### 3.1 Conversation Memory

- Why stateless LLMs need memory management
- `ConversationBufferMemory` — stores full history
- `ConversationBufferWindowMemory` — sliding window
- `ConversationSummaryMemory` — summarizes old messages
- `ConversationSummaryBufferMemory` — hybrid approach

### 3.2 Modern State Management

- Using `RunnableWithMessageHistory` with LCEL
- Chat history backends: in-memory, Redis, file-based
- Session management for multi-user apps
- Trimming and filtering message history

**Practice Project:** Build a stateful customer support chatbot that remembers context across turns.

---

## Phase 4 — Document Loaders, Text Splitters & Embeddings (Week 5)

### 4.1 Loading Data

- `TextLoader`, `PyPDFLoader`, `CSVLoader`, `WebBaseLoader`, `YoutubeLoader`
- Loading from databases, APIs, and cloud storage
- Building custom document loaders

### 4.2 Text Splitting

- Why chunking matters for RAG
- `RecursiveCharacterTextSplitter` (most commonly used)
- `CharacterTextSplitter`, `MarkdownTextSplitter`, `TokenTextSplitter`
- Chunk size vs overlap — tuning for your use case

### 4.3 Embeddings

- What embeddings are and how they work
- `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, `CohereEmbeddings`
- Embedding documents vs queries
- Dimensionality and similarity search (cosine, dot product, Euclidean)

**Practice Project:** Embed a collection of PDF documents and find the most semantically similar chunks to a user query.

---

## Phase 5 — Vector Stores & Retrieval (Week 6)

### 5.1 Vector Databases

- What vector stores do and how they work
- **Local/open-source:** FAISS, Chroma, Qdrant
- **Cloud/managed:** Pinecone, Weaviate, MongoDB Atlas
- CRUD operations: `add_documents`, `similarity_search`, `delete`

### 5.2 Retrievers

- `VectorStoreRetriever` with `similarity_search` and MMR
- `MultiQueryRetriever` — generates multiple query variants
- `ContextualCompressionRetriever` — compresses retrieved docs
- `SelfQueryRetriever` — filters by metadata using LLM
- `ParentDocumentRetriever` — retrieves parent chunks for context
- Ensemble Retriever — combining BM25 + vector search (hybrid)

**Practice Project:** Build a semantic search engine over a documentation website.

---

## Phase 6 — RAG (Retrieval-Augmented Generation) (Week 7)

### 6.1 Basic RAG Pipeline

- The full RAG architecture: Load → Split → Embed → Store → Retrieve → Generate
- Building a Q&A chain over documents with LCEL
- `create_retrieval_chain` and `create_stuff_documents_chain`
- Handling "I don't know" responses gracefully

### 6.2 Advanced RAG Techniques

- **HyDE** (Hypothetical Document Embeddings)
- **RAG Fusion** — reciprocal rank fusion across multiple retrievals
- **Reranking** — using Cohere Rerank or a cross-encoder
- **Corrective RAG (CRAG)** — self-correcting retrieval
- **Conversational RAG** — reformulating queries with chat history
- Evaluating RAG pipelines with RAGAS

**Practice Project:** Build a full RAG chatbot over your own knowledge base (PDFs, Notion docs, or a website).

---

## Phase 7 — Tools & Agents (Week 8–9)

### 7.1 Tools

- What tools are and how LLMs use them
- Built-in tools: `DuckDuckGoSearchRun`, `WikipediaQueryRun`, `PythonREPLTool`, `ArxivQueryRun`
- Creating custom tools with `@tool` decorator and `BaseTool`
- Tool schemas and argument validation with Pydantic

### 7.2 Agent Types

- `ReAct` agent — Reason + Act loop
- `OpenAI Functions` / `OpenAI Tools` agents
- `Structured Chat` agents for multi-input tools
- How agent scratchpads and intermediate steps work

### 7.3 AgentExecutor

- Running agents with `AgentExecutor`
- Controlling iterations and timeouts
- Handling errors and parsing failures
- Streaming agent outputs
- Adding memory to agents

**Practice Project:** Build a research assistant agent that can search the web, read Wikipedia, and run Python code to answer complex questions.

---

## Phase 8 — LangGraph (Week 10–11)

> LangGraph is LangChain's framework for building stateful, multi-actor agentic workflows using graphs.

### 8.1 LangGraph Fundamentals

- Why LangGraph exists — limitations of `AgentExecutor`
- Core concepts: **Nodes**, **Edges**, **State**, **Graph**
- `StateGraph` vs `MessageGraph`
- `TypedDict` for defining state schemas
- Compiling and running graphs

### 8.2 Control Flow

- Conditional edges for dynamic routing
- Cycles and loops in graphs
- Human-in-the-loop with `interrupt_before` / `interrupt_after`
- Checkpointing and state persistence with `MemorySaver`

### 8.3 Multi-Agent Systems

- Supervisor pattern — orchestrating multiple specialist agents
- Hierarchical agent teams
- Passing state between agents
- Parallelism with `Send` API

**Practice Project:** Build a multi-agent workflow where a supervisor routes tasks between a researcher, a writer, and a code executor agent.

---

## Phase 9 — LangSmith & Observability (Week 12)

### 9.1 LangSmith

- Setting up LangSmith tracing
- Understanding traces, runs, and spans
- Debugging chain and agent failures
- Evaluating LLM outputs with datasets and evaluators

### 9.2 Evaluation & Testing

- Building evaluation datasets
- Running automated evaluations (LLM-as-judge, exact match, embedding similarity)
- A/B testing prompts and models
- Regression testing your chains

**Practice Project:** Set up LangSmith tracing for one of your previous projects and write an evaluation suite for it.

---

## Phase 10 — Production & Deployment (Week 13–14)

### 10.1 LangServe

- Deploying chains as REST APIs with LangServe
- Auto-generated Swagger UI and playground
- Input/output schemas
- Authentication and middleware

### 10.2 Performance & Cost Optimization

- Caching LLM calls with `InMemoryCache` and `SQLiteCache`
- Semantic caching with `GPTCache`
- Streaming responses for better UX
- Batching requests efficiently
- Token counting and cost estimation

### 10.3 Production Best Practices

- Environment and secrets management
- Rate limiting and error handling
- Async execution with `ainvoke`, `astream`, `abatch`
- Structured logging and monitoring
- Containerizing LangChain apps with Docker

**Practice Project:** Deploy your RAG chatbot as a production API using LangServe + Docker.

---

## Recommended Project Progression

| Level | Project |
|-------|---------|
| Beginner | CLI Q&A chatbot with memory |
| Beginner | Document summarizer (PDF → bullet points) |
| Intermediate | RAG system over a personal knowledge base |
| Intermediate | Web research agent with tool use |
| Advanced | Multi-agent content pipeline with LangGraph |
| Advanced | Production RAG API with LangServe + LangSmith evals |

---

## Key Libraries & Integrations to Know

- **LLM Providers:** OpenAI, Anthropic, Google Gemini, Groq, Ollama (local)
- **Vector Stores:** FAISS, Chroma, Pinecone, Qdrant
- **Document Loaders:** PyPDF, BeautifulSoup, Docx, Notion, Confluence
- **Embeddings:** OpenAI, HuggingFace, Cohere
- **Frameworks:** FastAPI (for custom APIs), Streamlit/Gradio (for demos)

---

## Useful Resources

| Resource | Link |
|----------|------|
| Official Docs | https://python.langchain.com |
| LangChain Cookbook | https://github.com/langchain-ai/langchain/tree/master/cookbook |
| LangGraph Docs | https://langchain-ai.github.io/langgraph |
| LangSmith Docs | https://docs.smith.langchain.com |
| LangChain YouTube | https://www.youtube.com/@LangChain |
| Deeplearning.ai LangChain courses | https://www.deeplearning.ai |

---

## Estimated Timeline

| Phase | Duration |
|-------|----------|
| Prerequisites | 1–2 weeks (if needed) |
| Phases 1–3 (Core) | 3–4 weeks |
| Phases 4–6 (RAG) | 3 weeks |
| Phases 7–8 (Agents + LangGraph) | 3–4 weeks |
| Phases 9–10 (Production) | 2 weeks |
| **Total** | **~12–16 weeks** |

> Tip: Build something real at every phase. The framework moves fast — always check the latest docs.

---
