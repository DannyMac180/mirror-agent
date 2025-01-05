                                                    ┌───────────────────────────────────────────────────┐
                                                    │                  User Interface                   │
                                                    │  - Receives final AI responses                   │
                                                    │  - Sends queries/messages to the Agent           │
                                                    └───────────────▲───────────────────────────────────┘
                                                                    │
                                                                    │ (1) messages & user input
                                                                    │
                 ┌───────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
                 │                                    mirror-agent "RetrievalGraph"                                  │
                 │            (Defined in src/retrieval_graph/graph.py, compiled to `graph`)                         │
                 │                                                                                                   │
                 │   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
                 │   │  Flow Nodes (StateGraph from langgraph.graph)                                              │  │
                 │   │-------------------------------------------------------------------------------------------│  │
                 │   │                                                                                           │  │
                 │   │  1) "__start__" ────────────────────> "should_retrieve" (should_retrieve function)        │  │
                 │   │                               │ (decides whether to retrieve or respond)                  │  │
                 │   │                               └────────────────────────────────────────────────────────────▶  │
                 │   │                                                                                           │  │
                 │   │  2) "should_retrieve" ──(if next="retrieve")─> "generate_query" (generate_query function) │  │
                 │   │                               └──────────────(if next="respond")──────> "respond"         │  │
                 │   │                                                                                           │  │
                 │   │  3) "generate_query" ─────────────────────────────────────────────────────┬───────────────▶  │
                 │   │       (uses ChatPromptTemplate & LLM to refine user question into a query)│               │  │
                 │   │                                                                            │  (5) queries │  │
                 │   │  4) "retrieve" (retrieve function)                                          ▼               │  │
                 │   │       (performs doc retrieval via make_retriever(...) -> search)         "retrieved_docs"   │  │
                 │   │                                                                                           │  │
                 │   │  5) "respond" (respond function) ──────────────────────────────────────────────────────────▶  │
                 │   │       (generates final answer using LLM, system prompt, conversation state, etc.)         │  │
                 │   │                                                                                           │  │
                 │   └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
                 │                                                                                                   │
                 │                           Internal Data Structures (from state.py)                                 │
                 │   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
                 │   │ class State(InputState):                                                                    │  │
                 │   │    - messages: Sequence[AnyMessage]    ( conversation history )                             │  │
                 │   │    - queries: list[str]                ( generated search queries )                         │  │
                 │   │    - retrieved_docs: list[Document]    ( search results from the retriever )               │  │
                 │   │    - next: str                          ( routing field: "retrieve" / "respond" )           │  │
                 │   └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
                 │                                                                                                   │
                 │                              Prompt & LLM Integration (from prompts.py, utils.py)                 │
                 │   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │
                 │   │ - RESPONSE_SYSTEM_PROMPT & QUERY_SYSTEM_PROMPT define "system" instructions for LLM         │  │
                 │   │ - load_chat_model(...) returns a BaseChatModel (e.g. openai/gpt-4o)                          │  │
                 │   │ - ChatPromptTemplate structures messages for the LLM call                                    │  │
                 │   │ - The "respond" node uses "configuration.response_system_prompt" and "configuration.response_model"  
                 │   └─────────────────────────────────────────────────────────────────────────────────────────────┘  │
                 │                                                                                                   │
                 └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                    │
                                                                    │ (2) calls "retriever" logic with config
                                                                    ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        │                                          retrieval.py : make_retriever(...)                                            │
        │                                                                                                                        │
        │  This module dynamically creates a VectorStoreRetriever from the chosen provider:                                      │
        │     - "elastic" / "elastic-local" (Elasticsearch)                                                                      │
        │     - "pinecone" (Pinecone Vector DB)                                                                                  │
        │     - "mongodb" (MongoDB Atlas Vector Search)                                                                          │
        │                                                                                                                        │
        │  Steps:                                                                                                                │
        │  1) Read user config (IndexConfiguration or Configuration) from runnable config                                        │
        │  2) Instantiate embeddings (make_text_encoder(...)), e.g. "openai/text-embedding-3-small"                              │
        │  3) Connect to the appropriate vector store backend                                                                    │
        │  4) yield a .as_retriever(...) object to be invoked from the "retrieve" node                                           │
        └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                    │
                                                                    │ (3) obtains relevant docs
                                                                    ▼
                                ┌───────────────────────────────────────────────────────────────────────────────────┐
                                │                Retrieved Documents                                              │
                                │    - Document objects from langchain_core.documents.Document                    │
                                │    - Possibly contain "page_content", "metadata" incl. "user_id"                │
                                └───────────────────────────────────────────────────────────────────────────────────┘
                                                                    │
                                                                    │ (4) final LLM step to produce answer
                                                                    ▼
                  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                  │                                    mirror-agent "Respond" Node                                       │
                  │                                         (from graph.py : respond)                                    │
                  │ - Combines conversation messages, system prompts, and retrieved_docs into the final LLM call         │
                  │ - The "response_model" is loaded from the "Configuration" (e.g. openai/gpt-4o)                        │
                  │ - Output is appended as an AIMessage back to the conversation "messages"                              │
                  │ - Control returns to user with the answer                                                             │
                  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                    │
                                                                    │ (5) final answer
                                                                    ▼
                                                    ┌───────────────────────────────────────────────────┐
                                                    │                  User Interface                   │
                                                    │  <--- Receives final answer from "respond"       │
                                                    └───────────────────────────────────────────────────┘


                                                 ( Parallel Pipeline for Document Indexing )
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                                │
│    ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│    │                                              mirror-agent "IndexGraph"                                                  │  │
│    │               (Defined in src/retrieval_graph/index_graph.py, compiled to `index_graph`)                                │  │
│    │                                                                                                                         │  │
│    │  Flow Node (StateGraph with input=IndexState):                                                                          │  │
│    │                                                                                                                         │  │
│    │     "__start__"  ─────>  "index_docs"(index_docs function)  ───────────────────────────────────────────────────────────▶  │
│    │               - Takes state.docs (Sequence[Document])                                                                   │  │
│    │               - Ensures user_id is in metadata (ensure_docs_have_user_id(...))                                          │  │
│    │               - Adds them to the vector store (retriever.aadd_documents(...))                                           │  │
│    │               - Returns {"docs": "delete"} to signal clearing docs from state                                           │  │
│    │                                                                                                                         │  │
│    └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                                │
│    ┌───────────────────────────────────────────── IndexState (from state.py) ─────────────────────────────────────────────────┐  │
│    │  @dataclass                                                                                                            │  │
│    │  class IndexState:                                                                                                     │  │
│    │     docs: Annotated[Sequence[Document], reduce_docs]                                                                   │  │
│    └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                                │
│       (Example flow: Obsidian Monitor or external app triggers indexing → sends documents → index_graph → makes them queryable) │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


                    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                    │                                       Configuration Classes                                           │
                    │                            (from configuration.py : IndexConfiguration, Configuration)                │
                    │   - Both store settings such as:                                                                     │
                    │        user_id, embedding_model, retriever_provider, search_kwargs, etc.                              │
                    │   - Configuration extends IndexConfiguration: adds LLM model names, system prompts                    │
                    │   - from_runnable_config(...) merges a RunnableConfig with these dataclass fields                    │
                    └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                    │                                     Utility Functions (utils.py)                                       │
                    │   - get_message_text(msg): extracts string content from AnyMessage                                   │
                    │   - format_docs(docs): returns an <documents> ... </documents> XML wrapper for LLM consumption         │
                    │   - load_chat_model(name): creates a BaseChatModel from e.g. openai/gpt-4o                             │
                    └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘


┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│     Additional Tools & Context:                                                                                                 │
│     - Environment variables used for credentials (e.g. ELASTICSEARCH_API_KEY, PINECONE_INDEX_NAME, MONGODB_URI, etc.)           │
│     - Logging and local "tests/" show how to run with ephemeral "elastic-local"                                                 │
│     - docs/architecture.md outlines high-level Mirror Agent architecture with Obsidian -> Indexer -> Pinecone -> Retrieval      │
│     - The entire system is built on top of langchain_core, langgraph, and supports advanced config for embeddings & providers    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


   Summary of Key Interactions:
   1. **User** sends a message (or doc for indexing).
   2. **IndexGraph** (if docs) or **RetrievalGraph** (if user question).
   3. For retrieval:
       - Node "should_retrieve" decides if we skip retrieval or proceed to "generate_query" → "retrieve".
       - The "retrieve" node uses make_retriever(...) which depends on user’s embedding & vector DB config.
       - "respond" composes final answer from conversation history & retrieved docs.
   4. For indexing:
       - "index_docs" ensures user_id is in doc metadata, then calls retriever.aadd_documents(...) to store them.
       - The doc database (e.g., Pinecone) is updated, so subsequent retrieval queries can reference these new docs.
   5. Configuration classes unify user settings, embedding model, and retriever providers for both pipelines.
   6. Utility modules handle prompts, message formatting, date/time, and doc processing, hooking into the LLM.