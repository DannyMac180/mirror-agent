# Add Chroma Vector Store Retriever

## Input Brainstorm
```
Need to add Chroma as a retriever option in mirror-agent. Should integrate with existing retriever framework and match current patterns for MongoDB, Pinecone, and Elasticsearch implementations. Need to handle configuration, initialization, and proper connection management.
```

## Goal
Add Chroma as a new retriever provider option to mirror-agent, implementing all necessary integration code and configuration updates.

## Return Format
Return complete implementation including:
- New Chroma retriever factory function in `retrieval.py`
- Updates to `IndexConfiguration` to add Chroma as a provider option
- Required environment variable definitions
- Any necessary dependency additions to `requirements.txt`

The implementation should:
- Follow existing patterns for retriever initialization
- Use context managers for proper resource management
- Handle configuration consistently with other providers
- Include proper type hints and docstrings

Please provide full diffs that can be implemented and specify the file location for each diff.

## Warnings
- Ensure proper connection cleanup in context manager
- Handle potential ChromaDB connection errors gracefully
- Consider collection naming strategy for multi-user support
- Verify ChromaDB version compatibility
- Check for any conflicting dependencies

## Context Dump

Current retriever architecture:
1. Main retriever factory in `src/retrieval_graph/retrieval.py`:
```python
@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    configuration = IndexConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever
        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever
        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever
```

2. Provider configuration in `src/retrieval_graph/configuration.py`:
```python
retriever_provider: Annotated[
    Literal["elastic", "elastic-local", "pinecone", "mongodb"],
    {"__template_metadata__": {"kind": "retriever"}},
] = field(
    default="pinecone",
    metadata={
        "description": "The vector store provider to use for retrieval."
    },
)
```

3. Existing provider implementations follow context manager pattern:
```python
@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    connection_options = {}
    # ... connection setup ...
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)
```

Key considerations:
- All retrievers must implement VectorStoreRetriever interface
- Configuration is handled through IndexConfiguration class
- Connection management uses context managers
- Environment variables used for sensitive connection details
- Search configuration passed via configuration.search_kwargs
