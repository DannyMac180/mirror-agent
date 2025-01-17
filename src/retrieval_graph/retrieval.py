"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, MongoDB, and Chroma.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from retrieval_graph.configuration import Configuration, IndexConfiguration  # noqa

## Encoder constructors
def make_text_encoder(model: Optional[str]) -> Embeddings:
    """Connect to the configured text encoder.

    Expects 'model' in the form "provider/model-name", for example:
        "openai/text-embedding-3-small"
        "cohere/embed-english-light-v2"
    Raises ValueError otherwise.
    """
    if not model:
        raise ValueError(
            "No embedding_model was provided, but it is required for non-Chroma retrievers. "
            "Either specify something like 'openai/text-embedding-ada-002' or set retriever_provider='chroma'."
        )

    if "/" not in model:
        raise ValueError(
            f"Expected embedding_model in 'provider/model' format, got '{model}'. "
            "For example: 'openai/text-embedding-ada-002'."
        )

    provider, model_name = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name)
        case "cohere":
            from langchain_cohere import CohereEmbeddings

            return CohereEmbeddings(model=model_name)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Retriever constructors


@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    from langchain_elasticsearch import ElasticsearchStore

    connection_options = {}
    if configuration.retriever_provider == "elastic-local":
        connection_options = {
            "es_user": os.environ["ELASTICSEARCH_USER"],
            "es_password": os.environ["ELASTICSEARCH_PASSWORD"],
        }

    else:
        connection_options = {"es_api_key": os.environ["ELASTICSEARCH_API_KEY"]}

    vstore = ElasticsearchStore(
        **connection_options,  # type: ignore
        es_url=os.environ["ELASTICSEARCH_URL"],
        index_name="langchain_index",
        embedding=embedding_model,
    )

    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    from langchain_pinecone import PineconeVectorStore

    vstore = PineconeVectorStore.from_existing_index(
        os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

    vstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        namespace="langgraph_retrieval_agent.default",
        embedding=embedding_model,
    )
    yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)


@contextmanager
def make_chroma_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """
    Configure this agent to connect to a Chroma (ChromaDB) vector store,
    using environment variables for connection details.
    """
    try:
        # Use default values if environment variables are not set
        chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")
        chroma_collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "obsidian")

        # Import the Chroma-based vector store
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        # Override embedding model to match obsidian_indexer if not provided
        if not embedding_model:
            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # Initialize the Chroma vector store
        vstore = Chroma(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=embedding_model,
        )

        yield vstore.as_retriever(search_kwargs=configuration.search_kwargs)

    except Exception as e:
        logging.error("Failed to initialize Chroma retriever: %s", e)
        raise


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)

    # Only call make_text_encoder if we actually need an embedding model
    # i.e., if we're NOT using "chroma" or "elastic-local"/"elastic"/"pinecone"/"mongodb"
    # For "chroma", we handle the None case inside make_chroma_retriever.
    if configuration.retriever_provider == "chroma":
        embedding_model = None
    else:
        # For providers other than Chroma, we require a valid embedding_model
        embedding_model = make_text_encoder(configuration.embedding_model)

    match configuration.retriever_provider:
        case "elastic" | "elastic-local":
            with make_elastic_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "chroma":
            with make_chroma_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "pinecone":
            with make_pinecone_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case "mongodb":
            with make_mongodb_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(Configuration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
