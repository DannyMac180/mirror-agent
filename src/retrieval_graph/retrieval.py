"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, MongoDB, and Chroma.
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional, List, Dict, Any
import pickle

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import PrivateAttr
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from cohere import Client

from retrieval_graph.configuration import Configuration, IndexConfiguration  # noqa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize CohereRerank compressor globally
try:
    cohere_reranker = CohereRerank(model="rerank-english-v3.0")  # Ensure COHERE_API_KEY is set
    logging.info("CohereRerank compressor initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing CohereRerank: {e}", exc_info=True)
    cohere_reranker = None  # Handle cases where CohereRerank fails to initialize

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mirror-agent")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_embeddings_model(model_name: str) -> HuggingFaceEmbeddings:
    """
    Check if the embeddings model is cached locally; if not, download and cache it.
    """
    cache_path = os.path.join(CACHE_DIR, f"{model_name.replace('/', '_')}_embeddings.bin")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
            logging.info(f"Loaded cached embeddings model from {cache_path}")
            return embeddings
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
            logging.info(f"Cached embeddings model to {cache_path}")
        return embeddings

def create_compressed_retriever(base_retriever):
    """
    Wraps a base retriever with ContextualCompressionRetriever using CohereRerank.

    Args:
        base_retriever: The base retriever to wrap.

    Returns:
        ContextualCompressionRetriever: The retriever wrapped with CohereRerank,
                                      or the base retriever if CohereRerank is not initialized.
    """
    if cohere_reranker:
        try:
            compressed_retriever = ContextualCompressionRetriever(
                base_compressor=cohere_reranker,
                base_retriever=base_retriever
            )
            logging.info(f"Successfully wrapped retriever {type(base_retriever).__name__} with ContextualCompressionRetriever.")
            return compressed_retriever
        except Exception as e:
            logging.error(f"Error wrapping retriever with ContextualCompressionRetriever: {e}", exc_info=True)
            logging.warning("Falling back to the base retriever without compression.")
            return base_retriever  # Fallback to base retriever if compression fails
    else:
        logging.warning("CohereRerank not initialized. Returning base retriever without compression.")
        return base_retriever  # Fallback to base retriever if CohereRerank is not available

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
        case "BAAI":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            import torch
            device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")

## Retriever constructors
def _ensure_env_var_set(var_name: str) -> str:
    """
    Helper to ensure that a required environment variable is set.
    Raises a ValueError if missing.
    """
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}"
        )
    return value

def _check_elastic_env(retriever_provider: str) -> None:
    if retriever_provider == "elastic-local":
        _ensure_env_var_set("ELASTICSEARCH_USER")
        _ensure_env_var_set("ELASTICSEARCH_PASSWORD")
    else:
        _ensure_env_var_set("ELASTICSEARCH_API_KEY")
    _ensure_env_var_set("ELASTICSEARCH_URL")

@contextmanager
def make_elastic_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific elastic index."""
    try:
        from langchain_elasticsearch import ElasticsearchStore
        _check_elastic_env(configuration.retriever_provider)

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

        base_retriever = vstore.as_retriever(search_kwargs=configuration.search_kwargs)
        logging.info("Elasticsearch retriever initialized.")
        yield create_compressed_retriever(base_retriever)
    except Exception as e:
        logging.error(f"Error initializing Elasticsearch retriever: {e}", exc_info=True)
        yield None

def _check_pinecone_env() -> None:
    _ensure_env_var_set("PINECONE_INDEX_NAME")
    _ensure_env_var_set("PINECONE_API_KEY")  # if needed by your usage

@contextmanager
def make_pinecone_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific pinecone index."""
    try:
        from langchain_pinecone import PineconeVectorStore
        _check_pinecone_env()

        vstore = PineconeVectorStore.from_existing_index(
            os.environ["PINECONE_INDEX_NAME"], embedding=embedding_model
        )
        base_retriever = vstore.as_retriever(search_kwargs=configuration.search_kwargs)
        logging.info("Pinecone retriever initialized.")
        yield create_compressed_retriever(base_retriever)
    except Exception as e:
        logging.error(f"Error initializing Pinecone retriever: {e}", exc_info=True)
        yield None

def _check_mongodb_env() -> None:
    _ensure_env_var_set("MONGODB_URI")

@contextmanager
def make_mongodb_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
    try:
        from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
        _check_mongodb_env()

        vstore = MongoDBAtlasVectorSearch.from_connection_string(
            os.environ["MONGODB_URI"],
            namespace="langgraph_retrieval_agent.default",
            embedding=embedding_model,
        )
        base_retriever = vstore.as_retriever(search_kwargs=configuration.search_kwargs)
        logging.info("MongoDB Atlas Vector Search retriever initialized.")
        yield create_compressed_retriever(base_retriever)
    except Exception as e:
        logging.error(f"Error initializing MongoDB Atlas Vector Search retriever: {e}", exc_info=True)
        yield None

@contextmanager
def make_chroma_retriever(
    configuration: IndexConfiguration, embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    try:
        chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")
        chroma_collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "obsidian")

        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if not embedding_model:
            embedding_model = load_embeddings_model("BAAI/bge-large-en-v1.5")

        vstore = Chroma(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=embedding_model,
        )
        
        # Ensure search_kwargs is a dict and set default k=20 if not specified
        search_kwargs = configuration.search_kwargs.copy() if configuration.search_kwargs else {}
        if 'k' not in search_kwargs:
            search_kwargs['k'] = 20  # default for base retrieval prior to rerank
        logging.debug(f"Using search_kwargs: {search_kwargs}")
            
        base_retriever = vstore.as_retriever(search_kwargs=search_kwargs)
        logging.info("Chroma retriever initialized.")
        yield create_compressed_retriever(base_retriever)
        
    except Exception as e:
        logging.error(f"Error creating Chroma retriever: {str(e)}")
        yield None

@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = IndexConfiguration.from_runnable_config(config)

    if configuration.embedding_model is None:
        if configuration.retriever_provider == "chroma":
            default_model = "BAAI/bge-large-en-v1.5"
        else:
            default_model = "openai/text-embedding-ada-002"
        embedding_model = make_text_encoder(default_model)
    else:
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
