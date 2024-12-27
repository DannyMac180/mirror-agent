# https://python.langchain.com/docs/integrations/vectorstores/pinecone/ Documentation

Source: https://python.langchain.com/docs/integrations/vectorstores/pinecone/

ComponentsVector storesPineconeOn this pagePinecone

Pinecone is a vector database with broad functionality.

This notebook shows how to use functionality related to the Pinecone vector database.
Setup​
To use the PineconeVectorStore you first need to install the partner package, as well as the other packages used throughout this notebook.
%pip install -qU langchain-pinecone pinecone-notebooks
Migration note: if you are migrating from the langchain_community.vectorstores implementation of Pinecone, you may need to remove your pinecone-client v2 dependency before installing langchain-pinecone, which relies on pinecone-client v3.
Credentials​
Create a new Pinecone account, or sign into your existing one, and create an API key to use in this notebook.
import getpassimport osimport timefrom pinecone import Pinecone, ServerlessSpecif not os.getenv("PINECONE_API_KEY"):    os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")pinecone_api_key = os.environ.get("PINECONE_API_KEY")pc = Pinecone(api_key=pinecone_api_key)
If you want to get automated tracing of your model calls you can also set your LangSmith API key by uncommenting below:
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")# os.environ["LANGSMITH_TRACING"] = "true"
Initialization​
Before initializing our vector store, let's connect to a Pinecone index. If one named index_name doesn't exist, it will be created.
import timeindex_name = "langchain-test-index"  # change if desiredexisting_indexes = [index_info["name"] for index_info in pc.list_indexes()]if index_name not in existing_indexes:    pc.create_index(        name=index_name,        dimension=3072,        metric="cosine",        spec=ServerlessSpec(cloud="aws", region="us-east-1"),    )    while not pc.describe_index(index_name).status["ready"]:        time.sleep(1)index = pc.Index(index_name)
Now that our Pinecone index is setup, we can initialize our vector store.

Select embeddings model:OpenAI▾OpenAIAzureGoogleAWSHuggingFaceOllamaCohereMistralAINomicNVIDIAVoyage AIFakepip install -qU langchain-openaiimport getpassimport osif not os.environ.get("OPENAI_API_KEY"):  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")from langchain_openai import OpenAIEmbeddingsembeddings = OpenAIEmbeddings(model="text-embedding-3-large")
from langchain_pinecone import PineconeVectorStorevector_store = PineconeVectorStore(index=index, embedding=embeddings)API Reference:PineconeVectorStore
Manage vector store​
Once you have created your vector store, we can interact with it by adding and deleting different items.
Add items to vector store​
We can add items to our vector store by using the add_documents function.
from uuid import uuid4from langchain_core.documents import Documentdocument_1 = Document(    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",    metadata={"source": "tweet"},)document_2 = Document(    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",    metadata={"source": "news"},)document_3 = Document(    page_content="Building an exciting new project with LangChain - come check it out!",    metadata={"source": "tweet"},)document_4 = Document(    page_content="Robbers broke into the city bank and stole $1 million in cash.",    metadata={"source": "news"},)document_5 = Document(    page_content="Wow! That was an amazing movie. I can't wait to see it again.",    metadata={"source": "tweet"},)document_6 = Document(    page_content="Is the new iPhone worth the price? Read this review to find out.",    metadata={"source": "website"},)document_7 = Document(    page_content="The top 10 soccer players in the world right now.",    metadata={"source": "website"},)document_8 = Document(    page_content="LangGraph is the best framework for building stateful, agentic applications!",    metadata={"source": "tweet"},)document_9 = Document(    page_content="The stock market is down 500 points today due to fears of a recession.",    metadata={"source": "news"},)document_10 = Document(    page_content="I have a bad feeling I am going to get deleted :(",    metadata={"source": "tweet"},)documents = [    document_1,    document_2,    document_3,    document_4,    document_5,    document_6,    document_7,    document_8,    document_9,    document_10,]uuids = [str(uuid4()) for _ in range(len(documents))]vector_store.add_documents(documents=documents, ids=uuids)API Reference:Document
['167b8681-5974-467f-adcb-6e987a18df01', 'd16010fd-41f8-4d49-9c22-c66d5555a3fe', 'ffcacfb3-2bc2-44c3-a039-c2256a905c0e', 'cf3bfc9f-5dc7-4f5e-bb41-edb957394126', 'e99b07eb-fdff-4cb9-baa8-619fd8efeed3', '68c93033-a24f-40bd-8492-92fa26b631a4', 'b27a4ecb-b505-4c5d-89ff-526e3d103558', '4868a9e6-e6fb-4079-b400-4a1dfbf0d4c4', '921c0e9c-0550-4eb5-9a6c-ed44410788b2', 'c446fc23-64e8-47e7-8c19-ecf985e9411e']
Delete items from vector store​
vector_store.delete(ids=[uuids[-1]])
Query vector store​
Once your vector store has been created and the relevant documents have been added you will most likely wish to query it during the running of your chain or agent.
Query directly​
Performing a simple similarity search can be done as follows:
results = vector_store.similarity_search(    "LangChain provides abstractions to make working with LLMs easy",    k=2,    filter={"source": "tweet"},)for res in results:    print(f"* {res.page_content} [{res.metadata}]")
* Building an exciting new project with LangChain - come check it out! [{'source': 'tweet'}]* LangGraph is the best framework for building stateful, agentic applications! [{'source': 'tweet'}]
Similarity search with score​
You can also search with score:
results = vector_store.similarity_search_with_score(    "Will it be hot tomorrow?", k=1, filter={"source": "news"})for res, score in results:    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
* [SIM=0.553187] The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees. [{'source': 'news'}]
Other search methods​
There are more search methods (such as MMR) not listed in this notebook, to find all of them be sure to read the API reference.
Query by turning into retriever​
You can also transform the vector store into a retriever for easier usage in your chains.
retriever = vector_store.as_retriever(    search_type="similarity_score_threshold",    search_kwargs={"k": 1, "score_threshold": 0.5},)retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
[Document(metadata={'source': 'news'}, page_content='Robbers broke into the city bank and stole $1 million in cash.')]
Usage for retrieval-augmented generation​
For guides on how to use this vector store for retrieval-augmented generation (RAG), see the following sections:

Tutorials
How-to: Question and answer with RAG
Retrieval conceptual docs

API reference​
For detailed documentation of all __ModuleName__VectorStore features and configurations head to the API reference: https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html
Related​

Vector store conceptual guide
Vector store how-to guides
Edit this page

# API reference​

## PineconeVectorStore Class Reference

_class_ `PineconeVectorStore(index=None, embedding=None, text_key='text', namespace=None, distance_strategy=DistanceStrategy.COSINE, *, pinecone_api_key=None, index_name=None)`

Pinecone vector store integration for LangChain.

### Key Attributes

| Attribute | Description |
|-----------|-------------|
| embeddings | Access the query embedding object if available |

### Key Methods

#### Initialization Methods

- `__init__(index=None, embedding=None, text_key='text', namespace=None, distance_strategy=DistanceStrategy.COSINE, *, pinecone_api_key=None, index_name=None)`
  - Initialize the vector store with Pinecone index and embedding function

- `from_existing_index(embedding, index_name, text_key='text', namespace=None, pinecone_api_key=None)`
  - Create a vector store from an existing Pinecone index

#### Document Management

- `add_documents(documents, **kwargs)`
  - Add or update documents in the vectorstore
  
- `add_texts(texts, metadatas=None, ids=None, namespace=None)`
  - Run texts through embeddings and add to vectorstore

- `delete(ids=None)`
  - Delete by vector ID or other criteria

#### Search Methods

- `similarity_search(query, k=4, filter=None, namespace=None, **kwargs)`
  - Return documents most similar to query
  
- `similarity_search_with_score(query, k=4, filter=None, namespace=None)`
  - Return documents most similar to query with scores

- `max_marginal_relevance_search(query, k=4, fetch_k=20, lambda_mult=0.5, filter=None, namespace=None)`
  - Return docs selected using maximal marginal relevance

- `similarity_search_by_vector(embedding, k=4, **kwargs)`
  - Return docs most similar to embedding vector

#### Async Methods

- `aadd_documents(documents, **kwargs)`
- `aadd_texts(texts, metadatas=None, ids=None)`
- `adelete(ids=None)`
- `asimilarity_search(query, k=4, filter=None, namespace=None)`
- `asimilarity_search_with_score(query, k=4, filter=None, namespace=None)`

### Usage Example

```python
import time
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create or connect to index
index_name = "langchain-test-index"
index = pc.Index(index_name)

# Initialize vector store
vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(),
    namespace="my-namespace"
)

# Add documents
from langchain_core.documents import Document
docs = [Document(page_content="content", metadata={"key": "value"})]
vector_store.add_documents(documents=docs)

# Search
results = vector_store.similarity_search("query", k=3)
```

For detailed documentation of all features and configurations, visit the [API reference](https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html).

## Related​

- Vector store conceptual guide
- Vector store how-to guides