================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/collections/add-data.md
================================================
# Adding Data to Chroma Collections

Add data to Chroma with `.add`.

Raw documents:

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents: ["lorem ipsum...", "doc2", "doc3", ...],
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

If Chroma is passed a list of `documents`, it will automatically tokenize and embed them with the collection's embedding function (the default will be used if none was supplied at collection creation). Chroma will also store the `documents` themselves. If the documents are too large to embed using the chosen embedding function, an exception will be raised.

Each document must have a unique associated `id`. Trying to `.add` the same ID twice will result in only the initial value being stored. An optional list of `metadata` dictionaries can be supplied for each document, to store additional information and enable filtering.

Alternatively, you can supply a list of document-associated `embeddings` directly, and Chroma will store the associated documents without embedding them itself.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.add(
    documents=["doc1", "doc2", "doc3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents: ["lorem ipsum...", "doc2", "doc3", ...],
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

If the supplied `embeddings` are not the same dimension as the collection, an exception will be raised.

You can also store documents elsewhere, and just supply a list of `embeddings` and `metadata` to Chroma. You can use the `ids` to associate the embeddings with your documents stored elsewhere.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.add(
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
})
```
{% /Tab %}

{% /TabbedCodeBlock %}


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/collections/configure.md
================================================
# Configuring Chroma Collections

You can configure the embedding space of a collection by setting special keys on a collection's metadata. These configurations will help you customize your Chroma collections for different data, accuracy and performance requirements.

* `hnsw:space` defines the distance function of the embedding space. The default is `l2` (squared L2 norm), and other possible values are `cosine` (cosine similarity), and `ip` (inner product).

| Distance          | parameter |                                                                                                                                                   Equation |
| ----------------- | :-------: |-----------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Squared L2        |   `l2`    |                                                                                                {% Latex %} d =  \\sum\\left(A_i-B_i\\right)^2 {% /Latex %} |
| Inner product     |   `ip`    |                                                                                     {% Latex %} d = 1.0 - \\sum\\left(A_i \\times B_i\\right) {% /Latex %} |
| Cosine similarity | `cosine`  | {% Latex %} d = 1.0 - \\frac{\\sum\\left(A_i \\times B_i\\right)}{\\sqrt{\\sum\\left(A_i^2\\right)} \\cdot \\sqrt{\\sum\\left(B_i^2\\right)}} {% /Latex %} |

* `hnsw:construction_ef` determines the size of the candidate list used to select neighbors during index creation. A higher value improves index quality at the cost of more memory and time, while a lower value speeds up construction with reduced accuracy. The default value is `100`.
* `hnsw:search_ef` determines the size of the dynamic candidate list used while searching for the nearest neighbors. A higher value improves recall and accuracy by exploring more potential neighbors but increases query time and computational cost, while a lower value results in faster but less accurate searches. The default value is `100`.
* `hnsw:M` is the maximum number of neighbors (connections) that each node in the graph can have during the construction of the index. A higher value results in a denser graph, leading to better recall and accuracy during searches but increases memory usage and construction time. A lower value creates a sparser graph, reducing memory usage and construction time but at the cost of lower search accuracy and recall. The default value is `16`.
* `hnsw:num_threads` specifies the number of threads to use during index construction or search operations. The default value is `multiprocessing.cpu_count()` (available CPU cores).

Here is an example of how you can create a collection and configure it with custom HNSW settings:

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection = client.create_collection(
    name="my_collection", 
    embedding_function=emb_fn,
    metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 100
    }
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
let collection = await client.createCollection({
    name: "my_collection",
    embeddingFunction: emb_fn,
    metadata: {
        "hnsw:space": "cosine",
        "hnsw:search_ef": 100
    }
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

You can learn more in our [Embeddings section](../embeddings/embedding-functions).

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/collections/create-get-delete.md
================================================
# Create, Get, and Delete Chroma Collections

Chroma lets you manage collections of embeddings, using the `collection` primitive.

Chroma uses collection names in the url, so there are a few restrictions on naming them:

- The length of the name must be between 3 and 63 characters.
- The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.
- The name must not contain two consecutive dots.
- The name must not be a valid IP address.

Chroma collections are created with a name and an optional embedding function.

{% Banner type="note" %}
If you supply an embedding function, you must supply it every time you get the collection.
{% /Banner %}

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
let collection = await client.createCollection({
    name: "my_collection",
    embeddingFunction: emb_fn,
});

collection = await client.getCollection({
    name: "my_collection",
    embeddingFunction: emb_fn,
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

The embedding function takes text as input and embeds it. If no embedding function is supplied, Chroma will use [sentence transformer](https://www.sbert.net/index.html) as a default. You can learn more about [embedding functions](../embeddings/embedding-functions), and how to create your own.

When creating collections, you can pass the optional `metadata` argument to add a mapping of metadata key-value pairs to your collections. This can be useful for adding general about the collection like creation time, description of the data stored in the collection, and more.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
from datetime import datetime

collection = client.create_collection(
    name="my_collection", 
    embedding_function=emb_fn,
    metadata={
        "description": "my first Chroma collection",
        "created": str(datetime.now())
    }  
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
let collection = await client.createCollection({
    name: "my_collection",
    embeddingFunction: emb_fn,
    metadata: {
        description: "my first Chroma collection",
        created: (new Date()).toString()
    }
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

The collection metadata is also used to configure the embedding space of a collection. Learn more about it in [Configuring Chroma Collections](./configure).

The Chroma client allows you to get and delete existing collections by their name. It also offers a `get or create` method to get a collection if it exists, or create it otherwise.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection = client.get_collection(name="test") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
const collection = await client.getCollection({ name: "test" }); // Get a collection object from an existing collection, by name. Will raise an exception of it's not found.
collection = await client.getOrCreateCollection({ name: "test" }); // Get a collection object from an existing collection, by name. If it doesn't exist, create it.
await client.deleteCollection(collection); // Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
```
{% /Tab %}

{% /TabbedCodeBlock %}

Collections have a few useful convenience methods.

* `peek()` - returns a list of the first 10 items in the collection.
* `count()` - returns the number of items in the collection.
* `modify()` - rename the collection

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.peek() 
collection.count() 
collection.modify(name="new_name")
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.peek();
await collection.count();
await collection.modify({ name: "new_name" })
```
{% /Tab %}

{% /TabbedCodeBlock %}

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/collections/delete-data.md
================================================
# Deleting Data from Chroma Collections

Chroma supports deleting items from a collection by `id` using `.delete`. The embeddings, documents, and metadata associated with each item will be deleted.

{% Banner type="warn" %}
Naturally, this is a destructive operation, and cannot be undone.
{% /Banner %}

`.delete` also supports the `where` filter. If no `ids` are supplied, it will delete all items in the collection that match the `where` filter.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.delete(
    ids=["id1", "id2", "id3",...],
	where={"chapter": "20"}
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.delete({
    ids: ["id1", "id2", "id3",...], //ids
    where: {"chapter": "20"} //where
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/collections/update-data.md
================================================
# Updating Data in Chroma Collections

Any property of records in a collection can be updated with `.update`:

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.update({
    ids: ["id1", "id2", "id3", ...], 
    embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...], 
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...], 
    documents: ["doc1", "doc2", "doc3", ...]
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

If an `id` is not found in the collection, an error will be logged and the update will be ignored. If `documents` are supplied without corresponding `embeddings`, the embeddings will be recomputed with the collection's embedding function.

If the supplied `embeddings` are not the same dimension as the collection, an exception will be raised.

Chroma also supports an `upsert` operation, which updates existing items, or adds them if they don't yet exist.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.upsert(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.upsert({
    ids: ["id1", "id2", "id3"],
    embeddings: [
        [1.1, 2.3, 3.2],
        [4.5, 6.9, 4.4],
        [1.1, 2.3, 3.2],
    ],
    metadatas: [
        { chapter: "3", verse: "16" },
        { chapter: "3", verse: "5" },
        { chapter: "29", verse: "11" },
    ],
    documents: ["doc1", "doc2", "doc3"],
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

If an `id` is not present in the collection, the corresponding items will be created as per `add`. Items with existing `id`s will be updated as per `update`.

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/embeddings/embedding-functions.md
================================================
# Embedding Functions

Embeddings are the way to represent any kind of data, making them the perfect fit for working with all kinds of A.I-powered tools and algorithms. They can represent text, images, and soon audio and video. There are many options for creating embeddings, whether locally using an installed library, or by calling an API.

Chroma provides lightweight wrappers around popular embedding providers, making it easy to use them in your apps. You can set an embedding function when you create a Chroma collection, which will be used automatically, or you can call them directly yourself.

|                                                                                          | Python | Typescript |
|------------------------------------------------------------------------------------------|--------|------------|
| [OpenAI](../../integrations/embedding-models/openai)                                     | ✓      | ✓          |
| [Google Generative AI](../../integrations/embedding-models/google-gemini)                | ✓      | ✓          |
| [Cohere](../../integrations/embedding-models/cohere)                                     | ✓      | ✓          |
| [Hugging Face](../../integrations/embedding-models/hugging-face)                         | ✓      | -          |
| [Instructor](../../integrations/embedding-models/instructor)                             | ✓      | -          |
| [Hugging Face Embedding Server](../../integrations/embedding-models/hugging-face-server) | ✓      | ✓          |
| [Jina AI](../../integrations/embedding-models/jinaai)                                    | ✓      | ✓          |

We welcome pull requests to add new Embedding Functions to the community.

***

## Default: all-MiniLM-L6-v2

By default, Chroma uses the [Sentence Transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` model to create embeddings. This embedding model can create sentence and document embeddings that can be used for a wide variety of tasks. This embedding function runs locally on your machine, and may require you download the model files (this will happen automatically).

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
import { DefaultEmbeddingFunction } from "chromadb";
const defaultEF = new DefaultEmbeddingFunction();
```
{% /Tab %}

{% /TabbedCodeBlock %}

Embedding functions can be linked to a collection and used whenever you call `add`, `update`, `upsert` or `query`. You can also use them directly which can be handy for debugging.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
val = default_ef(["foo"])
print(val) # [[0.05035809800028801, 0.0626462921500206, -0.061827320605516434...]]
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
const val = defaultEf.generate(["foo"]);
console.log(val); // [[0.05035809800028801, 0.0626462921500206, -0.061827320605516434...]]
```
{% /Tab %}

{% /TabbedCodeBlock %}

## Sentence Transformers

Chroma can also use any [Sentence Transformers](https://www.sbert.net/) model to create embeddings.

You can pass in an optional `model_name` argument, which lets you choose which Sentence Transformers model to use. By default, Chroma uses `all-MiniLM-L6-v2`. You can see a list of all available models [here](https://www.sbert.net/docs/pretrained_models.html).

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
import { DefaultEmbeddingFunction } from "chromadb";
const modelName = "all-MiniLM-L6-v2";
const defaultEF = new DefaultEmbeddingFunction(modelName);
```
{% /Tab %}

{% /TabbedCodeBlock %}

## Custom Embedding Functions

You can create your own embedding function to use with Chroma, it just needs to implement the `EmbeddingFunction` protocol.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
class MyEmbeddingFunction {
    private api_key: string;

    constructor(api_key: string) {
        this.api_key = api_key;
    }

    public async generate(texts: string[]): Promise<number[][]> {
        // do things to turn texts into embeddings with an api_key perhaps
        return embeddings;
    }
}
```
{% /Tab %}

{% /TabbedCodeBlock %}

We welcome contributions! If you create an embedding function that you think would be useful to others, please consider [submitting a pull request](https://github.com/chroma-core/chroma) to add it to Chroma's `embedding_functions` module.


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/embeddings/multimodal.md
================================================
# Multimodal

{% Banner type="note" %}
Multimodal support is currently available only in Python. Javascript/Typescript support coming soon! 
{% /Banner %}

Chroma supports multimodal collections, i.e. collections which can store, and can be queried by, multiple modalities of data.

[Try it out in Colab](https://githubtocolab.com/chroma-core/chroma/blob/main/examples/multimodal/multimodal_retrieval.ipynb)

## Multi-modal Embedding Functions

Chroma supports multi-modal embedding functions, which can be used to embed data from multiple modalities into a single embedding space.

Chroma has the OpenCLIP embedding function built in, which supports both text and images.

```python
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()
```

## Data Loaders

Chroma supports data loaders, for storing and querying with data stored outside Chroma itself, via URI. Chroma will not store this data, but will instead store the URI, and load the data from the URI when needed.

Chroma has a data loader for loading images from a filesystem built in.

```python
from chromadb.utils.data_loaders import ImageLoader
data_loader = ImageLoader()
```

## Multi-modal Collections

You can create a multi-modal collection by passing in a multi-modal embedding function. In order to load data from a URI, you must also pass in a data loader.

```python
import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader)

```

### Adding data

You can add data to a multi-modal collection by specifying the data modality. For now, images are supported:

```python
collection.add(
    ids=['id1', 'id2', 'id3'],
    images=[...] # A list of numpy arrays representing images
)
```

Note that Chroma will not store the data for you, and you will have to maintain a mapping from IDs to data yourself.

However, you can use Chroma in combination with data stored elsewhere, by adding it via URI. Note that this requires that you have specified a data loader when creating the collection.

```python
collection.add(
    ids=['id1', 'id2', 'id3'],
    uris=[...] #  A list of strings representing URIs to data
)
```

Since the embedding function is multi-modal, you can also add text to the same collection:

```python
collection.add(
    ids=['id4', 'id5', 'id6'],
    documents=["This is a document", "This is another document", "This is a third document"]
)
```

### Querying

You can query a multi-modal collection with any of the modalities that it supports. For example, you can query with images:

```python
results = collection.query(
    query_images=[...] # A list of numpy arrays representing images
)
```

Or with text:

```python
results = collection.query(
    query_texts=["This is a query document", "This is another query document"]
)
```

If a data loader is set for the collection, you can also query with URIs which reference data stored elsewhere of the supported modalities:

```python
results = collection.query(
    query_uris=[...] # A list of strings representing URIs to data
)
```

Additionally, if a data loader is set for the collection, and URIs are available, you can include the data in the results:

```python
results = collection.query(
    query_images=[...], # # list of numpy arrays representing images
    includes=['data']
)
```

This will automatically call the data loader for any available URIs, and include the data in the results. `uris` are also available as an `includes` field.

### Updating

You can update a multi-modal collection by specifying the data modality, in the same way as `add`. For now, images are supported:

```python
collection.update(
    ids=['id1', 'id2', 'id3'],
    images=[...] # A list of numpy arrays representing images
)
```

Note that a given entry with a specific ID can only have one associated modality at a time. Updates will over-write the existing modality, so for example, an entry which originally has corresponding text and updated with an image, will no longer have that text after an update with images.



================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/guides/embeddings-guide.md
================================================
---
{
  "id": "embeddings-guide",
  "title": "Embeddings",
  "section": "Guides",
  "order": 1
}
---

# Embeddings

Embeddings are the A.I-native way to represent any kind of data, making them the perfect fit for working with all kinds of A.I-powered tools and algorithms. They can represent text, images, and soon audio and video. There are many options for creating embeddings, whether locally using an installed library, or by calling an API.

Chroma provides lightweight wrappers around popular embedding providers, making it easy to use them in your apps. You can set an embedding function when you create a Chroma collection, which will be used automatically, or you can call them directly yourself.

{% special_table %}
{% /special_table %}

|              | Python | JS |
|--------------|-----------|---------------|
| [OpenAI](/integrations/openai) | ✅  | ✅ |
| [Google Generative AI](/integrations/google-gemini) | ✅  | ✅ |
| [Cohere](/integrations/cohere) | ✅  | ✅ |
| [Hugging Face](/integrations/hugging-face) | ✅  | ➖ |
| [Instructor](/integrations/instructor) | ✅  | ➖ |
| [Hugging Face Embedding Server](/integrations/hugging-face-server) | ✅  | ✅ |
| [Jina AI](/integrations/jinaai) | ✅  | ✅ |

We welcome pull requests to add new Embedding Functions to the community.

***

## Default: all-MiniLM-L6-v2

By default, Chroma uses the [Sentence Transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` model to create embeddings. This embedding model can create sentence and document embeddings that can be used for a wide variety of tasks. This embedding function runs locally on your machine, and may require you download the model files (this will happen automatically).

```python
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
```

{% note type="default" %}
Embedding functions can be linked to a collection and used whenever you call `add`, `update`, `upsert` or `query`. You can also use them directly which can be handy for debugging.
```py
val = default_ef(["foo"])
```
-> [[0.05035809800028801, 0.0626462921500206, -0.061827320605516434...]]
{% /note %}


{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

## Sentence Transformers

Chroma can also use any [Sentence Transformers](https://www.sbert.net/) model to create embeddings.

```python
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```

You can pass in an optional `model_name` argument, which lets you choose which Sentence Transformers model to use. By default, Chroma uses `all-MiniLM-L6-v2`. You can see a list of all available models [here](https://www.sbert.net/docs/pretrained_models.html).

{% /tab %}
{% tab label="Javascript" %}
{% /tab %}
{% /tabs %}


***


## Custom Embedding Functions

{% tabs group="code-lang" hideContent=true %}

{% tab label="Python" %}
{% /tab %}

{% tab label="Javascript" %}
{% /tab %}

{% /tabs %}

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

You can create your own embedding function to use with Chroma, it just needs to implement the `EmbeddingFunction` protocol.

```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings
```

We welcome contributions! If you create an embedding function that you think would be useful to others, please consider [submitting a pull request](https://github.com/chroma-core/chroma) to add it to Chroma's `embedding_functions` module.


{% /tab %}
{% tab label="Javascript" %}

You can create your own embedding function to use with Chroma, it just needs to implement the `EmbeddingFunction` protocol. The `.generate` method in a class is strictly all you need.

```javascript
class MyEmbeddingFunction {
  private api_key: string;

  constructor(api_key: string) {
    this.api_key = api_key;
  }

  public async generate(texts: string[]): Promise<number[][]> {
    // do things to turn texts into embeddings with an api_key perhaps
    return embeddings;
  }
}
```

{% /tab %}
{% /tabs %}


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/guides/multimodal-guide.md
================================================
---
{
  "id": "multimodal-guide",
  "title": "Multimodal",
  "section": "Guides",
  "order": 2
}
---

# Multimodal

{% tabs group="code-lang" hideContent=true %}

{% tab label="Python" %}
{% /tab %}

{% tab label="Javascript" %}
{% /tab %}

{% /tabs %}

---

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Chroma supports multimodal collections, i.e. collections which can store, and can be queried by, multiple modalities of data.

Try it out in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/chroma-core/chroma/blob/main/examples/multimodal/multimodal_retrieval.ipynb)

## Multi-modal Embedding Functions

Chroma supports multi-modal embedding functions, which can be used to embed data from multiple modalities into a single embedding space.

Chroma has the OpenCLIP embedding function built in, which supports both text and images.

```python
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()
```

## Data Loaders

Chroma supports data loaders, for storing and querying with data stored outside Chroma itself, via URI. Chroma will not store this data, but will instead store the URI, and load the data from the URI when needed.

Chroma has an data loader for loading images from a filesystem built in.

```python
from chromadb.utils.data_loaders import ImageLoader
data_loader = ImageLoader()
```

## Multi-modal Collections

You can create a multi-modal collection by passing in a multi-modal embedding function. In order to load data from a URI, you must also pass in a data loader.

```python
import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader)

```

### Adding data

You can add data to a multi-modal collection by specifying the data modality. For now, images are supported:

```python
collection.add(
    ids=['id1', 'id2', 'id3'],
    images=[...] # A list of numpy arrays representing images
)
```

Note that Chroma will not store the data for you, and you will have to maintain a mapping from IDs to data yourself.

However, you can use Chroma in combination with data stored elsewhere, by adding it via URI. Note that this requires that you have specified a data loader when creating the collection.

```python
collection.add(
    ids=['id1', 'id2', 'id3'],
    uris=[...] #  A list of strings representing URIs to data
)
```

Since the embedding function is multi-modal, you can also add text to the same collection:

```python
collection.add(
    ids=['id4', 'id5', 'id6'],
    documents=["This is a document", "This is another document", "This is a third document"]
)
```

### Querying

You can query a multi-modal collection with any of the modalities that it supports. For example, you can query with images:

```python
results = collection.query(
    query_images=[...] # A list of numpy arrays representing images
)
```

Or with text:

```python
results = collection.query(
    query_texts=["This is a query document", "This is another query document"]
)
```

If a data loader is set for the collection, you can also query with URIs which reference data stored elsewhere of the supported modalities:

```python
results = collection.query(
    query_uris=[...] # A list of strings representing URIs to data
)
```

Additionally, if a data loader is set for the collection, and URIs are available, you can include the data in the results:

```python
results = collection.query(
    query_images=[...], # # list of numpy arrays representing images
    includes=['data']
)
```

This will automatically call the data loader for any available URIs, and include the data in the results. `uris` are also available as an `includes` field.

### Updating

You can update a multi-modal collection by specifying the data modality, in the same way as `add`. For now, images are supported:

```python
collection.update(
    ids=['id1', 'id2', 'id3'],
    images=[...] # A list of numpy arrays representing images
)
```

Note that a given entry with a specific ID can only have one associated modality at a time. Updates will over-write the existing modality, so for example, an entry which originally has corresponding text and updated with an image, will no longer have that text after an update with images.

{% /tab %}
{% tab label="Javascript" %}

Support for multi-modal retrieval for Chroma's JavaScript client is coming soon!

{% /tab %}

{% /tabs %}



================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/guides/usage-guide.md
================================================
---
{
  "id": "usage-guide",
  "title": "Usage Guide",
  "section": "Guides",
  "order": 0
}
---


# Usage Guide


{% tabs group="code-lang" hideContent=true %}

{% tab label="Python" %}
{% /tab %}

{% tab label="Javascript" %}
{% /tab %}

{% /tabs %}

---

## Initiating a persistent Chroma client

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
import chromadb
```

You can configure Chroma to save and load the database from your local machine. Data will be persisted automatically and loaded on start (if it exists).

```python
client = chromadb.PersistentClient(path="/path/to/save/to")
```

The `path` is where Chroma will store its database files on disk, and load them on start.

{% /tab %}
{% tab label="Javascript" %}

```js
// CJS
const { ChromaClient } = require("chromadb");

// ESM
import { ChromaClient } from "chromadb";
```

{% note type="note" title="Connecting to the backend" %}
To connect with the JS client, you must connect to a backend running Chroma. See [Running Chroma in client-server mode](#running-chroma-in-client-server-mode) for how to do this.
{% /note %}

```js
const client = new ChromaClient();
```

{% /tab %}

{% /tabs %}

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

The client object has a few useful convenience methods.

```python
client.heartbeat() # returns a nanosecond heartbeat. Useful for making sure the client remains connected.
client.reset() # Empties and completely resets the database. ⚠️ This is destructive and not reversible.
```

{% /tab %}
{% tab label="Javascript" %}

The client object has a few useful convenience methods.

```javascript
await client.reset() # Empties and completely resets the database. ⚠️ This is destructive and not reversible.
```

{% /tab %}

{% /tabs %}

## Running Chroma in client-server mode

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Chroma can also be configured to run in client/server mode. In this mode, the Chroma client connects to a Chroma server running in a separate process.

To start the Chroma server, run the following command:

```bash
chroma run --path /db_path
```

Then use the Chroma HTTP client to connect to the server:

```python
import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

That's it! Chroma's API will run in `client-server` mode with just this change.

---

Chroma also provides an async HTTP client. The behaviors and method signatures are identical to the synchronous client, but all methods that would block are now async. To use it, call `AsyncHttpClient` instead:

```python
import asyncio
import chromadb

async def main():
    client = await chromadb.AsyncHttpClient()
    collection = await client.create_collection(name="my_collection")

    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )

asyncio.run(main())
```

<!-- #### Run Chroma inside your application

To run the Chroma docker from inside your application code, create a docker-compose file or add to the existing one you have.

1. Download [`docker-compose.server.example.yml`](https://github.com/chroma-core/chroma/blob/main/docker-compose.server.example.yml) file and [`config`](https://github.com/chroma-core/chroma/tree/main/config) folder along with both the files inside from [GitHub Repo](https://github.com/chroma-core/chroma)
2. Rename `docker-compose.server.example.yml` to `docker-compose.yml`
3. Install docker on your local machine. [`Docker Engine`](https://docs.docker.com/engine/install/) or [`Docker Desktop`](https://docs.docker.com/desktop/install/)
4. Install docker compose [`Docker Compose`](https://docs.docker.com/compose/install/)

Use following command to manage Dockerized Chroma:
- __Command to Start Chroma__: `docker-compose up -d`
- __Command to Stop Chroma__: `docker-compose down`
- __Command to Stop Chroma and delete volumes__
This is distructive command. With this command volumes created earlier will be deleted along with data stored.: `docker-compose down -v` -->

#### Using the Python HTTP-only client

If you are running Chroma in client-server mode, you may not need the full Chroma library. Instead, you can use the lightweight client-only library.
In this case, you can install the `chromadb-client` package. This package is a lightweight HTTP client for the server with a minimal dependency footprint.

```python
pip install chromadb-client
```

```python
import chromadb
# Example setup of the client to connect to your chroma server
client = chromadb.HttpClient(host='localhost', port=8000)

# Or for async usage:
async def main():
    client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
```

Note that the `chromadb-client` package is a subset of the full Chroma library and does not include all the dependencies. If you want to use the full Chroma library, you can install the `chromadb` package instead.
Most importantly, there is no default embedding function. If you add() documents without embeddings, you must have manually specified an embedding function and installed the dependencies for it.

{% /tab %}
{% tab label="Javascript" %}

To run Chroma in client server mode, first install the chroma library and CLI via pypi:

```bash
pip install chromadb
```

Then start the Chroma server:

```bash
chroma run --path /db_path
```

The JS client then talks to the chroma server backend.

```js
// CJS
const { ChromaClient } = require("chromadb");

// ESM
import { ChromaClient } from "chromadb";

const client = new ChromaClient();
```

You can also run the Chroma server in a docker container, or deployed to a cloud provider. See the [deployment docs](./deployment.md) for more information.

{% /tab %}

{% /tabs %}

## Using collections

Chroma lets you manage collections of embeddings, using the `collection` primitive.

### Creating, inspecting, and deleting Collections

Chroma uses collection names in the url, so there are a few restrictions on naming them:

- The length of the name must be between 3 and 63 characters.
- The name must start and end with a lowercase letter or a digit, and it can contain dots, dashes, and underscores in between.
- The name must not contain two consecutive dots.
- The name must not be a valid IP address.

Chroma collections are created with a name and an optional embedding function. If you supply an embedding function, you must supply it every time you get the collection.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)
```

{% note type="caution" %}
If you later wish to `get_collection`, you MUST do so with the embedding function you supplied while creating the collection
{% /note %}

The embedding function takes text as input, and performs tokenization and embedding. If no embedding function is supplied, Chroma will use [sentence transformer](https://www.sbert.net/index.html) as a default.

{% /tab %}
{% tab label="Javascript" %}

```js
// CJS
const { ChromaClient } = require("chromadb");

// ESM
import { ChromaClient } from "chromadb";
```

The JS client talks to a chroma server backend. This can run on your local computer or be easily deployed to AWS.

```js
let collection = await client.createCollection({
  name: "my_collection",
  embeddingFunction: emb_fn,
});
let collection2 = await client.getCollection({
  name: "my_collection",
  embeddingFunction: emb_fn,
});
```

{% note type="caution" %}
If you later wish to `getCollection`, you MUST do so with the embedding function you supplied while creating the collection
{% /note %}

The embedding function takes text as input, and performs tokenization and embedding.

{% /tab %}

{% /tabs %}

You can learn more about [🧬 embedding functions](./guides/embeddings), and how to create your own.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Existing collections can be retrieved by name with `.get_collection`, and deleted with `.delete_collection`. You can also use `.get_or_create_collection` to get a collection if it exists, or create it if it doesn't.

```python
collection = client.get_collection(name="test") # Get a collection object from an existing collection, by name. Will raise an exception if it's not found.
collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
```

{% /tab %}
{% tab label="Javascript" %}

Existing collections can be retrieved by name with `.getCollection`, and deleted with `.deleteCollection`.

```javascript
const collection = await client.getCollection({ name: "test" }); // Get a collection object from an existing collection, by name. Will raise an exception of it's not found.
collection = await client.getOrCreateCollection({ name: "test" }); // Get a collection object from an existing collection, by name. If it doesn't exist, create it.
await client.deleteCollection(collection); // Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
```

{% /tab %}

{% /tabs %}

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Collections have a few useful convenience methods.

```python
collection.peek() # returns a list of the first 10 items in the collection
collection.count() # returns the number of items in the collection
collection.modify(name="new_name") # Rename the collection
```

{% /tab %}
{% tab label="Javascript" %}

There are a few useful convenience methods for working with Collections.

```javascript
await collection.peek(); // returns a list of the first 10 items in the collection
await collection.count(); // returns the number of items in the collection
```

{% /tab %}

{% /tabs %}

### Changing the distance function

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

`create_collection` also takes an optional `metadata` argument which can be used to customize the distance method of the embedding space by setting the value of `hnsw:space`.

```python
 collection = client.create_collection(
        name="collection_name",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )
```

{% /tab %}
{% tab label="Javascript" %}

`createCollection` also takes an optional `metadata` argument which can be used to customize the distance method of the embedding space by setting the value of `hnsw:space`

```js
let collection = client.createCollection({
  name: "collection_name",
  metadata: { "hnsw:space": "cosine" },
});
```

{% /tab %}

{% /tabs %}

Valid options for `hnsw:space` are "l2", "ip, "or "cosine". The **default** is "l2" which is the squared L2 norm.

{% special_table %}
{% /special_table %}

| Distance          | parameter |                                                                                                                                                            Equation |
| ----------------- | :-------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Squared L2        |   `l2`    |                                                                                                 {% math latexText="d = \\sum\\left(A_i-B_i\\right)^2" %}{% /math %} |
| Inner product     |   `ip`    |                                                                                    {% math latexText="d = 1.0 - \\sum\\left(A_i \\times B_i\\right) " %}{% /math %} |
| Cosine similarity | `cosine`  | {% math latexText="d = 1.0 - \\frac{\\sum\\left(A_i \\times B_i\\right)}{\\sqrt{\\sum\\left(A_i^2\\right)} \\cdot \\sqrt{\\sum\\left(B_i^2\\right)}}" %}{% /math %} |

### Adding data to a Collection

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Add data to Chroma with `.add`.

Raw documents:

```python
collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```

{% /tab %}
{% tab label="Javascript" %}

Add data to Chroma with `.addRecords`.

Raw documents:

```javascript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents: ["lorem ipsum...", "doc2", "doc3", ...],
})
// input order
// ids - required
// embeddings - optional
// metadata - optional
// documents - optional
```

{% /tab %}

{% /tabs %}

If Chroma is passed a list of `documents`, it will automatically tokenize and embed them with the collection's embedding function (the default will be used if none was supplied at collection creation). Chroma will also store the `documents` themselves. If the documents are too large to embed using the chosen embedding function, an exception will be raised.

Each document must have a unique associated `id`. Trying to `.add` the same ID twice will result in only the initial value being stored. An optional list of `metadata` dictionaries can be supplied for each document, to store additional information and enable filtering.

Alternatively, you can supply a list of document-associated `embeddings` directly, and Chroma will store the associated documents without embedding them itself.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection.add(
    documents=["doc1", "doc2", "doc3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents: ["lorem ipsum...", "doc2", "doc3", ...],
})

```

{% /tab %}

{% /tabs %}

If the supplied `embeddings` are not the same dimension as the collection, an exception will be raised.

You can also store documents elsewhere, and just supply a list of `embeddings` and `metadata` to Chroma. You can use the `ids` to associate the embeddings with your documents stored elsewhere.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection.add(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...]
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
await collection.add({
    ids: ["id1", "id2", "id3", ...],
    embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
})
```

{% /tab %}

{% /tabs %}

### Querying a Collection

You can query by a set of `query_embeddings`.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Chroma collections can be queried in a variety of ways, using the `.query` method.

```python
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```

{% /tab %}
{% tab label="Javascript" %}

Chroma collections can be queried in a variety of ways, using the `.queryRecords` method.

```javascript
const result = await collection.query({
    queryEmbeddings: [[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    nResults: 10,
    where: {"metadata_field": "is_equal_to_this"},
})
// input order
// queryEmbeddings - optional, exactly one of queryEmbeddings and queryTexts must be provided
// queryTexts - optional
// n_results - required
// where - optional
```

{% /tab %}

{% /tabs %}

The query will return the `n_results` closest matches to each `query_embedding`, in order.
An optional `where` filter dictionary can be supplied to filter by the `metadata` associated with each document.
Additionally, an optional `where_document` filter dictionary can be supplied to filter by contents of the document.

If the supplied `query_embeddings` are not the same dimension as the collection, an exception will be raised.

You can also query by a set of `query_texts`. Chroma will first embed each `query_text` with the collection's embedding function, and then perform the query with the generated embedding.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection.query(
    query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```

You can also retrieve items from a collection by `id` using `.get`.

```python
collection.get(
	ids=["id1", "id2", "id3", ...],
	where={"style": "style1"}
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
await collection.query({
    nResults: 10, // n_results
    where: {"metadata_field": "is_equal_to_this"}, // where
    queryTexts: ["doc10", "thus spake zarathustra", ...], // query_text
})
```

You can also retrieve records from a collection by `id` using `.getRecords`.

```javascript
await collection.get( {
	ids: ["id1", "id2", "id3", ...], //ids
	where: {"style": "style1"} // where
})
```

{% /tab %}

{% /tabs %}

`.get` also supports the `where` and `where_document` filters. If no `ids` are supplied, it will return all items in the collection that match the `where` and `where_document` filters.

##### Choosing which data is returned

When using get or query you can use the include parameter to specify which data you want returned - any of `embeddings`, `documents`, `metadatas`, and for query, `distances`. By default, Chroma will return the `documents`, `metadatas` and in the case of query, the `distances` of the results. `embeddings` are excluded by default for performance and the `ids` are always returned. You can specify which of these you want returned by passing an array of included field names to the includes parameter of the query or get method. Note that embeddings will be returned as a 2-d numpy array in `.get` and a python list of 2-d numpy arrays in `.query`.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
# Only get documents and ids
collection.get(
    include=["documents"]
)

collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    include=["documents"]
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
# Only get documents and ids
collection.get(
    {include=["documents"]}
)

collection.get({
    queryEmbeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    include=["documents"]
})
```

{% /tab %}

{% /tabs %}

### Using Where filters

Chroma supports filtering queries by `metadata` and `document` contents. The `where` filter is used to filter by `metadata`, and the `where_document` filter is used to filter by `document` contents.

##### Filtering by metadata

In order to filter on metadata, you must supply a `where` filter dictionary to the query. The dictionary must have the following structure:

```python
{
    "metadata_field": {
        <Operator>: <Value>
    }
}
```

Filtering metadata supports the following operators:

- `$eq` - equal to (string, int, float)
- `$ne` - not equal to (string, int, float)
- `$gt` - greater than (int, float)
- `$gte` - greater than or equal to (int, float)
- `$lt` - less than (int, float)
- `$lte` - less than or equal to (int, float)

Using the $eq operator is equivalent to using the `where` filter.

```python
{
    "metadata_field": "search_string"
}

# is equivalent to

{
    "metadata_field": {
        "$eq": "search_string"
    }
}
```

{% note type="note" %}
Where filters only search embeddings where the key exists. If you search `collection.get(where={"version": {"$ne": 1}})`. Metadata that does not have the key `version` will not be returned.
{% /note %}

##### Filtering by document contents

In order to filter on document contents, you must supply a `where_document` filter dictionary to the query. We support two filtering keys: `$contains` and `$not_contains`. The dictionary must have the following structure:

```python
# Filtering for a search_string
{
    "$contains": "search_string"
}
```

```python
# Filtering for not contains
{
    "$not_contains": "search_string"
}
```

##### Using logical operators

You can also use the logical operators `$and` and `$or` to combine multiple filters.

An `$and` operator will return results that match all of the filters in the list.

```python
{
    "$and": [
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        },
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        }
    ]
}
```

An `$or` operator will return results that match any of the filters in the list.

```python
{
    "$or": [
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        },
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        }
    ]
}
```

##### Using inclusion operators (`$in` and `$nin`)

The following inclusion operators are supported:

- `$in` - a value is in predefined list (string, int, float, bool)
- `$nin` - a value is not in predefined list (string, int, float, bool)

An `$in` operator will return results where the metadata attribute is part of a provided list:

```json
{
  "metadata_field": {
    "$in": ["value1", "value2", "value3"]
  }
}
```

An `$nin` operator will return results where the metadata attribute is not part of a provided list:

```json
{
  "metadata_field": {
    "$nin": ["value1", "value2", "value3"]
  }
}
```

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

{% note type="note" title="Practical examples" %}
For additional examples and a demo how to use the inclusion operators, please see provided notebook [here](https://github.com/chroma-core/chroma/blob/main/examples/basic_functionality/in_not_in_filtering.ipynb)
{% /note %}

{% /tab %}
{% tab label="Javascript" %}
{% /tab %}

{% /tabs %}

### Updating data in a collection

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

Any property of records in a collection can be updated using `.update`.

```python
collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```

{% /tab %}
{% tab label="Javascript" %}

Any property of records in a collection can be updated using `.updateRecords`.

```javascript
collection.update(
    {
      ids: ["id1", "id2", "id3", ...],
      embeddings: [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
      metadatas: [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
      documents: ["doc1", "doc2", "doc3", ...],
    },
)
```

{% /tab %}

{% /tabs %}

If an `id` is not found in the collection, an error will be logged and the update will be ignored. If `documents` are supplied without corresponding `embeddings`, the embeddings will be recomputed with the collection's embedding function.

If the supplied `embeddings` are not the same dimension as the collection, an exception will be raised.

Chroma also supports an `upsert` operation, which updates existing items, or adds them if they don't yet exist.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection.upsert(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
await collection.upsert({
  ids: ["id1", "id2", "id3"],
  embeddings: [
    [1.1, 2.3, 3.2],
    [4.5, 6.9, 4.4],
    [1.1, 2.3, 3.2],
  ],
  metadatas: [
    { chapter: "3", verse: "16" },
    { chapter: "3", verse: "5" },
    { chapter: "29", verse: "11" },
  ],
  documents: ["doc1", "doc2", "doc3"],
});
```

{% /tab %}

{% /tabs %}

If an `id` is not present in the collection, the corresponding items will be created as per `add`. Items with existing `id`s will be updated as per `update`.

### Deleting data from a collection

Chroma supports deleting items from a collection by `id` using `.delete`. The embeddings, documents, and metadata associated with each item will be deleted.
⚠️ Naturally, this is a destructive operation, and cannot be undone.

{% tabs group="code-lang" hideTabs=true %}
{% tab label="Python" %}

```python
collection.delete(
    ids=["id1", "id2", "id3",...],
	where={"chapter": "20"}
)
```

{% /tab %}
{% tab label="Javascript" %}

```javascript
await collection.delete({
    ids: ["id1", "id2", "id3",...], //ids
	where: {"chapter": "20"} //where
})
```

{% /tab %}

{% /tabs %}

`.delete` also supports the `where` filter. If no `ids` are supplied, it will delete all items in the collection that match the `where` filter.


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/about.md
================================================
---
{
  "id": "about",
  "title": "About",
  "section": "Overview",
  "order": 4
}
---

# About

{% Banner type="tip" title="We are hiring" %}
We are hiring software engineers and applied research scientists.
{% /Banner %}

## Who we are

[View open roles](https://careers.trychroma.com/)

Chroma as a project is coordinated by a small team of full-time employees who work at a company also called Chroma.

We work in the sunny Mission District in San Francisco.

Chroma is co-founded by [Jeff Huber](https://twitter.com/jeffreyhuber) (left) and [Anton Troynikov](https://twitter.com/atroyn) (right).

![](/team.JPG)

## Our commitment to open source

Chroma is a company that builds the open-source project also called Chroma.

We are committed to building open source software because we believe in the flourishing of humanity that will be unlocked through the democratization of robust, safe, and aligned AI systems. These tools need to be available to a new developer just starting in ML as well as the organizations that scale ML to millions (and billions) of users. Open source is about expanding the horizon of what’s possible.

Chroma is a _commercial_ open source company. What does that mean? We believe that organizing financially sustainable teams of people to work to manage, push and integrate the project enriches the health of the project and the community.

It is important that our values around this are very clear!

- We are committed to building Chroma as a ubiquitous open source standard
- A successful Chroma-based commercial product is essential for the success of the technology, and is a win-win for everyone. Simply put, many organizations will not adopt Chroma without the option of a commercially hosted solution; and the project must be backed by a company with a viable business model. We want to build an awesome project and an awesome business.
- We will decide what we provide exclusively in the commercial product based on clear, consistent criteria.

What code will be open source? As a general rule, any feature which an individual developer would find useful will be 100% open source forever. This approach, popularized by Gitlab, is called [buyer-based open source](https://about.gitlab.com/company/stewardship/). We believe that this is essential to accomplishing our mission.

Currently we don’t have any specific plans to monetize Chroma, we are working on a hosted service that will be launched as a free technical preview to make it easier for developers to get going. We are 100% focused on building valuable open source software with the community and for the community.


## Our investors

Chroma raised an $18M seed round led by Astasia Myers from Quiet Capital. Joining the round are angels including Naval Ravikant, Max and Jack Altman, Jordan Tigani (Motherduck), Guillermo Rauch (Vercel), Akshay Kothari (Notion), Amjad Masad (Replit), Spencer Kimball (CockroachDB), and other founders and leaders from ScienceIO, Gumroad, MongoDB, Scale, Hugging Face, Jasper and more.

{% CenteredContent %}
![chroma-investors](/investors.png)
{% /CenteredContent %}

Chroma raised a pre-seed in May 2022, led by Anthony Goldbloom (Kaggle) from AIX Ventures, James Cham from Bloomberg Beta, and Nat Friedman and Daniel Gross (AI Grant).

We're excited to work with a deep set of investors and enterpreneurs who have invested in and built some of the most successful open-source projects in the world.


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/contributing.md
================================================
---
{
  "id": "contributing",
  "title": "Contributing",
  "section": "Overview",
  "order": 3
}
---

# Contributing

We welcome all contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas.

## Getting Started
Here are some helpful links to get you started with contributing to Chroma

- The Chroma codebase is hosted on [Github](https://github.com/chroma-core/chroma)
- Issues are tracked on [Github Issues](https://github.com/chroma-core/chroma/issues). Please report any issues you find there making sure to fill out the correct [form for the type of issue you are reporting](https://github.com/chroma-core/chroma/issues/new/choose).
- In order to run Chroma locally you can follow the [Development Instructions](https://github.com/chroma-core/chroma/blob/main/DEVELOP.md).
- If you want to contribute and aren't sure where to get started you can search for issues with the [Good first issue](https://github.com/chroma-core/chroma/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) tag or take a look at our [Roadmap](https://docs.trychroma.com/roadmap).
- The Chroma documentation (including this page!) is hosted on [Github](https://github.com/chroma-core/docs) as well. If you find any issues with the documentation please report them on the Github Issues page for the documentation [here](https://github.com/chroma-core/docs/issues).


## Contributing Code and Ideas

### Pull Requests
In order to submit a change to Chroma please submit a [Pull Request](https://github.com/chroma-core/chroma/compare) against Chroma or the documentation. The pull request will be reviewed by the Chroma team and if approved, will be merged into the repository. We will do our best to review pull requests in a timely manner but please be patient as we are a small team. We will work to integrate your proposed changes as quickly as possible if they align with the goals of the project. We ask that you label your pull request with a title prefix that indicates the type of change you are proposing. The following prefixes are used:

```
ENH: Enhancement, new functionality
BUG: Bug fix
DOC: Additions/updates to documentation
TST: Additions/updates to tests
BLD: Updates to the build process/scripts
PERF: Performance improvement
TYP: Type annotations
CLN: Code cleanup
CHORE: Maintenance and other tasks that do not modify source or test files
```


### CIPs
Chroma Improvement Proposals or CIPs (pronounced "Chips") are the way to propose new features or large changes to Chroma. If you plan to make a large change to Chroma please submit a CIP first so that the core Chroma team as well as the community can discuss the proposed change and provide feedback. A CIP should provide a concise technical specification of the feature and a rationale for why it is needed. The CIP should be submitted as a pull request to the [CIPs folder](https://github.com/chroma-core/chroma/tree/main/docs). The CIP will be reviewed by the Chroma team and if approved will be merged into the repository. To learn more about writing a CIP you can read the [guide](https://github.com/chroma-core/chroma/blob/main/docs/CIP_Chroma_Improvment_Proposals.md). CIPs are not required for small changes such as bug fixes or documentation updates.

A CIP starts in the "Proposed" state, then moves to "Under Review" once the Chroma team has reviewed it and is considering it for implementation. Once the CIP is approved it will move to the "Accepted" state and the implementation can begin. Once the implementation is complete the CIP will move to the "Implemented" state. If the CIP is not approved it will move to the "Rejected" state. If the CIP is withdrawn by the author it will move to the "Withdrawn" state.


### Discord
For less fleshed out ideas you want to discuss with the community, you can join our [Discord](https://discord.gg/Fk2pH7k6) and chat with us in the #feature-ideas channel. We are always happy to discuss new ideas and features with the community.


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/getting-started.md
================================================
---
{
  "id": "getting-started",
  "title": "Getting Started",
  "section": "Overview",
  "order": 1
}
---

# Getting Started

Chroma is an AI-native open-source vector database. It comes with everything you need to get started built in, and runs on your machine. A [hosted version](https://airtable.com/shrOAiDUtS2ILy5vZ) is coming soon!

### 1. Install

{% Tabs %}

{% Tab label="python" %}

```terminal
pip install chromadb
```

{% /Tab %}

{% Tab label="typescript" %}

{% TabbedUseCaseCodeBlock language="Terminal" %}

{% Tab label="yarn" %}
```terminal
yarn install chromadb chromadb-default-embed 
```
{% /Tab %}

{% Tab label="npm" %}
```terminal
npm install --save chromadb chromadb-default-embed
```
{% /Tab %}

{% Tab label="pnpm" %}
```terminal
pnpm install chromadb chromadb-default-embed 
```
{% /Tab %}

{% /TabbedUseCaseCodeBlock %}

Install chroma via `pip` to easily run the backend server. Here are [instructions](https://pip.pypa.io/en/stable/installation/) for installing and running `pip`. Alternatively, you can also run Chroma in a [Docker](../../production/containers/docker) container.

```terminal
pip install chromadb
```

{% /Tab %}

{% /Tabs %}

### 2. Create a Chroma Client

{% Tabs %}

{% Tab label="python" %}
```python
import chromadb
chroma_client = chromadb.Client()
```
{% /Tab %}
{% Tab label="typescript" %}

Run the Chroma backend:

{% TabbedUseCaseCodeBlock language="Terminal" %}

{% Tab label="CLI" %}
```terminal
chroma run --path ./getting-started 
```
{% /Tab %}

{% Tab label="Docker" %}
```terminal
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```
{% /Tab %}

{% /TabbedUseCaseCodeBlock %}

Then create a client which connects to it:

{% TabbedUseCaseCodeBlock language="typescript" %}

{% Tab label="ESM" %}
```typescript
import { ChromaClient } from "chromadb";
const client = new ChromaClient();
```
{% /Tab %}

{% Tab label="CJS" %}
```typescript
const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
```
{% /Tab %}

{% /TabbedUseCaseCodeBlock %}

{% /Tab %}

{% /Tabs %}

### 3. Create a collection

Collections are where you'll store your embeddings, documents, and any additional metadata. Collections index your embeddings and documents, and enable efficient retrieval and filtering. You can create a collection with a name:

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection = chroma_client.create_collection(name="my_collection")
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
const collection = await client.createCollection({
  name: "my_collection",
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

### 4. Add some text documents to the collection

Chroma will store your text and handle embedding and indexing automatically. You can also customize the embedding model. You must provide unique string IDs for your documents.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.add({
    documents: [
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
    ids: ["id1", "id2"],
});
```
{% /Tab %}

{% /TabbedCodeBlock %}

### 5. Query the collection

You can query the collection with a list of query texts, and Chroma will return the `n` most similar results. It's that easy!

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
```

{% /Tab %}

{% Tab label="typescript" %}
```typescript
const results = await collection.query({
    queryTexts: "This is a query document about hawaii", // Chroma will embed this for you
    nResults: 2, // how many results to return
});

console.log(results);
```
{% /Tab %}

{% /TabbedCodeBlock %}

If `n_results` is not provided, Chroma will return 10 results by default. Here we only added 2 documents, so we set `n_results=2`.

### 6. Inspect Results

From the above query - you can see that our query about `hawaii` is the semantically most similar to the document about `pineapple`.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
{
  'documents': [[
      'This is a document about pineapple',
      'This is a document about oranges'
  ]],
  'ids': [['id1', 'id2']],
  'distances': [[1.0404009819030762, 1.243080496788025]],
  'uris': None,
  'data': None,
  'metadatas': [[None, None]],
  'embeddings': None,
}
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
{
    documents: [
        [
            'This is a document about pineapple', 
            'This is a document about oranges'
        ]
    ], 
    ids: [
        ['id1', 'id2']
    ], 
    distances: [[1.0404009819030762, 1.243080496788025]],
    uris: null,
    data: null,
    metadatas: [[null, null]],
    embeddings: null
}
```
{% /Tab %}

{% /TabbedCodeBlock %}

### 7. Try it out yourself

For example - what if we tried querying with `"This is a document about florida"`?

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
import chromadb
chroma_client = chromadb.Client()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
import { ChromaClient } from "chromadb";
const client = new ChromaClient();

// switch `createCollection` to `getOrCreateCollection` to avoid creating a new collection every time
const collection = await client.getOrCreateCollection({
    name: "my_collection",
});

// switch `addRecords` to `upsertRecords` to avoid adding the same documents every time
await collection.upsert({
    documents: [
        "This is a document about pineapple",
        "This is a document about oranges",
    ],
    ids: ["id1", "id2"],
});

const results = await collection.query({
    queryTexts: "This is a query document about florida", // Chroma will embed this for you
    nResults: 2, // how many results to return
});

console.log(results);
```
{% /Tab %}

{% /TabbedCodeBlock %}

## Next steps

In this guide we used Chroma's [ephemeral client](../run-chroma/ephemeral-client) for simplicity. It starts a Chroma server in-memory, so any data you ingest will be lost when your program terminates. You can use the [persistent client](../run-chroma/persistent-client) or run Chroma in [client-server mode](../run-chroma/client-server) if you need data persistence.

- Learn how to [Deploy Chroma](../../production/deployment) to a server
- Join Chroma's [Discord Community](https://discord.com/invite/MMeYNTmh3x) to ask questions and get help
- Follow Chroma on [Twitter (@trychroma)](https://twitter.com/trychroma) for updates


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/introduction.md
================================================
---
{
  "id": "introduction",
  "title": "Introduction",
  "section": "Overview",
  "order": 0
}
---

# Chroma

**Chroma is the open-source AI application database**. Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.

{% Banner type="tip" %}
New to Chroma? Check out the [getting started guide](./getting-started)
{% /Banner %}

![Chroma Computer](/computer.svg)

Chroma gives you everything you need for retrieval:

- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

Chroma runs as a server and provides `Python` and `JavaScript/TypeScript` client SDKs. Check out the [Colab demo](https://colab.research.google.com/drive/1QEzFyqnoFxq7LUGyP1vzR4iLt9PpCDXv?usp=sharing) (yes, it can run in a Jupyter notebook).

Chroma is licensed under [Apache 2.0](https://github.com/chroma-core/chroma/blob/main/LICENSE)

### Python
In Python, Chroma can run in a python script or as a server. Install Chroma with

```shell
pip install chromadb
```

### JavaScript
In JavaScript, use the Chroma JS/TS Client to connect to a Chroma server. Install Chroma with your favorite package manager:

{% TabbedUseCaseCodeBlock language="Terminal" %}

{% Tab label="yarn" %}
```terminal
yarn install chromadb chromadb-default-embed 
```
{% /Tab %}

{% Tab label="npm" %}
```terminal
npm install --save chromadb chromadb-default-embed
```
{% /Tab %}

{% Tab label="pnpm" %}
```terminal
pnpm install chromadb chromadb-default-embed 
```
{% /Tab %}

{% /TabbedUseCaseCodeBlock %}


Continue with the full [getting started guide](./getting-started).


***

## Language Clients

| Language      | Client                                                                                                                   |
|---------------|--------------------------------------------------------------------------------------------------------------------------|
| Python        | [`chromadb`](https://pypistats.org/packages/chromadb) (by Chroma)                                                        |
| Javascript    | [`chromadb`](https://www.npmjs.com/package/chromadb) (by Chroma)                                                         |
| Ruby          | [from @mariochavez](https://github.com/mariochavez/chroma)                                                               |
| Java          | [from @t_azarov](https://github.com/amikos-tech/chromadb-java-client)                                                    |
| Go            | [from @t_azarov](https://github.com/amikos-tech/chroma-go)                                                               |
| C#            | [from @microsoft](https://github.com/microsoft/semantic-kernel/tree/main/dotnet/src/Connectors/Connectors.Memory.Chroma) |
| Rust          | [from @Anush008](https://crates.io/crates/chromadb)                                                                      |
| Elixir        | [from @3zcurdia](https://hex.pm/packages/chroma/)                                                                        |
| Dart          | [from @davidmigloz](https://pub.dev/packages/chromadb)                                                                   |
| PHP           | [from @CodeWithKyrian](https://github.com/CodeWithKyrian/chromadb-php)                                                   |
| PHP (Laravel) | [from @HelgeSverre](https://github.com/helgeSverre/chromadb)                                                             |
| Clojure       | [from @levand](https://github.com/levand/clojure-chroma-client)                                                          |


{% br %}{% /br %}

We welcome [contributions](/markdoc/content/docs/overview/contributing.md) for other languages!



================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/roadmap.md
================================================
---
{
  "id": "roadmap",
  "title": "Roadmap",
  "section": "Overview",
  "order": 2
}
---


# Roadmap

The goal of this doc is to align *core* and *community* efforts for the project and to share what's in store for this year!

**Sections**
- What is the core Chroma team working on right now?
- What will Chroma prioritize over the next 6mo?
- What areas are great for community contributions?

## What is the core Chroma team working on right now?

- Standing up that distributed system as a managed service (aka "Hosted Chroma" - [sign up for waitlist](https://airtable.com/shrOAiDUtS2ILy5vZ)!)

## What did the Chroma team just complete?

Features like:
- *New* - [Chroma 0.4](https://www.trychroma.com/blog/chroma_0.4.0) - our first production-oriented release
- A more minimal python-client only build target
- Google PaLM embedding support
- OpenAI ChatGPT Retrieval Plugin

## What will Chroma prioritize over the next 6mo?

**Next Milestone: ☁️ Launch Hosted Chroma**

**Areas we will invest in**

Not an exhaustive list, but these are some of the core team’s biggest priorities over the coming few months. Use caution when contributing in these areas and please check-in with the core team first.

- **Workflow**: Building tools for answer questions like: what embedding model should I use? And how should I chunk up my documents?
- **Visualization**: Building visualization tool to give developers greater intuition embedding spaces
- **Query Planner**: Building tools to enable per-query and post-query transforms
- **Developer experience**: Extending Chroma into a CLI
- **Easier Data Sharing**: Working on formats for serialization and easier data sharing of embedding Collections
- **Improving recall**: Fine-tuning embedding transforms through human feedback
- **Analytical horsepower**: Clustering, deduplication, classification and more

## What areas are great for community contributions?

This is where you have a lot more free reign to contribute (without having to sync with us first)!

If you're unsure about your contribution idea, feel free to chat with us (@chroma) in the `#general` channel in [our Discord](https://discord.gg/rahcMUU5XV)! We'd love to support you however we can.

### Example Templates

We can always use [more integrations](../../integrations/chroma-integrations) with the rest of the AI ecosystem. Please let us know if you're working on one and need help!

Other great starting points for Chroma (please send PRs for more [here](https://github.com/chroma-core/docs/tree/swyx/addRoadmap/docs)):
- [Google Colab](https://colab.research.google.com/drive/1QEzFyqnoFxq7LUGyP1vzR4iLt9PpCDXv?usp=sharing)
- [Replit Template](https://replit.com/@swyx/BasicChromaStarter?v=1)

For those integrations we do have, like LangChain and LlamaIndex, we do always want more tutorials, demos, workshops, videos, and podcasts (we've done some pods [on our blog](https://trychroma.com/interviews)).

### Example Datasets

It doesn’t make sense for developers to embed the same information over and over again with the same embedding model.

We'd like suggestions for:

- "small" (<100 rows)
- "medium" (<5MB)
- "large" (>1GB)

datasets for people to stress test Chroma in a variety of scenarios.

### Embeddings Comparison

Chroma does ship with Sentence Transformers by default for embeddings, but we are otherwise unopinionated about what embeddings you use. Having a library of information that has been embedded with many models, alongside example query sets would make it much easier for empirical work to be done on the effectiveness of various models across different domains.

- [Preliminary reading on Embeddings](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526?gi=ee46baab0d8f)
- [Huggingface Benchmark of a bunch of Embeddings](https://huggingface.co/blog/mteb)
- [notable issues with GPT3 Embeddings](https://twitter.com/Nils_Reimers/status/1487014195568775173) and alternatives to consider

### Experimental Algorithms

If you have a research background, please consider adding to our `ExperimentalAPI`s. For example:

- Projections (t-sne, UMAP, the new hotness, the one you just wrote) and Lightweight visualization
- Clustering (HDBSCAN, PCA)
- Deduplication
- Multimodal (CLIP)
- Fine-tuning manifold with human feedback [eg](https://github.com/openai/openai-cookbook/blob/main/examples/Customizing_embeddings.ipynb)
- Expanded vector search (MMR, Polytope)
- Your research

You can find the REST OpenAPI spec at `localhost:8000/openapi.json` when the backend is running.

Please [reach out](https://discord.gg/MMeYNTmh3x) and talk to us before you get too far in your projects so that we can offer technical guidance/align on roadmap.


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/overview/telemetry.md
================================================
# Telemetry

Chroma contains a telemetry feature that collects **anonymous** usage information.

### Why?

We use this information to help us understand how Chroma is used, to help us prioritize work on new features and bug fixes, and to help us improve Chroma’s performance and stability.

### Opting out

If you prefer to opt out of telemetry, you can do this in two ways.

#### In Client Code

{% Tabs %}

{% Tab label="python" %}

Set `anonymized_telemetry` to `False` in your client's settings:

```python
from chromadb.config import Settings
client = chromadb.Client(Settings(anonymized_telemetry=False))
# or if using PersistentClient
client = chromadb.PersistentClient(path="/path/to/save/to", settings=Settings(anonymized_telemetry=False))
```

{% /Tab %}

{% Tab label="typescript" %}

Disable telemetry on you Chroma server (see next section).

{% /Tab %}

{% /Tabs %}

#### In Chroma's Backend Using Environment Variables

Set `ANONYMIZED_TELEMETRY` to `False` in your shell or server environment.

If you are running Chroma on your local computer with `docker-compose` you can set this value in an `.env` file placed in the same directory as the `docker-compose.yml` file:

```
ANONYMIZED_TELEMETRY=False
```

### What do you track?

We will only track usage details that help us make product decisions, specifically:

- Chroma version and environment details (e.g. OS, Python version, is it running in a container, or in a jupyter notebook)
- Usage of embedding functions that ship with Chroma and aggregated usage of custom embeddings (we collect no information about the custom embeddings themselves)
- Client interactions with our hosted Chroma Cloud service.
- Collection commands. We track the anonymized uuid of a collection as well as the number of items
    - `add`
    - `update`
    - `query`
    - `get`
    - `delete`

We **do not** collect personally-identifiable or sensitive information, such as: usernames, hostnames, file names, environment variables, or hostnames of systems being tested.

To view the list of events we track, you may reference the **[code](https://github.com/chroma-core/chroma/blob/main/chromadb/telemetry/product/events.py)**

### Where is telemetry information stored?

We use **[Posthog](https://posthog.com/)** to store and visualize telemetry data.

{% Banner type="tip" %}

Posthog is an open source platform for product analytics. Learn more about Posthog on **[posthog.com](https://posthog.com/)** or **[github.com/posthog](https://github.com/posthog/posthog)**

{% /Banner %}

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/querying-collections/full-text-search.md
================================================
# Full Text Search

In order to filter on document contents, you must supply a `where_document` filter dictionary to the query. We support two filtering keys: `$contains` and `$not_contains`. The dictionary must have the following structure:

```python
# Filtering for a search_string
{
    "$contains": "search_string"
}

# Filtering for not contains
{
    "$not_contains": "search_string"
}
```

You can combine full-text search with Chroma's metadata filtering.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.query(
    query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.query({
    queryTexts: ["doc10", "thus spake zarathustra", ...],
    nResults: 10,
    where: {"metadata_field": "is_equal_to_this"},
    whereDocument: {"$contains": "search_string"}
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/querying-collections/metadata-filtering.md
================================================
# Metadata Filtering

Chroma supports filtering queries by `metadata` and `document` contents. The `where` filter is used to filter by `metadata`.

In order to filter on metadata, you must supply a `where` filter dictionary to the query. The dictionary must have the following structure:

```python
{
    "metadata_field": {
        <Operator>: <Value>
    }
}
```

Filtering metadata supports the following operators:

- `$eq` - equal to (string, int, float)
- `$ne` - not equal to (string, int, float)
- `$gt` - greater than (int, float)
- `$gte` - greater than or equal to (int, float)
- `$lt` - less than (int, float)
- `$lte` - less than or equal to (int, float)

Using the `$eq` operator is equivalent to using the `where` filter.

```python
{
    "metadata_field": "search_string"
}

# is equivalent to

{
    "metadata_field": {
        "$eq": "search_string"
    }
}
```

{% Banner type="note" %}
Where filters only search embeddings where the key exists. If you search `collection.get(where={"version": {"$ne": 1}})`. Metadata that does not have the key `version` will not be returned.
{% /Banner %}

#### Using logical operators

You can also use the logical operators `$and` and `$or` to combine multiple filters.

An `$and` operator will return results that match all of the filters in the list.

```python
{
    "$and": [
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        },
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        }
    ]
}
```

An `$or` operator will return results that match any of the filters in the list.

```python
{
    "$or": [
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        },
        {
            "metadata_field": {
                <Operator>: <Value>
            }
        }
    ]
}
```

#### Using inclusion operators (`$in` and `$nin`)

The following inclusion operators are supported:

- `$in` - a value is in predefined list (string, int, float, bool)
- `$nin` - a value is not in predefined list (string, int, float, bool)

An `$in` operator will return results where the metadata attribute is part of a provided list:

```json
{
  "metadata_field": {
    "$in": ["value1", "value2", "value3"]
  }
}
```

An `$nin` operator will return results where the metadata attribute is not part of a provided list:

```json
{
  "metadata_field": {
    "$nin": ["value1", "value2", "value3"]
  }
}
```

{% Banner type="tip" %}

For additional examples and a demo how to use the inclusion operators, please see provided notebook [here](https://github.com/chroma-core/chroma/blob/main/examples/basic_functionality/in_not_in_filtering.ipynb)

{% /Banner %}

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/querying-collections/query-and-get.md
================================================
# Query and Get Data from Chroma Collections

Chroma collections can be queried in a variety of ways, using the `.query` method.

You can query by a set of `query embeddings`.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
const result = await collection.query({
    queryEmbeddings: [[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    nResults: 10,
    where: {"metadata_field": "is_equal_to_this"},
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

The query will return the `n results` closest matches to each `query embedding`, in order.
An optional `where` filter dictionary can be supplied to filter by the `metadata` associated with each document.
Additionally, an optional `where document` filter dictionary can be supplied to filter by contents of the document.

If the supplied `query embeddings` are not the same dimension as the collection, an exception will be raised.

You can also query by a set of `query texts`. Chroma will first embed each `query text` with the collection's embedding function, and then perform the query with the generated embedding.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.query(
    query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.query({
    queryTexts: ["doc10", "thus spake zarathustra", ...],
    nResults: 10,
    where: {"metadata_field": "is_equal_to_this"},
    whereDocument: {"$contains": "search_string"}
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

You can also retrieve items from a collection by `id` using `.get`.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
collection.get(
	ids=["id1", "id2", "id3", ...],
	where={"style": "style1"}
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await collection.get( {
    ids: ["id1", "id2", "id3", ...],
    where: {"style": "style1"}
})
```
{% /Tab %}

{% /TabbedCodeBlock %}

`.get` also supports the `where` and `where document` filters. If no `ids` are supplied, it will return all items in the collection that match the `where` and `where document` filters.

### Choosing Which Data is Returned

When using get or query you can use the `include` parameter to specify which data you want returned - any of `embeddings`, `documents`, `metadatas`, and for query, `distances`. By default, Chroma will return the `documents`, `metadatas` and in the case of query, the `distances` of the results. `embeddings` are excluded by default for performance and the `ids` are always returned. You can specify which of these you want returned by passing an array of included field names to the includes parameter of the query or get method. Note that embeddings will be returned as a 2-d numpy array in `.get` and a python list of 2-d numpy arrays in `.query`.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
# Only get documents and ids
collection.get(
    include=["documents"]
)

collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    include=["documents"]
)
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
// Only get documents and ids
await collection.get({
    include: ["documents"]
})

await collection.query({
    query_embeddings: [[11.1, 12.1, 13.1], [1.1, 2.3, 3.2], ...],
    include: ["documents"]
})
```
{% /Tab %}

{% /TabbedCodeBlock %}


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/run-chroma/client-server.md
================================================
# Running Chroma in Client-Server Mode

Chroma can also be configured to run in client/server mode. In this mode, the Chroma client connects to a Chroma server running in a separate process.

To start the Chroma server, run the following command:

```terminal
chroma run --path /db_path
```

{% Tabs %}

{% Tab label="python" %}

Then use the Chroma HTTP client to connect to the server:

```python
import chromadb

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
```

That's it! Chroma's API will run in `client-server` mode with just this change.

Chroma also provides an async HTTP client. The behaviors and method signatures are identical to the synchronous client, but all methods that would block are now async. To use it, call `AsyncHttpClient` instead:

```python
import asyncio
import chromadb

async def main():
    client = await chromadb.AsyncHttpClient()

    collection = await client.create_collection(name="my_collection")
    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )

asyncio.run(main())
```

If you [deploy](../../production/deployment) your Chroma server, you can also use our [http-only](./python-http-client) package.

{% /Tab %}

{% Tab label="typescript" %}

Then you can connect to it by instantiating a new `ChromaClient`:

```typescript
import { ChromaClient } from "chromadb";

const client = new ChromaClient();
```

{% /Tab %}

{% /Tabs %}


================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/run-chroma/ephemeral-client.md
================================================
# Ephemeral Client

In Python, you can run a Chroma server in-memory and connect to it with the ephemeral client:

```python
import chromadb

client = chromadb.Client()
```

The `Client()` method starts a Chroma server in-memory and also returns a client with which you can connect to it.

This is a great tool for experimenting with different embedding functions and retrieval techniques in a Python notebook, for example. If you don't need data persistence, the ephemeral client is a good choice for getting up and running with Chroma.

================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/run-chroma/persistent-client.md
================================================
# Persistent Client

{% Tabs %}

{% Tab label="python" %}

You can configure Chroma to save and load the database from your local machine, using the `PersistentClient`. 

Data will be persisted automatically and loaded on start (if it exists).

```python
import chromadb

client = chromadb.PersistentClient(path="/path/to/save/to")
```

The `path` is where Chroma will store its database files on disk, and load them on start. If you don't provide a path, the default is `.chroma`

{% /Tab %}

{% Tab label="typescript" %}

To connect with the JS/TS client, you must connect to a Chroma server. 

To run a Chroma server locally that will persist your data, install Chroma via `pip`:

```terminal
pip install chromadb
```

And run the server using our CLI:

```terminal
chroma run --path ./getting-started 
```

The `path` is where Chroma will store its database files on disk, and load them on start. The default is `.chroma`.

Alternatively, you can also use our official Docker image:

```terminal
docker pull chromadb/chroma
docker run -p 8000:8000 chromadb/chroma
```

With a Chroma server running locally, you can connect to it by instantiating a new `ChromaClient`:

```typescript
import { ChromaClient } from "chromadb";

const client = new ChromaClient();
```

See [Running Chroma in client-server mode](../client-server-mode) for more.

{% /Tab %}

{% /Tabs %}

The client object has a few useful convenience methods.

* `heartbeat()` - returns a nanosecond heartbeat. Useful for making sure the client remains connected.
* `reset()` - empties and completely resets the database. ⚠️ This is destructive and not reversible.

{% TabbedCodeBlock %}

{% Tab label="python" %}
```python
client.heartbeat()
client.reset()
```
{% /Tab %}

{% Tab label="typescript" %}
```typescript
await client.heartbeat();
await client.reset();
```
{% /Tab %}

{% /TabbedCodeBlock %}



================================================
File: /docs/docs.trychroma.com/markdoc/content/docs/run-chroma/python-http-client.md
================================================
# The Python HTTP-Only Client

If you are running Chroma in client-server mode, where you run a Chroma server and client on separate machines, you may not need the full Chroma package where you run your client. Instead, you can use the lightweight client-only library.
In this case, you can install the `chromadb-client` package. This package is a lightweight HTTP client for the server with a minimal dependency footprint.

On your server, install chroma with

```terminal
pip install chromadb
```

And run a Chroma server:

```terminal
chroma run --path [path/to/persist/data]
```

Then, on your client side, install the HTTP-only client: 

```terminal
pip install chromadb-client
```

```python
import chromadb
# Example setup of the client to connect to your chroma server
client = chromadb.HttpClient(host='localhost', port=8000)

# Or for async usage:
async def main():
    client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
```

Note that the `chromadb-client` package is a subset of the full Chroma library and does not include all the dependencies. If you want to use the full Chroma library, you can install the `chromadb` package instead.
Most importantly, there is no default embedding function. If you add() documents without embeddings, you must have manually specified an embedding function and installed the dependencies for it.


