Mem0.ai Python Library Documentation
====================================

Overview
--------
The mem0.ai Python library provides integration with the mem0.ai API for managing an agent's memory state. It allows you to add, search, and retrieve different types of memories—including plain text memories, conversation messages, and documents. This documentation explains how to use the library via the provided Mem0Memory class.

Environment Setup
-----------------
Before using the library, ensure you have the following environment variables set:
  - MEM0_API_KEY: Your mem0.ai API key
  - MEM0_ORG_ID: Your organization ID
  - MEM0_PROJECT_ID: Your project ID
  - MEM0_USER_ID: A unique identifier for the user (e.g., 'test-user')
  - MEM0_APP_ID: The application ID (default is "mirror-agent")
  - MEM0_RUN_ID: The run ID (default is "test-run")

Example (in Python):

    import os
    os.environ['MEM0_API_KEY'] = 'your_api_key'
    os.environ['MEM0_ORG_ID'] = 'your_org_id'
    os.environ['MEM0_PROJECT_ID'] = 'your_project_id'
    os.environ['MEM0_USER_ID'] = 'test-user'
    os.environ['MEM0_APP_ID'] = 'your_app_id'
    os.environ['MEM0_RUN_ID'] = 'your_run_id'

    from src.retrieval_graph.mem0_client import Mem0Memory
    mem0 = Mem0Memory()

Basic Methods
-------------

1. Adding a Memory
------------------
Use `add_memory` to store a simple text memory.

Example:

    mem0.add_memory("This is a sample memory text.")

Internally, this method constructs a message dictionary that includes:
  - text: The memory content
  - user_id: The user identifier from the environment
  - agent_id: Set to "mirror-agent"
  - app_id and run_id: Sourced from environment variables (with defaults if not set)

2. Adding a Message
--------------------
Use `add_message` to store conversation messages. This method accepts a message object (e.g., an instance of HumanMessage or AIMessage from langchain_core.messages).

Example:

    from langchain_core.messages import HumanMessage
    human_msg = HumanMessage(content="Hello, AI!")
    mem0.add_message(human_msg)

The method creates a message dictionary with additional metadata such as message type and any extra kwargs provided in the message object.

3. Adding a Document
---------------------
Use `add_document` to store a document. It accepts a Document object from langchain.schema.

Example:

    from langchain.schema import Document
    doc = Document(
        page_content="This is the document content.",
        metadata={"source": "example", "timestamp": "2023-01-01T12:34:56"}
    )
    mem0.add_document(doc)

This method stores the document content along with metadata indicating its type.

4. Retrieving Memories
----------------------
Use `get_memories` to search for and retrieve memories that match a query. This method applies filters based on user_id, agent_id, app_id, and run_id.

Example:

    results = mem0.get_memories("sample")
    for mem in results:
        print(mem)

5. Retrieving Stored Messages and Documents
---------------------------------------------
The library provides helper methods to fetch only messages or only documents:

- `get_messages`: Retrieves messages, converts them to message objects (AnyMessage), and returns them.
- `get_documents`: Retrieves documents and converts them to Document objects.

Example:

    stored_messages = mem0.get_messages()
    for msg in stored_messages:
        print(msg.content)

    stored_docs = mem0.get_documents()
    for doc in stored_docs:
        print(doc.page_content)

Conclusion
----------
The mem0.ai Python library streamlines memory management by allowing you to add and retrieve various types of memories with simple method calls. Just ensure that your environment variables are set correctly, and use the Mem0Memory class to interact with mem0.ai's API.

For further details, check the source code in src/retrieval_graph/mem0_client.py which implements these methods in detail.

## Important Notes

### Required Filters
The mem0.ai API requires at least one of the following filters for all requests:
- `agent_id`
- `user_id` 
- `app_id`
- `run_id`

These filters must be included in the root level of the request payload, not nested under a "filters" key. For example:

```python
# Correct way to include filters
memory_message = {
    "text": "Memory content",
    "user_id": "test-user",  # Required filter
    "agent_id": "my-agent",  # Required filter
    "app_id": "my-app",      # Required filter
    "run_id": "test-run"     # Required filter
}

# Incorrect way (will cause 400 error)
memory_message = {
    "text": "Memory content",
    "filters": {  # Don't nest under "filters"
        "user_id": "test-user",
        "agent_id": "my-agent",
        "app_id": "my-app",
        "run_id": "test-run"
    }
}
```

You must set these values either through environment variables or directly in your requests. The library will automatically include them if you've set the corresponding environment variables:

```bash
export MEM0_USER_ID="test-user"
export MEM0_APP_ID="my-app"
export MEM0_RUN_ID="test-run"
``` 