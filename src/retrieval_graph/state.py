"""State management for the retrieval graph.

This module defines the state structures and reduction functions used in the
retrieval graph. It includes definitions for document indexing, retrieval,
and conversation management using mem0.ai as the memory store.

Classes:
    IndexState: Represents the state for document indexing operations.
    RetrievalState: Represents the state for document retrieval operations.
    ConversationState: Represents the state of the ongoing conversation.

Functions:
    reduce_docs: Processes and reduces document inputs into a sequence of Documents.
    reduce_retriever: Updates the retriever in the state.
    reduce_messages: Manages the addition of new messages to the conversation state.
    reduce_retrieved_docs: Handles the updating of retrieved documents in the state.

The module also includes type definitions and utility functions to support
these state management operations.
"""

import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from .mem0_client import Mem0Memory

# Initialize mem0 client
mem0 = Mem0Memory()

############################  Doc Indexing State  #############################

def reduce_docs(
    existing: Optional[Sequence[Document]],
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    All documents are stored in mem0.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete" to clear existing documents.

    Returns:
        Sequence[Document]: The processed sequence of Document objects.
    """
    if new == "delete":
        return []

    # Convert input to Document objects
    if isinstance(new, str):
        docs = [Document(page_content=new)]
    elif isinstance(new, Sequence):
        if not new:
            return existing or []
        if isinstance(new[0], Document):
            docs = list(new)
        elif isinstance(new[0], dict):
            docs = [Document(**d) for d in new]
        elif isinstance(new[0], str):
            docs = [Document(page_content=s) for s in new]
        else:
            raise ValueError(f"Unsupported document type: {type(new[0])}")
    else:
        raise ValueError(f"Unsupported input type: {type(new)}")

    # Store new documents in mem0
    for doc in docs:
        mem0.add_document(doc)

    # Return combined documents
    return docs


@dataclass
class IndexState:
    """Represents the state for document indexing and retrieval.

    This class defines the structure of the index state, which includes
    the documents to be indexed and the retriever used for searching
    these documents. Documents are stored in mem0.
    """

    docs: Annotated[Sequence[Document], reduce_docs] = field(default_factory=list)


#############################  Agent State  ###################################


def reduce_messages(existing: Sequence[AnyMessage], new: AnyMessage) -> Sequence[AnyMessage]:
    """Add a new message to the conversation state.

    Args:
        existing (Sequence[AnyMessage]): The current messages in the state.
        new (AnyMessage): The new message to add.

    Returns:
        Sequence[AnyMessage]: A new sequence containing all messages.
    """
    # Store new message in mem0
    mem0.add_message(new)
    
    # Return updated messages from mem0
    return mem0.get_messages()


@dataclass
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. Messages are stored in mem0.
    """

    messages: Annotated[Sequence[AnyMessage], reduce_messages] = field(default_factory=list)


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


@dataclass
class State:
    """The state of your graph / agent.

    All state is persisted in mem0, providing durable storage and retrieval
    of conversation history, documents, and other data.
    """

    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    retrieved_docs: list[Document] = field(default_factory=list)
    next: str = field(default="retrieve")
    reasoning_trace: Optional[str] = field(default=None)

    def dict(self) -> dict:
        """Convert the state to a dictionary representation."""
        return {
            "queries": self.queries,
            "retrieved_docs": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.retrieved_docs
            ],
            "next": self.next,
            "reasoning_trace": self.reasoning_trace,
        }
