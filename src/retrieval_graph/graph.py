"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import Literal, cast
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model


class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str


class RetrievalDecision(BaseModel):
    """Decision on whether to retrieve documents."""
    next: Literal["retrieve", "respond"]


async def should_retrieve(
    state: State, *, config: RunnableConfig
) -> State:
    """Decide whether to retrieve documents or respond directly.
    
    This function analyzes the user's query to determine if document retrieval
    would be beneficial for generating a response.
    """
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a wise assistant who decides whether a query requires retrieving personal documents and context to provide a good response.

Respond with 'retrieve' if:
- The query asks about personal information, history, or patterns
- The query references past interactions or documents
- The query requires specific context to answer effectively

Respond with 'respond' if:
- The query is general or philosophical in nature
- The query is about the current interaction only
- The query can be answered with general knowledge or wisdom

Current query: {query}"""),
    ])
    
    model = load_chat_model(configuration.query_model).with_structured_output(RetrievalDecision)
    
    message_value = await prompt.ainvoke(
        {"query": get_message_text(state.messages[-1])},
        config,
    )
    decision = await model.ainvoke(message_value, config)
    
    state.next = decision.next
    return state


async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    messages = state.messages
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }


async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    with retrieval.make_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the graph with conditional routing
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add all nodes
builder.add_node("should_retrieve", should_retrieve)
builder.add_node("generate_query", generate_query)
builder.add_node("retrieve", retrieve)
builder.add_node("respond", respond)

# Add conditional edges
builder.add_edge("__start__", "should_retrieve")
builder.add_conditional_edges(
    "should_retrieve",
    lambda state: state.next if hasattr(state, "next") else "retrieve",
    {
        "retrieve": "generate_query",
        "respond": "respond",
    },
)
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
