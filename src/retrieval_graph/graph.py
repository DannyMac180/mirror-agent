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


class ReasoningDecision(BaseModel):
    """Decision on whether to apply reasoning."""
    next: Literal["reason", "respond"]
    explanation: str


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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
    reasoning = f"\nReasoning Trace:\n{state.reasoning_trace}" if state.reasoning_trace else ""
    
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs + reasoning,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    return {"messages": [response]}


async def should_reason(
    state: State, *, config: RunnableConfig
) -> State:
    """Decide whether to apply reasoning to the query.
    
    This function analyzes the query to determine if it requires complex reasoning
    like logic, scientific research, math, coding or data analysis.
    """
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You decide if a query needs complex reasoning. 

Respond with 'reason' if the query requires:
- Logic or analytical thinking
- Scientific research or analysis
- Mathematical calculations
- Coding or technical solutions
- Data analysis or interpretation
- Multi-step problem solving
- Complex decision making

Respond with 'respond' if:
- The query is simple and straightforward
- Only requires retrieved context
- Doesn't need complex analysis

Return your response in this exact format:
{
    "next": <either "reason" or "respond">,
    "explanation": <brief explanation>
}

Query: {query}"""),
    ])
    
    model = load_chat_model(configuration.query_model).with_structured_output(ReasoningDecision)
    
    message_value = await prompt.ainvoke(
        {"query": get_message_text(state.messages[-1])},
        config,
    )
    decision = await model.ainvoke(message_value, config)
    
    state.next = decision.next
    return state


async def apply_reasoning(
    state: State, *, config: RunnableConfig
) -> State:
    """Apply reasoning to complex queries using Gemini 2.0 Flash Thinking."""
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at breaking down and solving complex problems.
Use Gemini 2.0 Flash Thinking to analyze queries requiring:
- Logic/analytical thinking
- Scientific research
- Math
- Coding
- Data analysis

Format your response as:
1. Break down the problem
2. Show your step-by-step reasoning
3. Provide a clear conclusion

Context from retrieval:
{context}

Query: {query}"""),
    ])
    
    model = load_chat_model("gemini-2.0-flash-exp")
    
    context = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
    message_value = await prompt.ainvoke(
        {
            "query": get_message_text(state.messages[-1]),
            "context": context
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    state.reasoning_trace = response.content
    return state


# Define the graph with conditional routing
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add all nodes
builder.add_node("should_retrieve", should_retrieve)
builder.add_node("generate_query", generate_query)
builder.add_node("retrieve", retrieve)
builder.add_node("should_reason", should_reason)
builder.add_node("apply_reasoning", apply_reasoning)
builder.add_node("respond", respond)

# Add conditional edges
builder.add_edge("__start__", "should_retrieve")
builder.add_conditional_edges(
    "should_retrieve",
    lambda state: state.next if hasattr(state, "next") else "retrieve",
    {
        "retrieve": "generate_query",
        "respond": "should_reason",
    },
)
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "should_reason")
builder.add_conditional_edges(
    "should_reason",
    lambda state: state.next if hasattr(state, "next") else "respond",
    {
        "reason": "apply_reasoning",
        "respond": "respond",
    },
)
builder.add_edge("apply_reasoning", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile()
graph.name = "RetrievalGraph"
