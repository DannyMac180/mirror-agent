"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import Literal, cast
from enum import Enum
import os

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

    Raises:
        ValueError: If the retriever is not properly configured or initialized.
    """
    configuration = Configuration.from_runnable_config(config)
    with retrieval.make_retriever(config) as retriever:
        if retriever is None:
            if configuration.retriever_provider != "chroma":
                missing_env = [var for var in ["RETRIEVER_URL", "RETRIEVER_API_KEY"] if not os.getenv(var)]
                if missing_env:
                    raise ValueError("Missing environment variables: " + ", ".join(missing_env) + ". See [docs/langgraph.md] for configuration details.")
            raise ValueError("Retriever is not configured or failed to initialize. Please check your configuration.")
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
{{
    "next": <either "reason" or "respond">,
    "explanation": <brief explanation>
}}

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
) -> dict[str, str]:
    """Apply reasoning to complex queries using Gemini 2.0 Flash Thinking."""
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a wise assistant who applies reasoning to complex queries.

Analyze the query and provide step-by-step reasoning that:
1. Breaks down the problem into components
2. Applies logical deduction
3. Considers multiple perspectives
4. Reaches clear conclusions

Current query: {query}

Available context: {context}"""),
    ])
    
    if "gemini" in configuration.reasoning_model.lower():
        # Use the native Google Gen AI Python SDK for the Gemini Thinking model.
        import os
        import asyncio
        import google.generativeai as genai

        # Configure the SDK using the GOOGLE_API_KEY environment variable.
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)

        # Prepare the prompt text using our template.
        context = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
        prompt_result = await prompt.ainvoke(
            {
                "query": get_message_text(state.messages[-1]),
                "context": context
            },
            config,
        )
        # Extract the text content from the prompt result
        prompt_text = str(prompt_result) if prompt_result else ""

        # Call the synchronous SDK function in a non-blocking way.
        model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
        response = await asyncio.to_thread(
            model.generate_content,
            prompt_text
        )

        # Extract the generated result
        reasoning_result = response.text
    else:
        # Fallback to the legacy LangChain-based behavior.
        model = load_chat_model(configuration.reasoning_model)
        if "gemini" in configuration.reasoning_model.lower():
            from types import MethodType
            original_convert_input = model._convert_input
            def gemini_convert_input(self, prompt_input):
                if isinstance(prompt_input, dict) and "contents" in prompt_input:
                    return prompt_input
                return original_convert_input(prompt_input)
            model._convert_input = MethodType(gemini_convert_input, model)

        context = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
        message_value = await prompt.ainvoke(
            {
                "query": get_message_text(state.messages[-1]),
                "context": context
            },
            config,
        )
        payload = {"contents": [message_value]}
        response = await model.ainvoke(payload, config)
        reasoning_result = str(response)

    return {"reasoning_trace": reasoning_result}


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
