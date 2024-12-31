# Mirror Agent: Project Plan

## Overview

The **Mirror Agent** is an AI assistant concept that aims to help users understand themselves better by providing Socratic guidance, personalized reflection, and memory-based interactions. The agent is:

- **Model-agnostic**: The user chooses which LLM (OpenAI, Anthropic, or other) is used under the hood.
- **Socratic and reflective**: Encourages users to explore their own thoughts and feelings by asking open-ended questions and delivering feedback to foster self-discovery.
- **Memory-enhanced**: Maintains user-specific context, conversation history, and documents for more personalized and coherent interactions.
- **Extensible**: Utilizes a retriever to access additional documents/files and a reflection module that adds “self-awareness” to the agent, making interactions more meaningful over time.

Based on the attached design sketch and the [LangGraph documentation](https://langchain-ai.github.io/langgraph/), this plan outlines the goals and phases for building the Mirror Agent.

---

## High-Level Architecture

1. **User Interface**  
   - **Chat, Audio, and Video Interfaces** to handle multi-modal inputs/outputs.  
   - This layer mediates between the user and the Mirror Agent core.

2. **Mirror Agent**  
   - **LLM layer**: Model-agnostic, can be replaced with any preferred LLM backend (OpenAI GPT, Anthropic Claude, etc.).  
   - **Core logic**: Orchestrates requests between user, retriever, memory, and the reflection module.  
   - **Personality/Persona**: A helpful, wise, and Socratic guide that encourages introspection.

3. **Retriever**  
   - **Document/file ingestion**: Allows the user to connect external docs or files that the agent can reference.  
   - **Retrieval**: Looks up relevant information from user-provided or previously stored documents.

4. **Memory**  
   - **User profile and context**: Continuously updated with user interactions and introspective data.  
   - **Conversation logs**: Preserves logs of interactions for personalized, coherent responses.

5. **Reflector**  
   - **Reflection system**: Summarizes recent context, learns from user interactions, and provides introspective insights.  
   - **Integration**: Periodically surfaces reflections or prompts the user with clarifying questions based on memory and docs.

6. **LangGraph Integration**  
   - **StateGraph**: Defines the flow among the `Mirror Agent`, `Retriever`, `Memory`, `Reflector`, and user interface.  
   - **Nodes**: Each module (retriever, memory, reflection steps) is represented as a node.  
   - **Edges**: Capture the logic (e.g., “if user requests retrieval, run retriever node”).  
   - **Conditional edges**: Decide next steps (reflection, memory update, direct user response) based on context.

---

## Required Aspects

1. **LLM Model Selection**
   - Configuration for each supported LLM (e.g., OpenAI, Anthropic).
   - Ability to switch or configure the LLM in real time or at startup.

2. **Prompt Engineering**
   - System messages for setting the Mirror Agent’s style and tone (Socratic, wise, supportive).
   - Templates for reflection, retrieving context, and memory usage.

3. **Memory Management**
   - Storage of conversation history, user metadata, and reflection output.
   - Efficient updating: only relevant aspects of memory get stored or retrieved to keep context manageable.

4. **Retriever Module**
   - Interface for uploading or referencing documents/files by the user.
   - Mechanisms to index and retrieve relevant content on-demand (e.g., semantic search or keyword-based).

5. **Reflection Module**
   - Logic for summarizing and analyzing previous user interactions.
   - Mechanisms for scheduling or triggering “reflections” that incorporate the user’s personal patterns and documents.

6. **User Interface & Experience**
   - Chat, audio, or video-based user front-end.
   - Real-time feedback and conversation with the Mirror Agent.
   - Clear “explanations” or “reflections” surfaces to the user, optionally with a togglable detail level.

7. **LangGraph Implementation**
   - `langgraph.json` or equivalent configuration to specify dependencies, environment variables, and compiled graphs.
   - **Nodes** (agent node, retriever node, memory node, reflection node) and edges for controlling the flow.
   - Deployment scripts or Docker-based solution to run the final integrated system.

8. **Deployment & Hosting**
   - Option to self-host or deploy to a cloud service (e.g., LangGraph Cloud).
   - `.env` management for sensitive keys (OpenAI, Anthropic).
   - Proper logging or monitoring for usage and debugging.

---

## Phases & To-Do Checklist

### Phase 1: Requirements & Architecture
- [ ] **Define MVP Feature Scope**  
  - Identify minimal features for the Mirror Agent’s first release (single LLM, minimal memory, basic reflection).
- [ ] **Finalize High-Level Architecture**  
  - Confirm design of each module (Retriever, Memory, Reflector) and how they interconnect.
- [ ] **Specify Tech Stack**  
  - Choose which LLMs to support first, which memory storage to use, how to handle user interface.

### Phase 2: LangGraph Setup & Basic Agent
- [x] **Initialize Project Structure**  
  - [x] Create `pyproject.toml` for Python dependencies
  - [x] Configure `.env` with LLM API keys
  - [x] Create `langgraph.json` referencing compiled graphs and environment variables
- [ ] **Implement Basic Agent**  
  - [ ] Build a minimal StateGraph with a single node for the Mirror Agent that echoes user input
  - [ ] Validate local runs or partial deployment using Python-based approach

### Current Project Status
We are currently in Phase 2, with the basic project structure and configuration in place. The next immediate steps are:
1. Implementing the basic Mirror Agent with a minimal StateGraph
2. Setting up the initial agent-user interaction loop
3. Validating the basic setup with local test runs

### Phase 3: Memory Module
- [ ] **Design Conversation Storage**  
  - Decide on in-memory vs persistent storage (vector DB, local DB, etc.).
  - Create CRUD operations for storing user messages and relevant metadata.
- [ ] **Integrate Memory Node**  
  - In the LangGraph flow, incorporate a memory node that updates and retrieves conversation context.
- [ ] **Validate**  
  - Ensure the agent can recall recent interactions and maintain state across multiple user turns.

### Phase 4: Retriever & Document Ingestion
- [x] **Implement Document Loader**  
  - [x] Provide UI or endpoints for the user to upload or link docs.
  - [x] Index or store them so they can be searched later by the agent.
- [x] **Retriever Node**  
  - [x] Add a node to the LangGraph that receives the user query and returns relevant doc chunks.
  - [x] Integrate bridging logic so the Mirror Agent can request references from the retriever node on-demand.
- [x] **Test Retrieval**  
  - [x] Confirm the agent can blend retrieved content into answers or reflections.

### Current Project Status
We have completed Phase 4 and are now focusing on retrieval evaluation and improvement before moving to Phase 5. Key accomplishments:

1. Basic project structure and configuration in place
2. Mirror Agent with StateGraph implemented
3. Initial agent-user interaction loop working
4. Retriever module set up and tested successfully
5. Obsidian document indexing working
6. Basic retrieval functionality operational

The next immediate steps are:

### Phase 4.5: Retrieval Evaluation & Optimization
- [ ] **Design Evaluation Framework**
  - [ ] Define metrics for retrieval quality (precision, recall, relevance)
  - [ ] Create test cases with known expected results
  - [ ] Implement logging for retrieval performance
  
- [ ] **Systematic Testing**
  - [ ] Test with various query types (direct questions, conceptual queries, etc.)
  - [ ] Analyze retrieval patterns and failure modes
  - [ ] Document common issues and edge cases

- [ ] **Retrieval Improvements**
  - [ ] Optimize embedding strategies
  - [ ] Fine-tune retrieval parameters (k, similarity thresholds)
  - [ ] Implement query expansion or reformulation
  - [ ] Consider hybrid retrieval approaches

- [ ] **Documentation & Monitoring**
  - [ ] Document best practices for querying
  - [ ] Set up ongoing performance monitoring
  - [ ] Create dashboard for retrieval metrics

### Phase 5: Reflection Module
- [ ] **Self-Reflection Logic**  
  - Add a node that receives a summary of interactions from memory and surfaces potential insights.
- [ ] **Prompt Templates for Reflection**  
  - Introduce specialized prompts for the agent to generate deeper questions or observations about the user’s patterns.
- [ ] **Trigger Points**  
  - Decide how often or when reflections occur (e.g., time-based, conversation-based triggers).
- [ ] **User Testing**  
  - Gather feedback on the helpfulness and frequency of reflections.

### Phase 6: Final Integration & Polishing
- [ ] **Refine Orchestration**  
  - Revisit the entire flow. Ensure the agent transitions smoothly between reflection, memory updates, and normal Q&A.
- [ ] **User Interface Polishing**  
  - Make sure chat/audio/video modes are functional and intuitive.
  - Provide optional controls for users to enable/disable the reflection prompts or retrieve documents on demand.
- [ ] **Performance & Scalability**  
  - Optimize memory usage and retrieval for large documents.
  - Evaluate concurrency or session management if multiple users need parallel sessions.
- [ ] **Security & Privacy**  
  - Evaluate data encryption, user authentication, and secure storage of personal data.

### Phase 7: Testing & Deployment
- [ ] **Comprehensive QA**  
  - Test across a variety of user queries and doc retrieval scenarios.
  - Evaluate the clarity of the agent’s reflection prompts.
- [ ] **Deploy**  
  - Containerize the solution using the `langgraph.json` Docker instructions or suitable environment approach.
  - Deploy to a staging or production environment (e.g., LangGraph Cloud).
- [ ] **Observability**  
  - Monitor logs, track usage, handle errors.
  - Collect feedback from actual usage for iterative improvements.

---

## Conclusion

By following these phases, we will develop a **Mirror Agent** capable of Socratic dialogue, memory-driven personalization, on-demand retrieval of user documents, and reflective insights. The architecture relies on **LangGraph** to orchestrate the flow, ensuring modularity, extensibility, and clarity in how each part of the system (Retriever, Memory, Reflector, LLM) interacts. Once built, the agent will provide a unique and supportive user experience aimed at deeper self-understanding.