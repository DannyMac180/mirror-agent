# O1 Reasoning Node Prompt Template

## Input Format
Paste your reasoning node brainstorm/thoughts below:
```
[Your brainstorm here]
```

## Output Format

### Goal
Create a conditional reasoning node that processes complex user queries requiring logic, scientific research, math, coding or data analysis using Gemini 2.0 Flash Thinking Exp 1-21.

### Return Format
- Return complete implementation of the reasoning node
- Include integration with existing retrieval pipeline
- Specify reasoning trace format in responses
- Define clear conditions for when reasoning should be triggered
- Include all necessary imports and model configs

### Warnings
- Handle model API rate limits and timeouts
- Consider memory usage for long reasoning chains
- Validate model outputs for hallucinations
- Ensure proper error handling for model failures
- Monitor reasoning performance and latency

### Context Dump
- Node runs after retrieval step in pipeline
- Uses Gemini 2.0 Flash Thinking Exp 1-21 model
- Must return both answer and reasoning trace
- Should integrate with existing graph structure
- Follow project's error handling patterns
- Consider future model version upgrades

---

## Example

### Input Brainstorm
```
Want to create a 'reasoning' node in the graph. Should be a conditional node that get called AFTER the retrieval step, in the case where the user message requires reasoning - things like complicated questions that require logic, scientific research, math, coding or data analysis. Model to use is Gemini 2.0 Flash Thinking Exp 1-21. Should also return the reasoning trace or 'thought' process in the response to the user.
```

### Structured O1 Prompt

I need a reasoning node implementation that conditionally processes complex user queries using Gemini 2.0 Flash Thinking.

For the implementation, return a complete solution that:
- Creates a ReasoningNode class extending base node structure
- Implements logic to detect when reasoning is needed
- Integrates with Gemini 2.0 Flash Thinking model
- Captures and formats reasoning traces
- Handles model interactions and retries
- Returns structured responses with both answers and thought process

Be careful to handle:
- Model API failures and retries
- Input validation and sanitization
- Memory efficient reasoning chains
- Clear trace formatting
- Integration with retrieval results

For context: This node runs after retrieval in the processing pipeline. It needs to detect when a query requires deeper reasoning and augment retrieval results with logical analysis. The system should maintain high performance while adding this additional processing step.
