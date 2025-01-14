### Reflection Step Prompt

Need to add a reflection step to the graph.py in mirror-agent that analyzes the conversation history and user's behavior patterns to improve future interactions. Should integrate with existing retrieval and response flow.

### Structured O1 Prompt

Add a reflection step to the retrieval graph that:
1. Analyzes conversation patterns and user behavior after each interaction
2. Stores insights about user preferences and interaction patterns
3. Uses these insights to improve future responses

Implementation should:
- Add a new `reflect` async function after the respond step
- Store reflection data in a format that can be retrieved later
- Update the graph to include reflection step
- Maintain existing flow while adding reflection capabilities

Return complete implementation including:
- Full reflect function with necessary type hints
- Graph modifications to add reflection step
- Any new state fields or types needed
- Storage mechanism for reflections

Key considerations:
- Reflection should happen after response is sent to maintain responsiveness
- Need efficient storage/retrieval of reflection data
- Should handle both successful and failed interactions
- Must preserve existing graph functionality