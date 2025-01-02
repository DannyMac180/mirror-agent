# Braintrust Documentation

Source: https://www.braintrust.dev/docs
Additional Sources: 
- https://www.braintrust.dev/docs/start
- https://www.braintrust.dev/docs/start/eval-sdk

## Overview

Braintrust is an end-to-end platform for building AI applications. It makes software development with large language models (LLMs) robust and iterative.

## Key Features

### Iterative experimentation
Rapidly prototype with different prompts and models in the playground. This allows developers to:
- Test different prompt variations
- Compare model outputs
- Iterate quickly on improvements

### Performance insights
Built-in tools to evaluate how models and prompts are performing in production, and dig into specific examples:
- Monitor model performance
- Analyze specific examples
- Track improvements over time

### Real-time monitoring
Log, monitor, and take action on real-world interactions with robust and flexible monitoring:
- Track model behavior in production
- Set up alerts and notifications
- Take action on issues in real-time

### Data management
Manage and review data to store and version your test sets centrally:
- Store test datasets
- Version control your data
- Maintain evaluation sets

## Developer Workflow

What makes Braintrust powerful is how these tools work together. With Braintrust, developers can:
1. Move faster through rapid experimentation
2. Run more experiments with integrated tooling
3. Build better AI products through systematic evaluation
4. Maintain quality through continuous monitoring

## Getting Started

To get started with Braintrust, you can:
1. Use the playground for rapid prototyping
2. Set up monitoring for your models
3. Create evaluation datasets
4. Integrate with your existing workflow

The platform provides a comprehensive set of tools that work together to make LLM development more robust and iterative.

## SDK Evaluation Guide

### Installing Braintrust Libraries

You can install the Braintrust SDK using npm/yarn (TypeScript) or pip (Python).

**TypeScript:**
```bash
npm install braintrust autoevals
# or
yarn add braintrust autoevals
```
Note: Node version >= 18 is required

**Python:**
```bash
pip install braintrust autoevals
```

### Creating an Evaluation Script

The evaluation framework allows you to declaratively define evaluations in your code. Name your files following these conventions:
- TypeScript: `*.eval.ts` or `*.eval.js`
- Python: `eval_*.py`

Here's a sample evaluation script:

**Python Example:**
```python
from braintrust import Eval
from autoevals import Levenshtein

Eval(
    "Say Hi Bot",  # Replace with your project name
    {
        "data": lambda: [
            {
                "input": "Foo",
                "expected": "Hi Foo",
            },
            {
                "input": "Bar",
                "expected": "Hello Bar",
            },
        ],  # Replace with your eval dataset
        "task": lambda input: "Hi " + input,  # Replace with your LLM call
        "scores": [Levenshtein],
    },
)
```

The script components:
- `data`: Array/iterator of evaluation data
- `task`: Function that processes input and returns output
- `scores`: Array of scoring functions for output evaluation

### Running Evaluations

1. Create an API key in the Braintrust settings page
2. Run your evaluation script:
   ```bash
   BRAINTRUST_API_KEY="YOUR_API_KEY" python eval_tutorial.py
   ```
3. View results in the experiment dashboard

### Experiment Dashboard

The experiment view provides:
- High-level performance metrics
- Individual example analysis
- Performance comparison over time
- Detailed scoring breakdowns

### Next Steps

After your first evaluation:
1. Analyze the initial score (typically around 77.8%)
2. Make improvements to your implementation
3. Re-run evaluations to track progress
4. Compare results across experiments

## Additional Resources

1. **Guides and Documentation:**
   - Comprehensive Evals guide
   - Cookbook for common use cases (RAG, summarization, text-to-sql)
   - Trace logging documentation
   - Platform architecture overview

2. **Integration Options:**
   - API documentation
   - SDK references
   - Monitoring setup
   - Custom evaluation creation
