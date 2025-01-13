# O1 Coding Prompt Template

## Input Format
Paste your coding brainstorm/thoughts below:
```
[Your brainstorm here]
```

## Output Format

### Goal
[One clear sentence stating what you want to achieve]

### Return Format
- Return the complete working code solution
- Include all necessary imports and dependencies
- Specify file structure and locations
- Note any environment setup requirements
- Flag any potential gotchas or edge cases

### Warnings
- Highlight any performance considerations
- Note any security implications
- List potential breaking changes
- Specify version dependencies
- Flag any rate limits or API constraints

### Context Dump
- Current codebase structure and relevant files
- Related systems or services this interacts with
- Previous solutions or attempts
- Performance requirements
- Scale considerations
- Team conventions or preferences to follow
- Any existing patterns to maintain

---

## Example

### Input Brainstorm
```
Need to build something that can watch a directory and sync new markdown files to a database. Should probably use watchdog for the file system stuff. Want it to be efficient and not re-process files unnecessarily.
```

### Structured O1 Prompt

I need a file watcher system that syncs markdown files to a database.

For the implementation, return a complete solution that uses the watchdog library to monitor a directory for markdown file changes. The system should efficiently track which files have been processed to avoid redundant operations.

Return the code structure including:
- Main watcher class implementation
- Database integration layer
- File processing utilities
- Configuration handling

Be careful to handle:
- File system edge cases (moves, renames, deletions)
- Proper database connection management
- Efficient diff checking to avoid reprocessing unchanged files
- Graceful shutdown

For context: This will be part of a larger document processing pipeline. The system needs to scale to handle thousands of files and maintain consistency between the file system and database state. We use SQLAlchemy for other database operations and follow PEP-8 style guidelines.
