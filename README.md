# AI Task Agent

An AI Agent built as part of the **Codomax Generative AI Internship – Module 5: AI Agents, Tools & Automation**.

The project demonstrates how an AI agent can understand a user's request, decide whether a tool is required, call the appropriate tool, process the result, and generate a final response.

## Features

* LLM-powered AI Agent
* Hugging Face Inference API
* Function / Tool Calling
* Calculator tool
* Text Length tool
* Agent decision-making
* Conversation state
* Multi-step tool execution
* Error handling
* Gradio web interface

## Agent Workflow

```text
User Task
    ↓
Understand Task
    ↓
Decide Action
    ↓
Choose Tool
    ↓
Execute Tool
    ↓
Process Result
    ↓
Generate Final Response
```

## Available Tools

### 1. Calculator

The calculator tool can perform:

* Addition
* Subtraction
* Multiplication
* Division

Example:

```text
User: What is 125 + 375?

Agent → Calculator Tool → 500

Final Answ
```
