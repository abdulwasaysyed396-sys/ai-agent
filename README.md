# AI Agent with Tools

This project was built as part of my Generative AI Internship at Codomax Digital Solutions.

## Project Overview

This project demonstrates the basic concept of an AI agent.

Unlike a simple chatbot, the application can use tools to perform specific tasks.

The agent currently supports:

- Normal AI conversations
- Calculator tool
- Current time tool
- Hugging Face LLM integration
- Gradio web interface

## Architecture

User
↓
AI Agent
↓
Tool Selection
├── Calculator Tool
├── Time Tool
└── LLM Response

## Technologies Used

- Python
- Hugging Face Inference API
- Llama 3.1 8B Instruct
- Gradio
- python-dotenv

## How to Run

1. Clone the repository.
2. Create a Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
