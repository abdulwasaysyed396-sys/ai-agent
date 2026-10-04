import os
import json

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN was not found. "
        "Make sure your .env file contains HF_TOKEN=your_token"
    )


# =========================================================
# 2. CREATE HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    api_key=HF_TOKEN
)


# =========================================================
# 3. MODEL
# =========================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


# =========================================================
# 4. TOOL 1 - CALCULATOR
# =========================================================

def calculator(a, b, operation):

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            return "Error: Cannot divide by zero."

        return a / b

    else:
        return "Error: Unknown operation."


# =========================================================
# 5. TOOL 2 - TEXT LENGTH
# =========================================================

def text_length(text):

    return len(text)


# =========================================================
# 6. DESCRIBE TOOLS TO THE LLM
# =========================================================

tools = [

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Perform mathematical calculations. "
                "Use this tool for addition, subtraction, "
                "multiplication, or division."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "a": {
                        "type": "number",
                        "description": "The first number."
                    },

                    "b": {
                        "type": "number",
                        "description": "The second number."
                    },

                    "operation": {
                        "type": "string",
                        "enum": [
                            "add",
                            "subtract",
                            "multiply",
                            "divide"
                        ],
                        "description": "The mathematical operation."
                    }

                },
                "required": [
                    "a",
                    "b",
                    "operation"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "text_length",
            "description": (
                "Count the number of characters "
                "in a piece of text."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "text": {
                        "type": "string",
                        "description": "The text to count."
                    }

                },
                "required": [
                    "text"
                ]
            }
        }
    }

]


# =========================================================
# 7. EXECUTE TOOLS
# =========================================================

def execute_tool(tool_name, arguments):

    try:

        if tool_name == "calculator":

            return calculator(
                arguments["a"],
                arguments["b"],
                arguments["operation"]
            )

        elif tool_name == "text_length":

            return text_length(
                arguments["text"]
            )

        else:

            return f"Unknown tool: {tool_name}"

    except Exception as error:

        return f"Tool error: {str(error)}"


# =========================================================
# 8. AI AGENT
# =========================================================

def run_agent(user_input, conversation_history=None):

    messages = []

    # Add previous conversation to state
    if conversation_history:

        for item in conversation_history:

            if isinstance(item, (list, tuple)) and len(item) == 2:

                user_message, assistant_message = item

                if user_message:
                    messages.append(
                        {
                            "role": "user",
                            "content": user_message
                        }
                    )

                if assistant_message:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_message
                        }
                    )

    # Add current user request
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # =====================================================
    # AGENT LOOP
    # =====================================================

    max_iterations = 5

    for _ in range(max_iterations):

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=500
        )

        assistant_message = response.choices[0].message


        # No tool required
        if not assistant_message.tool_calls:

            return assistant_message.content


        # Save assistant tool request
        messages.append(assistant_message)


        # Execute requested tools
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                arguments = {}


            # Execute the selected tool
            result = execute_tool(
                tool_name,
                arguments
            )


            # Send tool result back to LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }
            )


    return (
        "The agent reached its maximum number of steps "
        "without completing the task."
    )


# =========================================================
# 9. GRADIO CHAT FUNCTION
# =========================================================

def chat(message, history):

    try:

        answer = run_agent(
            user_input=message,
            conversation_history=history
        )

        return answer

    except Exception as error:

        return (
            "Sorry, something went wrong.\n\n"
            f"Error: {str(error)}"
        )


# =========================================================
# 10. GRADIO INTERFACE
# =========================================================

demo = gr.ChatInterface(
    fn=chat,
    title="AI Task Agent",
    description=(
        "An AI agent that understands requests, "
        "chooses tools, executes them, and uses "
        "the results to generate an answer."
    ),
    examples=[
        "What is 125 + 375?",
        "Calculate 50 multiplied by 8.",
        "What is 100 divided by 4?",
        "How many characters are in Hello World?",
        "What is 20 + 30, and then multiply the result by 2?"
    ]
)


# =========================================================
# 11. START APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch()
