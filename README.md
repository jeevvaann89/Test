import asyncio
import os
import sys
import json
from dataclasses import dataclass

from browser_use.browser.context import BrowserContextConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Third-party imports
import gradio as gr
from langchain_openai import AzureChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local module imports
from browser_use import Agent, BrowserProfile, BrowserSession


@dataclass
class ActionResult:
    is_done: bool
    extracted_content: str | None
    error: str | None
    include_in_memory: bool
    screenshot_base64: str | None = None


@dataclass
class AgentHistoryList:
    all_results: list[ActionResult]
    all_model_outputs: list[dict]

def parse_agent_history(history_str: str) -> None:
    console = Console()
    # Split the content into sections based on ActionResult entries
    sections = history_str.split('ActionResult(')
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract relevant information
        content = ''
        if 'extracted_content=' in section:
            content = section.split('extracted_content=')[1].split(',')[0].strip("'")
        if content:
            header = Text(f'Step {i}', style='bold blue')
            panel = Panel(content, title=header, border_style='blue')
            console.print(panel)
            console.print()

async def run_browser_task(
    task: str,
    azure_api_key: str,
    azure_endpoint: str,
    model: str = 'gpt-35-turbo',
    headless: bool = True,
    file: str | None = None,
) -> str:
    if not azure_api_key.strip() or not azure_endpoint.strip():
        return 'Please provide Azure API key and endpoint'
    os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

    try:
        if file:
            # Process the uploaded file
            with open(file, 'r') as f:
                file_content = f.read()
            # Use the file content as needed
            task_description = task + "\n\nFile Content:\n" + file_content
        else:
            task_description = task
        agent = Agent(
            task=task_description,
            llm=AzureChatOpenAI(
                model=model,
                api_version='2024-12-01-preview'
            ),
            use_vision=False,
            validate_output=True,

        )
        history: AgentHistoryList = await agent.run(max_steps=5)
        # --- Diagnostic Print Statements ---
        print(f"Type of history object: {type(history)}")
        print(f"Attributes of history object: {dir(history)}")
        # --- End Diagnostic Print Statements ---

        # Extract screenshots from history
        screenshots_base64 = []
        if history and hasattr(history, 'all_model_outputs') and history.all_model_outputs:  # Added hasattr check
            for output_entry in history.all_model_outputs:
                # Assuming screenshots are stored under a key like 'screenshot_base64'
                # and are already in data URI format (e.g., 'data:image/png;base64,...')
                if 'screenshot_base64' in output_entry and output_entry['screenshot_base64']:
                    screenshots_base64.append(output_entry['screenshot_base64'])
                # If ActionResult itself had a screenshot field, you'd check history.all_results too
                for result_entry in history.all_results:
                    if result_entry.screenshot_base64:
                        screenshots_base64.append(result_entry.screenshot_base64)

        # Convert history object to a readable string for the Textbox output
        # Use json.dumps for pretty printing the JSON representation of the history
        # history.model_dump_json() returns a JSON string, so we parse it then dump it for pretty print
        # Added a check for model_dump_json() existence
        if hasattr(history, 'model_dump_json'):
            history_dict = json.loads(history.model_dump_json())
            output_text = f"Agent History:\n{json.dumps(history_dict, indent=2)}"
        else:
            output_text = f"Agent History (raw): {history}"  # Fallback if not a Pydantic model

        return output_text, screenshots_base64
    except Exception as e:
        return f'Error: {str(e)}'

def create_ui():
    with gr.Blocks(title='Browser Use GUI') as interface:
        gr.Markdown('# TCS QES')
        with gr.Row():
            with gr.Column():
                azure_api_key = gr.Textbox(label='Azure OpenAI API Key', placeholder='...', type='password')
                azure_endpoint = gr.Textbox(label='Azure OpenAI Endpoint', placeholder='https://...',)
                task = gr.Textbox(
                    label='Task Description',
                    placeholder='E.g., Find flights from New York to London for next week',
                    lines=3,
                )
                model = gr.Dropdown(choices=['gpt-35-turbo'], label='Model', value='gpt-35-turbo')
                headless = gr.Checkbox(label='Run Headless', value=True)
                file_upload = gr.File(label='Upload File')
                submit_btn = gr.Button('Run Task')
            with gr.Column():
                output = gr.Textbox(label='Output', lines=10, interactive=False)
                screenshot_gallery = gr.Gallery(label='Screenshots')
                submit_btn.click(
                    fn=lambda *args: asyncio.run(run_browser_task(*args)),
                    inputs=[task, azure_api_key, azure_endpoint, model, headless, file_upload],
                    outputs=[output, screenshot_gallery],
                )
    return interface


if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
