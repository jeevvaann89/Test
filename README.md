import asyncio
import os
import sys
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

    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            trace_path=r'C:\\Users\\jeevan\\OneDrive\\Documents\\book\\python\\BrowserUse\\traces',
            save_recording_path=r'C:\\Users\\jeevan\\OneDrive\\Documents\\book\\python\\BrowserUse\\recording'
        )
    )
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
            browser_session=browser_session
        )
        result = await agent.run(max_steps=5)
        # TODO: The result cloud be parsed better
        return result
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
                submit_btn.click(
                    fn=lambda *args: asyncio.run(run_browser_task(*args)),
                    inputs=[task, azure_api_key, azure_endpoint, model, headless, file_upload],
                    outputs=output,
                )
    return interface

if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
