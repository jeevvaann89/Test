import asyncio
import os
import sys
from dataclasses import dataclass
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
from browser_use import Agent

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
) -> str:
    if not azure_api_key.strip() or not azure_endpoint.strip():
        return 'Please provide Azure API key and endpoint'

    os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

    try:
        agent = Agent(
            task=task,
            llm=AzureChatOpenAI(
                model=model,
            ),
        )
        result = await agent.run()
        # TODO: The result cloud be parsed better
        return result
    except Exception as e:
        return f'Error: {str(e)}'

def create_ui():
    with gr.Blocks(title='Browser Use GUI') as interface:
        gr.Markdown('# Browser Use Task Automation')
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
                submit_btn = gr.Button('Run Task')
            with gr.Column():
                output = gr.Textbox(label='Output', lines=10, interactive=False)

        submit_btn.click(
            fn=lambda *args: asyncio.run(run_browser_task(*args)),
            inputs=[task, azure_api_key, azure_endpoint, model, headless],
            outputs=output,
        )
    return interface

if __name__ == '__main__':
    demo = create_ui()
    demo.launch()
