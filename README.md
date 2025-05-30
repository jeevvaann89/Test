import asyncio
import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image

# Add the parent directory to the system path to allow imports from browser_use
# This assumes the script is run from a sub-directory of the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import gradio as gr
from langchain_openai import AzureChatOpenAI
# The rich library is used for console output, but not directly in the Gradio UI.
# Keeping it for consistency with the original script, though it's not strictly needed for the GUI.
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Assuming these imports are available in the 'browser_use' package
# from browser_use.browser.context import BrowserContextConfig # Not directly used in GUI logic
from browser_use import Agent, BrowserProfile, BrowserSession


# --- Pydantic Models for Agent History (from the second code block, ensuring consistency) ---
# These models define the structure of the JSON data expected from the agent.
class InteractedElement(BaseModel):
    tag_name: Optional[str] = None
    xpath: Optional[str] = None
    highlight_index: Optional[Any] = None


class TabInfo(BaseModel):
    page_id: int
    url: str
    title: str
    parent_page_id: Optional[int] = None


class PageState(BaseModel):
    url: str
    title: str
    tabs: List[TabInfo]
    interacted_element: List[Optional[InteractedElement]]
    screenshot: Optional[str] = None  # Base64 encoded image string


class CurrentState(BaseModel):
    evaluation_previous_goal: str
    memory: str
    next_goal: str


class ModelOutput(BaseModel):
    current_state: CurrentState
    action: List[Any]  # Can be empty dict or other structures, depends on agent's action format


class ActionResultEntry(BaseModel):
    is_done: bool
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool


class AgentHistoryEntry(BaseModel):
    model_output: ModelOutput
    result: List[ActionResultEntry]
    state: PageState
    metadata: dict


class AgentHistory(BaseModel):
    history: List[AgentHistoryEntry]


# --- Agent History Processing Function for Report View ---
def process_agent_history_json(json_string: str) -> Tuple[str, List[Image.Image]]:
    """
    Parses the agent history JSON string, formats it into a human-readable Markdown summary,
    and extracts PIL Image objects from base64-encoded screenshots.

    Args:
        json_string (str): A JSON string representing the AgentHistory.

    Returns:
        Tuple[str, List[Image.Image]]: A tuple containing:
            - A Markdown formatted string summarizing the agent's steps.
            - A list of PIL Image objects extracted from the screenshots.
    """
    text_summary_lines: List[str] = []
    images: List[Image.Image] = []

    if not json_string.strip():
        return "No agent history data to display. Run a task first.", []

    try:
        # Parse the JSON string into the Pydantic AgentHistory model
        history_data = AgentHistory(**json.loads(json_string))

        # Iterate through each entry in the agent's history
        for i, entry in enumerate(history_data.history):
            # Create a Markdown header for each step, including the next goal
            text_summary_lines.append(f"## Step {i + 1}: {entry.model_output.current_state.next_goal}")
            text_summary_lines.append(
                f"**Previous Goal Evaluation:** {entry.model_output.current_state.evaluation_previous_goal}")
            text_summary_lines.append(f"**Memory:** {entry.model_output.current_state.memory}")
            text_summary_lines.append(f"**Current URL:** {entry.state.url}")
            text_summary_lines.append(f"**Current Title:** {entry.state.title}")

            # Display action results if available
            if entry.result:
                text_summary_lines.append("**Action Results:**")
                for res in entry.result:
                    if res.extracted_content:
                        text_summary_lines.append(f"- Extracted Content: {res.extracted_content}")
                    if res.error:
                        text_summary_lines.append(f"- Error: {res.error}")

            # Display metadata, specifically step duration if available
            if entry.metadata:
                step_start_time = entry.metadata.get('step_start_time')
                step_end_time = entry.metadata.get('step_end_time')
                if step_start_time is not None and step_end_time is not None:
                    try:
                        duration = float(step_end_time) - float(step_start_time)
                        text_summary_lines.append(f"**Metadata:** Step Duration: {duration:.2f}s")
                    except (ValueError, TypeError):
                        text_summary_lines.append(f"**Metadata:** Step Time: N/A (Invalid timestamps)")
                else:
                    text_summary_lines.append(f"**Metadata:** Step Time: N/A")

            # Extract and decode screenshot if present
            if entry.state.screenshot:
                try:
                    # Decode the base64 string to bytes
                    image_data = base64.b64decode(entry.state.screenshot)
                    # Open the image from bytes using PIL (Pillow)
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
                    text_summary_lines.append(f"*(Screenshot available for this step)*")
                except Exception as img_e:
                    text_summary_lines.append(f"*(Error loading screenshot for this step: {img_e})*")

            # Add a horizontal rule to separate steps in the Markdown output
            text_summary_lines.append("\n---\n")

        # Join all Markdown lines into a single string
        full_text_summary = "\n".join(text_summary_lines)
        return full_text_summary, images

    except json.JSONDecodeError as e:
        # Handle JSON parsing errors gracefully
        return f"Error decoding Agent History JSON: {e}", []
    except Exception as e:
        # Catch any other unexpected errors during processing
        return f"An unexpected error occurred during history processing: {e}", []


# --- Browser Task Runner Function ---
async def run_browser_task(
        task: str,
        azure_api_key: str,
        azure_endpoint: str,
        model: str = 'gpt-35-turbo',
        headless: bool = True,
        file: gr.File | None = None,  # Gradio File component provides a named temporary file
) -> str:  # This function now returns a JSON string of the AgentHistory
    """
    Runs the browser automation task using the Agent and returns its full history as a JSON string.

    Args:
        task (str): The description of the task for the agent.
        azure_api_key (str): Azure OpenAI API key.
        azure_endpoint (str): Azure OpenAI endpoint URL.
        model (str): The OpenAI model to use (e.g., 'gpt-35-turbo').
        headless (bool): Whether to run the browser in headless mode.
        file (gr.File | None): An uploaded file object from Gradio.

    Returns:
        str: A JSON string representation of the AgentHistory, or an error JSON string.
    """
    if not azure_api_key.strip() or not azure_endpoint.strip():
        return json.dumps({"error": "Please provide Azure API key and endpoint."})

    # Set environment variables for the Agent's LLM configuration
    os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

    try:
        task_description = task
        # If a file is uploaded, read its content and append to the task description
        if file and file.name:  # Check if file object exists and has a name (path)
            with open(file.name, 'r', encoding='utf-8') as f:  # Use file.name for Gradio File component
                file_content = f.read()
            task_description += "\n\nFile Content:\n" + file_content

        # Initialize the Agent with the task and LLM configuration
        agent = Agent(
            task=task_description,
            llm=AzureChatOpenAI(
                model=model,
                api_version='2024-12-01-preview'  # Specify API version
            ),
            use_vision=False,  # Set to True if vision capabilities are desired and configured
            validate_output=True,
            # Pass browser context config if needed, e.g.:
            # browser_context_config=BrowserContextConfig(headless=headless)
        )

        # Run the agent for a maximum of 5 steps
        history: AgentHistory = await agent.run(max_steps=5)

        # Convert the Pydantic AgentHistory model to a JSON string
        if history and hasattr(history, 'model_dump_json'):
            return history.model_dump_json(indent=2)
        else:
            # Fallback for unexpected history object types
            return json.dumps(
                {"error": "Agent history could not be serialized to JSON.", "raw_history_type": str(type(history))})

    except Exception as e:
        # Return a JSON error message if any exception occurs during task execution
        return json.dumps({"error": f"An error occurred during task execution: {str(e)}"})


# --- Gradio UI Creation ---
def create_unified_gui():
    """
    Creates the unified Gradio Blocks interface with Home and Agent Report tabs.
    """
    with gr.Blocks(title='TCS QES Browser Automation') as interface:
        # A hidden Gradio State component to store the agent history JSON output
        # This allows data to be passed between different UI interactions.
        agent_history_json_state = gr.State(value="")

        # Use gr.Tabs for navigation between the "Home" and "Agent Report" sections
        with gr.Tabs() as tabs:
            with gr.TabItem("Home", id="home_tab"):
                gr.Markdown('# TCS QES Browser Automation')
                gr.Markdown("Configure your Azure OpenAI credentials and define a browser automation task.")

                with gr.Row():
                    with gr.Column():
                        # Input fields for Azure OpenAI configuration and task
                        azure_api_key = gr.Textbox(label='Azure OpenAI API Key', placeholder='...', type='password')
                        azure_endpoint = gr.Textbox(label='Azure OpenAI Endpoint', placeholder='https://...', )
                        task = gr.Textbox(
                            label='Task Description',
                            placeholder='E.g., Find flights from New York to London for next week',
                            lines=3,
                        )
                        model = gr.Dropdown(choices=['gpt-35-turbo'], label='Model', value='gpt-35-turbo')
                        headless = gr.Checkbox(label='Run Headless', value=True)
                        file_upload = gr.File(label='Upload File (Optional)')

                        # Buttons for running the task and viewing the report
                        run_task_btn = gr.Button('Run Browser Task')
                        # The 'View Agent Report' button is initially disabled until a task is run
                        view_report_btn = gr.Button('View Agent Report', interactive=False)

                    with gr.Column():
                        # Textbox to display the raw JSON output from the browser task (for debugging/verification)
                        task_output_status = gr.Textbox(label='Task Status / Raw Agent History JSON', lines=15,
                                                        interactive=False)

                # Define the action when the 'Run Browser Task' button is clicked
                run_task_btn.click(
                    # Call the asynchronous run_browser_task function
                    fn=lambda *args: asyncio.run(run_browser_task(*args)),
                    inputs=[task, azure_api_key, azure_endpoint, model, headless, file_upload],
                    outputs=[task_output_status],  # The raw JSON output goes to task_output_status
                ).success(
                    # This .success() method is chained. It runs only if the previous function (run_browser_task)
                    # completes successfully. It takes the output of run_browser_task as its first argument (x).
                    fn=lambda x: [
                        x,  # Update agent_history_json_state with the new JSON history
                        gr.update(interactive=True),  # Enable the 'View Agent Report' button
                        gr.update(selected="report_tab")  # Automatically switch to the 'Agent Report' tab
                    ],
                    inputs=[task_output_status],  # Use the content of task_output_status as input to this lambda
                    outputs=[agent_history_json_state, view_report_btn, tabs]  # Update these components
                )

                # Define the action when the 'View Agent Report' button is clicked
                view_report_btn.click(
                    # Simply switch the active tab to the 'Agent Report' tab
                    fn=lambda: gr.update(selected="report_tab"),
                    inputs=[],
                    outputs=[tabs]
                )

            with gr.TabItem("Agent Report", id="report_tab"):
                gr.Markdown("## Agent History Viewer")
                gr.Markdown(
                    "This section displays the detailed history and screenshots of the last executed browser task.")

                # Textbox to display the raw Agent History JSON (read-only)
                report_json_input = gr.Textbox(
                    label="Agent History JSON (Read-Only)",
                    lines=10,
                    interactive=False  # This textbox is for display only
                )
                # Button to manually process the history (useful if user pastes JSON directly)
                process_report_button = gr.Button("Process History for Report")

                with gr.Row():
                    # Markdown component to display the formatted text summary of agent steps
                    report_text_summary = gr.Markdown(label="Agent Activity Summary")
                    # Gallery to display screenshots captured during the agent's execution
                    report_image_gallery = gr.Gallery(
                        label="Screenshots per Step",
                        height="auto",
                        columns=1,  # Display one image per row in the gallery
                        rows=2,
                        object_fit="contain",
                        interactive=False  # Gallery is for display only
                    )

                # Automatically update the report view whenever the agent_history_json_state changes
                # This ensures the report is populated as soon as the task completes
                agent_history_json_state.change(
                    fn=lambda json_str: [json_str] + list(process_agent_history_json(json_str)),
                    inputs=[agent_history_json_state],
                    outputs=[report_json_input, report_text_summary, report_image_gallery]
                )

                # Allow manual processing of the report if the user clicks the button
                # This is useful if the user manually edits the JSON in report_json_input
                process_report_button.click(
                    fn=process_agent_history_json,
                    inputs=[report_json_input],
                    outputs=[report_text_summary, report_image_gallery]
                )

    return interface


# --- Main execution block ---
if __name__ == '__main__':
    # Create the unified Gradio interface
    demo = create_unified_gui()
    # Launch the Gradio application
    demo.launch()
