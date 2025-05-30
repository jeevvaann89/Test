import gradio as gr
import json
from typing import Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image


# --- Pydantic Models (as provided) ---
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
    action: List[Any]  # Can be empty dict or other structures


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


# --- Processing Function ---
def process_json(json_string: str) -> Tuple[str, List[Image.Image]]:
    """
    Parses the agent history JSON, formats it into a Markdown summary,
    and extracts PIL Image objects from screenshots.
    """
    text_summary_lines: List[str] = []
    images: List[Image.Image] = []

    try:
        history_data = AgentHistory(**json.loads(json_string))

        for i, entry in enumerate(history_data.history):
            # --- Markdown for Collapsible-like Steps ---
            text_summary_lines.append(f"## Step {i + 1}: {entry.model_output.current_state.next_goal}")
            text_summary_lines.append(
                f"**Previous Goal Evaluation:** {entry.model_output.current_state.evaluation_previous_goal}")
            text_summary_lines.append(f"**Memory:** {entry.model_output.current_state.memory}")
            text_summary_lines.append(f"**Current URL:** {entry.state.url}")
            text_summary_lines.append(f"**Current Title:** {entry.state.title}")

            # Action Results
            if entry.result:
                text_summary_lines.append("**Action Results:**")
                for res in entry.result:
                    if res.extracted_content:
                        text_summary_lines.append(f"- Extracted Content: {res.extracted_content}")
                    if res.error:
                        text_summary_lines.append(f"- Error: {res.error}")

            # Metadata (optional, can be expanded)
            if entry.metadata:
                text_summary_lines.append(
                    f"**Metadata:** Step Time: {entry.metadata.get('step_end_time', 'N/A') - entry.metadata.get('step_start_time', 'N/A'):.2f}s")

            # --- Screenshot Extraction ---
            if entry.state.screenshot:
                try:
                    # The JSON shows a direct base64 string, not a data URI prefix.
                    # If it were 'data:image/png;base64,...', you'd split by ','
                    # For a raw base64 string:
                    image_data = base64.b64decode(entry.state.screenshot)
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
                    text_summary_lines.append(f"*(Screenshot available for this step)*")
                except Exception as img_e:
                    text_summary_lines.append(f"*(Error loading screenshot for this step: {img_e})*")

            text_summary_lines.append("\n---\n")  # Separator between steps

        full_text_summary = "\n".join(text_summary_lines)
        return full_text_summary, images

    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}", []
    except Exception as e:
        return f"An unexpected error occurred during processing: {e}", []


# --- Gradio Interface ---
def create_gui():
    """Creates the Gradio Blocks interface for agent history viewer."""
    with gr.Blocks(title="Agent History Viewer") as demo:
        gr.Markdown("# Agent Browser History Viewer")
        gr.Markdown("Paste your agent's history JSON below to visualize its steps and screenshots.")

        with gr.Row():
            json_input = gr.Textbox(
                label="Agent History JSON",
                placeholder="Paste your JSON here...",
                lines=20,
                interactive=True
            )

        process_button = gr.Button("Process History")

        with gr.Row():
            # Changed to gr.Markdown for formatted text output
            text_output = gr.Markdown(
                label="Agent Activity Summary",
                # interactive=False
            )
            # gr.Gallery for displaying multiple images
            image_output = gr.Gallery(
                label="Screenshots per Step",
                height="auto",
                columns=1,  # Display one image per row in the gallery
                rows=2,
                object_fit="contain",
                interactive=False
            )

        # Link the button click to the processing function
        process_button.click(
            fn=process_json,
            inputs=[json_input],
            outputs=[text_output, image_output]
        )

    return demo


# --- Main execution ---
if __name__ == "__main__":
    demo = create_gui()
    demo.launch()
