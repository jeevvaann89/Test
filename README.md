import asyncio
import os
import sys
import json
from typing import Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image
import pytest  # Import pytest for test functions and fixtures

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv

load_dotenv()
import allure  # Import Allure


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
    screenshot: Optional[str] = None


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


def process_agent_history_json(json_string: str) -> Tuple[str, List[Image.Image]]:
    text_summary_lines: List[str] = []
    images: List[Image.Image] = []

    # This 'with allure.step' will be nested inside the pytest test's allure scope
    with allure.step("Processing Agent Execution History Details"):
        if not json_string.strip():
            allure.attach("No agent history data provided.", name="History Status",
                          attachment_type=allure.attachment_type.TEXT)
            return "No agent history data to display. Run a task first.", []

        try:
            history_data = AgentHistory(**json.loads(json_string))

            allure.attach(json_string, name="Raw Agent History JSON", attachment_type=allure.attachment_type.JSON)

            for i, entry in enumerate(history_data.history):
                step_title = f"Step {i + 1}: {entry.model_output.current_state.next_goal}"
                with allure.step(step_title):  # Create an Allure step for each agent step

                    # Attach key details as text attachments
                    allure.attach(
                        f"Previous Goal Evaluation: {entry.model_output.current_state.evaluation_previous_goal}",
                        name="Previous Goal Evaluation",
                        attachment_type=allure.attachment_type.TEXT
                    )
                    allure.attach(
                        f"Memory: {entry.model_output.current_state.memory}",
                        name="Agent Memory",
                        attachment_type=allure.attachment_type.TEXT
                    )
                    allure.attach(
                        f"Current URL: {entry.state.url}",
                        name="Current URL",
                        attachment_type=allure.attachment_type.TEXT
                    )
                    allure.attach(
                        f"Current Title: {entry.state.title}",
                        name="Current Title",
                        attachment_type=allure.attachment_type.TEXT
                    )

                    if entry.model_output.action:
                        allure.attach(
                            json.dumps(entry.model_output.action, indent=2),
                            name="Agent Action",
                            attachment_type=allure.attachment_type.JSON
                        )

                    # if entry.result:
                    #     action_results_summary = []
                    #     for res in entry.result:
                    #         if res.extracted_content:
                    #             action_results_summary.append(f"Extracted Content: {res.extracted_content}")
                    #         if res.error:
                    #             action_results_summary.append(f"Error: {res.error}")
                    #     if action_results_summary:
                    #         allure.attach(
                    #             "\n".join(action_results_summary),
                    #             name="Action Results",
                    #             attachment_type=allure.attachment_type.TEXT
                    #         )
                    if entry.result:
                        action_results_summary = []
                        for res in entry.result:
                            status_icon = "✅" if res.success else "❌"
                            result_line = f"{status_icon} Result: Success={res.success}"
                            if res.extracted_content:
                                result_line += f", Content: {res.extracted_content}"
                            if res.error:
                                result_line += f", Error: {res.error}"
                            action_results_summary.append(result_line)
                        if action_results_summary:
                            allure.attach(
                                "\n".join(action_results_summary),
                                name="Action Results",
                                attachment_type=allure.attachment_type.TEXT
                            )

                    if entry.metadata:
                        step_start_time = entry.metadata.get('step_start_time')
                        step_end_time = entry.metadata.get('step_end_time')
                        if step_start_time is not None and step_end_time is not None:
                            try:
                                duration = float(step_end_time) - float(step_start_time)
                                allure.attach(
                                    f"Step Duration: {duration:.2f}s",
                                    name="Step Duration",
                                    attachment_type=allure.attachment_type.TEXT
                                )
                            except (ValueError, TypeError):
                                allure.attach(f"Step Time: N/A (Invalid timestamps)", name="Step Duration",
                                              attachment_type=allure.attachment_type.TEXT)
                        else:
                            allure.attach(f"Step Time: N/A", name="Step Duration",
                                          attachment_type=allure.attachment_type.TEXT)
                        # Attach full metadata as JSON
                        allure.attach(json.dumps(entry.metadata, indent=2), name="Step Metadata",
                                      attachment_type=allure.attachment_type.JSON)

                    # --- Attach Screenshot ---
                    if entry.state.screenshot:
                        try:
                            image_data = base64.b64decode(entry.state.screenshot)
                            image = Image.open(BytesIO(image_data))
                            images.append(image)  # Keep for the return value

                            img_byte_arr = BytesIO()
                            image.save(img_byte_arr, format='JPEG')
                            final_img_bytes = img_byte_arr.getvalue()

                            allure.attach(
                                final_img_bytes,
                                name=f"Screenshot After Step {i + 1}",
                                attachment_type=allure.attachment_type.JPG
                            )
                        except Exception as img_e:
                            allure.attach(f"Error loading screenshot: {img_e}", name="Screenshot Error",
                                          attachment_type=allure.attachment_type.TEXT)

                    text_summary_lines.append("\n---\n")

            full_text_summary = "\n".join(text_summary_lines)
            return full_text_summary, images

        except json.JSONDecodeError as e:
            allure.attach(f"Error decoding Agent History JSON: {e}", name="JSON Decode Error",
                          attachment_type=allure.attachment_type.TEXT)
            return f"Error decoding Agent History JSON: {e}", []
        except Exception as e:
            allure.attach(f"An unexpected error occurred during history processing: {e}", name="Unexpected Error",
                          attachment_type=allure.attachment_type.TEXT)
            return f"An unexpected error occurred during history processing: {e}", []


# --- Pytest test function to run the history processing ---
@allure.title("Agent History Allure Report Generation")
@allure.feature("Agent Output Reporting")
@allure.story("Generate Detailed Allure Report from Agent History JSON")
@pytest.mark.parametrize("file_name", ["output_json_2025-06-12_16-42-41.json"])  # Use a fixture or direct path
def test_agent_history_allure_reporting(file_name: str):
    """
    Reads agent history from a JSON file and processes it to generate a detailed Allure report.
    """
    report_folder = 'Report'

    if not os.path.exists(report_folder):
        print(f"Error: The folder '{report_folder}' does not exist.")

    else:

        for filename in os.listdir(report_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(report_folder, filename)
                try:

                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_content = f.read()

                    summary_text, captured_images = process_agent_history_json(json_content)
                    print(f"\n--- Agent History Processing Summary for {filename} ---")
                    # print(json_content)
                    print(summary_text)
                    print(f"\nTotal screenshots captured: {len(captured_images)} for {filename}")

                    if "\"success\": false" in json_content:  # Simple check if any failure icon was added to summary
                        pytest.fail("One or more agent history steps reported failures.")

                except FileNotFoundError:
                    # This specific error might occur if a file is deleted between listdir and open
                    print(f"Error: The file '{filepath}' was not found during processing in the loop.")
                except json.JSONDecodeError as e:
                    print(f"Error: Could not decode JSON from '{filepath}'. Check file content. Details: {e}")

    #
    #         # You can add assertions here if you want the test to fail based on summary content
    #         # For example, if summary_text indicates a critical error:
    #         # assert "Error decoding Agent History JSON" not in summary_text, "Failed to decode JSON."
