import asyncio
import os
import sys
import json

from typing import Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_openai import AzureChatOpenAI
from browser_use import Agent as BrowserUseAgent, BrowserProfile, BrowserSession
# Import agno components for Gherkin generation
from agno.agent import Agent as AgnoAgent
from agno.models.azure import AzureOpenAI
from agno.tools.reasoning import ReasoningTools
from textwrap import dedent


# --- Pydantic Models for Agent History ---
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



def process_agent_history_json(json_string: str) -> Tuple[str, List[Image.Image]]:

    text_summary_lines: List[str] = []
    images: List[Image.Image] = []

    if not json_string.strip():
        return "No agent history data to display. Run a task first.", []

    try:
        history_data = AgentHistory(**json.loads(json_string))

        for i, entry in enumerate(history_data.history):
            text_summary_lines.append(f"## Step {i + 1}: {entry.model_output.current_state.next_goal}")
            text_summary_lines.append(
                f"**Previous Goal Evaluation:** {entry.model_output.current_state.evaluation_previous_goal}")
            text_summary_lines.append(f"**Memory:** {entry.model_output.current_state.memory}")
            text_summary_lines.append(f"**Current URL:** {entry.state.url}")
            text_summary_lines.append(f"**Current Title:** {entry.state.title}")

            if entry.result:
                text_summary_lines.append("**Action Results:**")
                for res in entry.result:
                    if res.extracted_content:
                        text_summary_lines.append(f"- Extracted Content: {res.extracted_content}")
                    if res.error:
                        text_summary_lines.append(f"- Error: {res.error}")

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

            if entry.state.screenshot:
                try:
                    image_data = base64.b64decode(entry.state.screenshot)
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
                    text_summary_lines.append(f"*(Screenshot available for this step)*")
                except Exception as img_e:
                    text_summary_lines.append(f"*(Error loading screenshot for this step: {img_e})*")

            text_summary_lines.append("\n---\n")

        full_text_summary = "\n".join(text_summary_lines)
        return full_text_summary, images

    except json.JSONDecodeError as e:
        return f"Error decoding Agent History JSON: {e}", []
    except Exception as e:
        return f"An unexpected error occurred during history processing: {e}", []


# --- Browser Task Runner Function ---
async def run_browser_task(
        task: str,
        azure_api_key: str,
        azure_endpoint: str,
        model: str = 'gpt-4-vision-preview',
        headless: bool = True,
        max_steps: int = 5,
        file: gr.File | None = None,
) -> str:
    """
    Runs the browser automation task using the Agent and returns its full history as a JSON string.
    """
    if not azure_api_key.strip() or not azure_endpoint.strip():
        return json.dumps({"error": "Please provide Azure API key and endpoint."})

    os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key
    os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

    try:
        task_description = task
        if file and file.name:
            with open(file.name, 'r', encoding='utf-8') as f:
                file_content = f.read()
            task_description += "\n\nFile Content:\n" + file_content

        use_vision_model = (model == 'gpt-4-vision-preview')

        agent = BrowserUseAgent(
            task=task_description,
            llm=AzureChatOpenAI(
                model=model,
                api_version='2024-12-01-preview'
            ),
            use_vision=use_vision_model,
            validate_output=True,
        )

        history: AgentHistory = await agent.run(max_steps=max_steps)

        if history and hasattr(history, 'model_dump_json'):
            return history.model_dump_json(indent=2)
        else:
            return json.dumps(
                {"error": "Agent history could not be serialized to JSON.", "raw_history_type": str(type(history))})

    except Exception as e:
        return json.dumps({"error": f"An error occurred during task execution: {str(e)}"})


# --- Gherkin Generator Function ---
async def generate_gherkin_scenario(
    manual_test_cases_input: str,
    gherkin_api_key: str,
    gherkin_endpoint: str,
    gherkin_model: str,
    gherkin_file_upload: gr.File | None = None
) -> str:
    """
    Generates Gherkin scenarios from manual test cases using the agno agent.
    """
    if not gherkin_api_key.strip() or not gherkin_endpoint.strip():
        return "Please provide Azure API key and endpoint for Gherkin generation."

    full_input_text = manual_test_cases_input
    if gherkin_file_upload and gherkin_file_upload.name:
        with open(gherkin_file_upload.name, 'r', encoding='utf-8') as f:
            file_content = f.read()
        full_input_text += "\n\n" + file_content # Append file content

    try:
        # Define the agno Agent for Gherkin generation
        gherkhin_agent = AgnoAgent(
            model=AzureOpenAI(
                api_key=gherkin_api_key,
                azure_endpoint=gherkin_endpoint, # Use the gherkin_endpoint parameter directly
                id=gherkin_model, # Use the model selected from UI
                api_version='2024-12-01-preview',
            ),
            markdown=True,
            description=dedent("""
            You are a highly skilled Quality Assurance (QA) expert specializing in
            converting detailed manual test cases (which are derived from user stories and
            acceptance criteria) into comprehensive, well-structured, and human-readable
            Gherkin scenarios and scenario outlines. You understand that Gherkin serves
            as living documentation and a communication tool for the whole team. Your goal
            is to create Gherkin feature files that accurately represent the desired
            behavior, are easy to understand for both technical and non-technical
            stakeholders, and serve as a solid foundation for test automation.
            """),
            instructions=dedent("""
            Analyze the provided input, which is a set of detailed manual test cases.
            Each manual test case represents a specific scenario or example of how the
            system should behave based on the original user story and its acceptance criteria.

                Your task is to convert these manual test cases into comprehensive and
                well-structured Gherkin scenarios and scenario outlines within a single
                Feature file.

                **Best Practices for Gherkin Generation:**

                1.  **Feature Description:** Start the output with a clear and concise `Feature:` description that summarizes the overall functionality being tested. This should align with the user story's main goal.
                2.  **Scenario vs. Scenario Outline:**
                    * Use a `Scenario:` for individual test cases that cover a unique flow or specific set of inputs/outcomes.
                    * Use a `Scenario Outline:` when multiple manual test cases cover the *same* workflow or steps but with *different test data* (inputs and potentially expected simple outcomes). Extract the varying data into an `Examples:` table below the Scenario Outline and use placeholders (< >) in the steps. This promotes the DRY (Don't Repeat Yourself) principle.
                3.  **Descriptive Titles:** Use clear, concise, and action-oriented titles for both `Scenario` and `Scenario Outline`, derived from the manual test case titles or descriptions. The title should quickly convey the purpose of the scenario.
                4.  **Tags:** Apply relevant and meaningful `@tags` above each Scenario or Scenario Outline (e.g., `@smoke`, `@regression`, `@login`, `@negative`, `@boundary`). Consider tags based on the test case type, priority, or related feature area to aid in test execution filtering and reporting.
                5.  **Structured Steps (Given/When/Then/And/But):**
                    * `Given`: Describe the initial context or preconditions required to perform the test (e.g., "Given the user is logged in", "Given the product is out of stock"). These set the scene. Avoid user interaction details here.
                    * `When`: Describe the specific action or event that triggers the behavior being tested (e.g., "When the user adds the item to the cart", "When invalid credentials are provided"). There should ideally be only one main `When` per scenario.
                    * `Then`: Describe the expected outcome or result after the action is performed. This verifies the behavior (e.g., "Then the item should appear in the cart", "Then an error message should be displayed"). This should directly map to the Expected Result in the manual test case.
                    * `And` / `But`: Use these to extend a previous Given, When, or Then step. `And` is typically for additive conditions or actions, while `But` can be used for negative conditions (though `And not` is often clearer). Limit the number of `And` steps to maintain readability.
                6.  **Level of Abstraction (What, Not How):** Write Gherkin steps at a high level, focusing on the *intent* and *behavior* (what the system does or what the user achieves) rather than the technical implementation details (how it's done, e.g., "click button X", "fill field Y"). Abstract away UI interactions where possible.
                7.  **Clarity and Readability:** Use plain, unambiguous language that is easy for both technical and non-technical team members to understand. Avoid technical jargon. Maintain consistent phrasing. Use empty lines to separate scenarios for better readability.
                8.  **Background:** If multiple scenarios within the feature file share the same initial preconditions, consider using a `Background:` section at the top of the feature file. This reduces repetition but ensure it doesn't make scenarios harder to understand.
                9.  **Traceability (Optional but Recommended):** If the manual test cases reference user story or requirement IDs (e.g., Jira IDs), you can include these as tags or comments (using `#`) near the Feature or Scenario title for traceability.

                Convert each relevant manual test case into one or more Gherkin scenarios/scenario outlines based on the above principles. Ensure the generated Gherkin accurately reflects the preconditions, steps, and expected results described in the manual test cases, while elevating the level of abstraction.

                **IMPORTANT:** Your final output MUST be ONLY the markdown code block containing the Gherkin feature file content. Do not include any other text, explanations, or tool calls before or after the code block.
            """),
            expected_output=dedent("""\
            Feature: [Clear and Concise Feature Description aligned with User Story]

            @tag1 @tag2
            Background:
            Given [Common precondition 1]
            And [Common precondition 2]
            # Use Background for steps repeated at the start of every scenario in the file

            @tag3
            Scenario: [Descriptive Scenario Title for a specific case]
            Given [Precondition specific to this scenario, if not in Background]
            When [Action performed by the user or system event]
            Then [Expected verifiable outcome 1]
            And [Another expected outcome, if any]

            @tag4 @tag5
            Scenario Outline: [Descriptive Title for a set of similar cases with varying data]
            Given [Precondition(s)]
            When [Action using <placeholder>]
            Then [Expected outcome using <placeholder>]
            And [Another expected outcome using <placeholder>]

            Examples:
                | placeholder1 | placeholder2 | expected_outcome_data |
                | data1_row1   | data2_row1   | outcome_data_row1     |
                | data1_row2   | data2_row2   | outcome_data_row2     |
                # Include columns for all placeholders in steps and relevant expected data

            # Include scenarios/scenario outlines for positive, negative, edge, and boundary cases
            # derived from the manual test cases.           
            Return ONLY the markdown code block containing the Gherkin feature file content.
            """),
        )
        response =  gherkhin_agent.run(full_input_text)
        return  response.content 
    except Exception as e:
        return f"Error generating Gherkin: {str(e)}"


# --- Gradio UI Creation ---
def create_unified_gui():
    """
    Creates the unified Gradio Blocks interface with Home, Agent Report, and Gherkin Generator tabs.
    """
    with gr.Blocks(title='TCS QES Automation Suite') as interface:
        agent_history_json_state = gr.State(value="")

        with gr.Tabs() as tabs:
            with gr.TabItem("Browser Automation", id="browser_tab"):
                gr.Markdown('# TCS QES Browser Automation')
                gr.Markdown("Configure LLM Configuration and define a browser automation task.")

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("LLM Configuration", open=False):
                            browser_api_key = gr.Textbox(label='Azure OpenAI API Key', placeholder='...', type='password')
                            browser_endpoint = gr.Textbox(label='Azure OpenAI Endpoint', placeholder='https://...')
                            browser_model = gr.Dropdown(choices=['gpt-35-turbo', 'gpt-4-vision-preview'], label='Model', value='gpt-35-turbo')

                        with gr.Accordion("Browser Settings", open=False):
                            with gr.Row():
                                headless = gr.Checkbox(label='Run Headless', value=True, scale=1)
                                max_steps = gr.Number(label='Max Steps', value=5, precision=0, minimum=1, scale=1)

                        browser_task = gr.Textbox(
                            label='Task Description',
                            placeholder='E.g., Find flights from New York to London for next week',
                            lines=3,
                        )
                        browser_file_upload = gr.File(label='Upload File (Optional)')

                        run_browser_task_btn = gr.Button('Run Browser Task')
                        view_report_btn = gr.Button('View Agent Report', interactive=False)

                    with gr.Column():
                        browser_task_output_status = gr.Textbox(label='Task Status / Raw Agent History JSON', lines=15, interactive=False)

                run_browser_task_btn.click(
                    fn=lambda *args: asyncio.run(run_browser_task(*args)),
                    inputs=[browser_task, browser_api_key, browser_endpoint, browser_model, headless, max_steps, browser_file_upload],
                    outputs=[browser_task_output_status],
                ).success(
                    fn=lambda x: [
                        x,
                        gr.update(interactive=True),
                        gr.update(selected="report_tab")
                    ],
                    inputs=[browser_task_output_status],
                    outputs=[agent_history_json_state, view_report_btn, tabs]
                )

                view_report_btn.click(
                    fn=lambda: gr.update(selected="report_tab"),
                    inputs=[],
                    outputs=[tabs]
                )

            with gr.TabItem("Agent Report", id="report_tab"):
                gr.Markdown("## Agent History Viewer")
                gr.Markdown("This section displays the detailed history and screenshots of the last executed browser task.")

                report_json_input = gr.Textbox(
                    label="Agent History JSON (Read-Only)",
                    lines=10,
                    interactive=False
                )
                process_report_button = gr.Button("Process History for Report")

                with gr.Row():
                    report_text_summary = gr.Markdown(label="Agent Activity Summary")
                    report_image_gallery = gr.Gallery(
                        label="Screenshots per Step",
                        height="auto",
                        columns=1,
                        rows=2,
                        object_fit="contain",
                        interactive=False
                    )

                agent_history_json_state.change(
                    fn=lambda json_str: [json_str] + list(process_agent_history_json(json_str)),
                    inputs=[agent_history_json_state],
                    outputs=[report_json_input, report_text_summary, report_image_gallery]
                )

                process_report_button.click(
                    fn=process_agent_history_json,
                    inputs=[report_json_input],
                    outputs=[report_text_summary, report_image_gallery]
                )

            with gr.TabItem("Gherkin Generator", id="gherkin_tab"):
                gr.Markdown("## Gherkin Generator")
                gr.Markdown("Convert detailed manual test cases into comprehensive Gherkin scenarios.")

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("LLM Configuration", open=False): # Open by default for Gherkin tab
                            gherkin_api_key = gr.Textbox(label='Azure OpenAI API Key', placeholder='...', type='password')
                            gherkin_endpoint = gr.Textbox(label='Azure OpenAI Endpoint', placeholder='https://...')
                            gherkin_model = gr.Dropdown(choices=['gpt-35-turbo', 'gpt-4-vision-preview'], label='Model', value='gpt-35-turbo')

                        gherkin_input_text = gr.Textbox(
                            label='Manual Test Cases Input',
                            placeholder=dedent("""
                            # Example:
                            Test Case 1: Successful login
                            - Preconditions: User has valid credentials
                            - Steps: User enters valid username and password, clicks login button
                            - Expected Result: User is logged in and redirected to dashboard
                            """).strip(),
                            lines=10,
                            info="Provide your manual test cases here, or upload a file."
                        )
                        gherkin_file_upload = gr.File(label='Upload Manual Test Cases File (Optional)')
                        run_gherkin_gen_btn = gr.Button('Generate Gherkin Scenario')
                    with gr.Column():
                        gherkin_output = gr.Code(
                            label='Generated Gherkin Syntax',
                            lines=20,
                            interactive=False,
                        )

                # Connect the Gherkin generation button to the function
                run_gherkin_gen_btn.click(
                    fn=lambda *args: asyncio.run(generate_gherkin_scenario(*args)),
                    inputs=[gherkin_input_text, gherkin_api_key, gherkin_endpoint, gherkin_model, gherkin_file_upload],
                    outputs=[gherkin_output]
                )

    return interface


# --- Main execution block ---
if __name__ == '__main__':
    # Create the unified Gradio
    demo = create_unified_gui()
    demo.launch()
