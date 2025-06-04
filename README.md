import asyncio
import os
import sys
import pytest
import allure
import json
from browser_use.agent.views import (
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.context import BrowserSession
# from browser_use.browser.views import BrowserState
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel
import io
from pprint import pprint
from PIL import Image
from typing import List, Dict
from typing import Optional, Any
import base64

from Allure import process_agent_history_json

# Add the project root to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from browser_use import ActionResult, Agent, Controller

controller = Controller()

azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')


@allure.title("Automation")
@allure.feature("Agent Interaction")  # Add feature
@allure.story("Output Validation")  # Add story
@pytest.mark.asyncio
async def test_main():  # Changed function name to test_main for pytest


    TASKSauce = """
    1. Go to https://www.saucedemo.com/ and enter username as standard_user and password as secret_sauce and click on login button
    2. Add {Backpack} to the cart and do checkout
     """

    llmnew = AzureChatOpenAI(
        model='gpt-35-turbo',
        api_key=azure_openai_api_key,
        azure_endpoint=azure_openai_endpoint,  # Corrected to use azure_endpoint instead of openai_api_base
        api_version='2024-12-01-preview',  # Explicitly set the API version here
    )
    agent = Agent(
        task=TASKSauce,
        llm=llmnew,
        # message_context='verify the launch page',
        max_actions_per_step='1',
        max_failures='1',
        controller=controller,
        use_vision=False,
        validate_output=True,

    )
    # Now on_step is in scope and can be assigned
    # agent.on_step = on_step

    with allure.step("Agent run with validation"):
        try:
            history: AgentHistoryList = await agent.run(max_steps=5)
            # await agent.run(max_steps=5)
        except Exception as e:
            print("Error Details : ", e)
            allure.attach(str(e), "Error Details")
            raise e

    data = json.dumps(history.model_dump_json())

    process_agent_history_json(data)
    allure.title("feature")

if __name__ == '__main__':
    asyncio.run(test_main())

    =============================================================

    # --- Pydantic Models for Agent History ---
from typing import Optional

from pydantic import BaseModel
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
import allure


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

    with allure.step("Check if JSON string is empty"):
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


if __name__ == '__main__':
    process_agent_history_json(str)


    ==========================================================

    @self.registry.action("Verify text {text} is present on the page")
async def verify_text_presence(text: str, browser: BrowserContext):
    page_text = await browser.page.content()
    if text in page_text:
        return ActionResult(
            extracted_content=f"Text '{text}' is present on the page",
            success=True
        )
    else:
        return ActionResult(
            extracted_content=f"Text '{text}' is not present on the page",
            success=False
        )

