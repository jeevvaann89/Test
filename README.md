import asyncio
import os
import sys
import pytest
import allure
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Union, Any
from browser_use import Agent, Controller
from datetime import datetime
from browser_use.agent.views import ActionResult, AgentHistoryList
from browser_use.browser import BrowserProfile, BrowserSession, Browser  # Import Browser for type hinting
from playwright.async_api import Page
from AllureReportDuplicate import process_agent_history_json


class HoverAction(BaseModel):
    index: Optional[int] = None
    xpath: Optional[str] = None
    selector: Optional[str] = None


class VerifyTextPresenceAction(BaseModel):
    text: str
    case_sensitive: Optional[bool] = True

class VerifyTextPresenceActionElement(BaseModel):
    """
    Parameters for verifying text presence within a specific element.
    Requires text and exactly one locator.
    """
    text: str = Field(description="The text string to verify within the specified element.")
    case_sensitive: Optional[bool] = Field(False, description="Whether the text verification should be case-sensitive.")
    index: Optional[int] = Field(None, description="Index of the element in the cached selector map to verify text within.")
    xpath: Optional[str] = Field(None, description="XPath to locate the element to verify text within.")
    selector: Optional[str] = Field(None, description="CSS selector to locate the element to verify text within.")

    # Custom validation for mutually exclusive locators and ensuring at least one is provided
    @classmethod
    def __pydantic_validator__(cls):
        # This is a Pydantic v2 way of adding a validator.
        # For Pydantic v1, you'd typically use a @validator or override __init__ or model_validate.
        # I'll use a simple post-init check for broader compatibility, as Pydantic's internal
        # validator for @classmethod is more complex.
        return super().__pydantic_validator__() # Call parent validator

    def model_post_init(self, __context: Any) -> None:
        # Pydantic v2 style for post-init validation
        # For Pydantic v1, you might put this logic in a @root_validator or similar.
        provided_locators_count = sum([
            1 if self.index is not None else 0,
            1 if self.xpath is not None else 0,
            1 if self.selector is not None else 0
        ])

        if provided_locators_count > 1:
            raise ValueError("Only one locator (index, xpath, or selector) can be provided for text verification.")
        if provided_locators_count == 0:
            raise ValueError("At least one locator (index, xpath, or selector) must be provided when verifying text by element.")


from dotenv import load_dotenv

browser_profile = BrowserProfile(
    headless=False,
)

controller = Controller()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')



@controller.registry.action(
    'Verify text {text} is present on the page',
    param_model=VerifyTextPresenceAction,
)
async def verify_text_presence(params: VerifyTextPresenceAction, browser: BrowserSession) -> ActionResult:
    try:
        page_content = await browser.get_page_html()
        target_text = params.text
        content_to_search = page_content
        target_text = target_text.strip()
        content_to_search = content_to_search.strip()
        if not params.case_sensitive:
            target_text = target_text.lower()
            content_to_search = content_to_search.lower()
        if target_text in content_to_search:
            msg = f"✅ Text '{params.text}' is present on the page."
            return ActionResult(
                extracted_content=msg,
                success=True,
                include_in_memory=False
            )
        else:
            err_msg = f"❌ Text '{params.text}' is NOT present on the page."
    except Exception as e:
        err_msg = f"An error occurred while verifying text: {str(e)}"
        return ActionResult(extracted_content=err_msg, success=False, error=err_msg, include_in_memory=True)

class GetAttrUrlParams(BaseModel):
    text: str = Field(description="The visible text content used to locate the element.")
    attribute_name: str = Field(description="The name of the attribute to retrieve (e.g., 'href', 'id', 'class').")


class ScrollByPixelsParams(BaseModel):
    pixels: int = Field(description="The number of pixels to scroll down the page.")

@controller.registry.action(
    'Scroll down by {pixels} pixels',
    param_model=ScrollByPixelsParams,
)
async def scroll_by_pixels(params: ScrollByPixelsParams, browser_session: BrowserSession) -> ActionResult:

    try:
        page = await browser_session.get_current_page()
        await page.evaluate(f"window.scrollBy(0, {params.pixels})")

        msg = f"⬇️ Successfully scrolled down by {params.pixels} pixels."

        return ActionResult(
            extracted_content=msg,
            success=True,
            include_in_memory=True # Include in memory as it's a performed action
        )
    except Exception as e:
        err_msg = f"❌ Failed to scroll down by {params.pixels} pixels: {str(e)}"
        return ActionResult(extracted_content=err_msg, success=False, error=err_msg, include_in_memory=True)

@controller.registry.action(
    'Switch to new tab'
)
async def scroll_by_pixels(browser_session: BrowserSession) -> ActionResult:
    try:
        page = await browser_session.browser_context.new_page()
        await page.bring_to_front()
        return ActionResult(
            success=True,
            include_in_memory=True
        )
    except Exception as e:
        err_msg = f"❌ Failed to go to new tab: {str(e)}"
        return ActionResult(extracted_content=err_msg, success=False, error=err_msg, include_in_memory=True)


# @allure.title("Automation")
# @allure.feature("Agent Interaction")
# @allure.story("Output Validation")
@pytest.mark.asyncio
async def main():

    browser_session = BrowserSession(browser_profile=browser_profile)

    TASKSauce = """
             ONLY perform the following steps:
            1. Go to https://www.saucedemo.com/ and enter username as standard_user and password as secret_sauce and click on login button
            2. Verify text 'Swag Labs' is present by locating element with CSS Selector '.app_logo'.
            3. Verify text 'Sauce Labs Backpack' is present by locating element with CSS Selector '[data-test="inventory-item-name"]'.
            4. Verify text 'Add to cart' is present on the page.
             """

    Taskpwc = (
        'Important : I am UI Automation tester validating the tasks'
        # 'ONLY perform the following steps:'
        'Open website https://www.pwc.com/us/en.html'
        'locate the search icon or button on the page and click it to open the search bar.'
        'In the search bar, type Audit Services and submit the search.'
        'Verify text Audit Services is present on the page'
        'click on the link Audit Services'
        'verify url https://www.pwc.com/us/en/careers/why-pwc/what-we-do/what-we-do-audit-services.html'
        'Scroll down by 500 pixels. '
        'Verify text Search entry level opportunities is present on the page. '
        'click on Search entry level opportunities link and wait for new page to load. '
        'Switch to new tab. '
        'Enter keyword as Senior Associate on the new tab. '
        'Select Accounting from Field of Study. '
        'Select New York from Region. '
        'Select New York from City. '
        'Click on Search button. '
        'Scroll down by 500 pixels. '
        'Verify text New York is present on the page. '
    )

    Taskpwc1 = (
        'Important : I am UI Automation tester validating the tasks'
        'ONLY perform the following steps:'
        'Open website https://www.pwc.com/us/en.html'
        'Scroll down. '
        'click on US offices link. '
        'Scroll down by 500 pixels. '
        'click on Florida link. '
        'Verify text Miami is present on the page. '
        'Click on Tampa ESC Link. '
        'Verify text PricewaterhouseCoupers LLP 4040 West Boy Scout Boulevard - 10th Floor, Tampa, Florida 33607 is on the page. '
    )

    llmnew = AzureChatOpenAI(
        model='gpt-35-turbo',
        api_key=azure_openai_api_key,
        azure_endpoint=azure_openai_endpoint,  # Corrected to use azure_endpoint instead of openai_api_base
        api_version='2024-12-01-preview',  # Explicitly set the API version here
    )
    agent = Agent(
        task=Taskpwc,
        llm=llmnew,
        # enable_memory=True,
        # browser_session=browser_session,
        # message_context='verify the launch page',
        max_actions_per_step=1,
        max_failures=1,
        controller=controller,
        use_vision=False,
        validate_output=True,

    )
    report_folder= 'Report'
    if not os.path.exists(report_folder):
     os.makedirs(report_folder)
    timestamp= datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    history: AgentHistoryList = await agent.run(max_steps=21)
    overall_success = history.is_successful()
    print(f"\nOverall Agent Success Status: {overall_success}")
    history_json_string = history.model_dump_json(indent=2)
    # print(history_json_string)
    filename= f'output_json_{timestamp}.json'
    filepath = os.path.join(report_folder,filename)
    with open(filepath,'w',encoding='utf-8') as f:
        f.write(history_json_string)

    # with allure.step("Agent run with validation"):
    # allure.title("feature")
    # with allure.step("Agent execution started"):
    #     try:
    #
    # history: AgentHistoryList = await agent.run(max_steps=21)
    # history_json_string = history.model_dump_json(indent=2)
    # process_agent_history_json(history_json_string)
    #
    #             # Get and print the overall success status
    #             overall_success = history.is_successful()
    #             print(f"\nOverall Agent Success Status: {overall_success}")
    #             allure.attach(f"Overall Agent Success Status: {overall_success}", name="Agent Overall Status",
    #                           attachment_type=allure.attachment_type.TEXT)
    #
    #
    #             if overall_success is False:
    #                 pytest.fail("Agent reported task completion as unsuccessful.")
    #             elif overall_success is None:
    #                 # Handle cases where the agent hasn't explicitly marked done=True
    #                 pytest.fail("Agent did not explicitly mark the task as successful or unsuccessful.")
    #
    #             # Extract and attach next_goal for each step
    #             model_outputs = history.model_outputs()
    #             for i, output in enumerate(model_outputs):
    #                 next_goal = output.current_state.next_goal
    #                 if next_goal:
    #                     print(f"Step {i + 1} Next Goal: {next_goal}")
    #                     with allure.step(f"Agent's Goal for Step {i + 1}: {next_goal}"):
    #                         allure.attach(next_goal, name="Next Goal Description",
    #                                       attachment_type=allure.attachment_type.TEXT)
    #                 else:
    #                     print(f"Step {i + 1}: No model output or current state available for next goal.")
    #
    #
    #             final_goal_from_history = history.final_result()
    #             if final_goal_from_history:
    #                 print(f"\nFinal Agent's Reported Goal (from final_result method): {final_goal_from_history}")
    #                 allure.attach(f"Final Reported Goal: {final_goal_from_history}", name="Final Agent Goal",
    #                               attachment_type=allure.attachment_type.TEXT)
    #
    #
    #     except Exception as e:
    #         print(f"Error during agent run: {e}")
    #         allure.attach(str(e), name="Error Details during Agent Run", attachment_type=allure.attachment_type.TEXT)
    #         pytest.fail(f"Agent test failed due to an unhandled exception: {e}")
    #     finally:
    #         print("Browser lifecycle managed by 'async with Controller' block.")


# input('Press Enter to close...')

if __name__ == '__main__':
    asyncio.run(main())
