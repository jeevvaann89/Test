import asyncio
import base64
import io
import json
import os,sys
import shutil
import tempfile
import time
import traceback
import zipfile
import re
import inspect  # Added for openurl tool
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union, Callable
from browser_logger import get_browser_loggernew
import httpx
import nest_asyncio

import browser_logger
from dommutationchanger import subscribe, unsubscribe
from js_helper import get_js_with_element_finder

nest_asyncio.apply()
from dotenv import load_dotenv
import autogen
from textwrap import dedent
from string import Template
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import BrowserContext, BrowserType, ElementHandle
from playwright.async_api import Error as PlaywrightError  # for exception handling
from playwright.async_api import Page, Playwright
from playwright.async_api import async_playwright as playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError  # Added for openurl tool


azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
# --- Simplified / Placeholder imports from testzeus_hercules ---
# In a real scenario, you'd properly import or redefine these.
# For this example, I'm providing minimal implementations.

class DummyConfig:
    def get_global_conf(self):
        return self

    def should_record_video(self): return False

    def get_proof_path(self, test_id="default"):  # Added default test_id for logging paths
        proof_dir = f"./proofs/{test_id}"
        os.makedirs(proof_dir, exist_ok=True)
        return proof_dir

    def get_browser_type(self): return "chromium"

    def get_browser_channel(self): return None

    def get_browser_path(self): return None

    def get_browser_version(self): return None

    def should_run_headless(self): return True

    def get_cdp_config(self): return None

    def should_take_screenshots(self): return True

    def should_take_bounding_box_screenshots(self): return False  # Set this to True to see bounding box screenshots

    def should_capture_network(self): return False

    def should_enable_tracing(self): return False

    def should_enable_ublock_extension(self): return False  # Set to True to enable uBlock

    def get_run_device(self): return None

    def get_resolution(self): return "1280,720"

    def get_locale(self): return None

    def get_timezone(self): return None

    def get_geolocation(self): return None

    def get_color_scheme(self): return None

    def get_browser_cookies(self): return None

    def should_auto_accept_screen_sharing(self): return False

    def should_use_dynamic_ltm(self): return False

    def get_source_log_folder_path(self):  # Added for get_page_text
        log_dir = os.path.join(self.get_proof_path("default"), "logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir


# Global instance for the dummy config
_global_conf_instance = DummyConfig()
get_global_conf = _global_conf_instance.get_global_conf


class DummyLogger:
    def info(self, msg, *args): print(f"INFO: {msg}" % args)

    def warning(self, msg, *args): print(f"WARNING: {msg}" % args)

    def error(self, msg, *args): print(f"ERROR: {msg}" % args)

    def debug(self, msg, *args): print(f"DEBUG: {msg}" % args)


logger = DummyLogger()  # Using dummy logger


class DummyBrowserLogger:
    """A minimal placeholder for browser logger interactions."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)  # Ensure log directory exists

    async def log_browser_interaction(self, tool_name: str, action: str, interaction_type: str, success: bool,
                                      **kwargs):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "action": action,
            "interaction_type": interaction_type,
            "success": success,
            **kwargs
        }
        # print(f"[BROWSER_LOG]: {json.dumps(log_entry)}")
        # In a real scenario, you'd write this to a specific file or database
        # For now, just print to demonstrate.


def get_browser_logger(proof_path: str) -> DummyBrowserLogger:
    """Returns a dummy browser logger instance."""
    # This simplified version just takes proof_path
    return DummyBrowserLogger(proof_path)


# --- End of Simplified / Placeholder imports ---

# Ensures that playwright does not wait for font loading when taking screenshots.
os.environ["PW_TEST_SCREENSHOT_NO_FONTS_READY"] = "1"

MAX_WAIT_PAGE_LOAD_TIME = 0.6
WAIT_FOR_NETWORK_IDLE = 2
MIN_WAIT_PAGE_LOAD_TIME = 0.05

ALL_POSSIBLE_PERMISSIONS = [
    "geolocation",
    "notifications",
]

CHROMIUM_PERMISSIONS = [
    "clipboard-read",
    "clipboard-write",
]

BROWSER_CHANNELS = Literal[
    "chrome",
    "chrome-beta",
    "chrome-dev",
    "chrome-canary",
    "msedge",
    "msedge-beta",
    "msedge-dev",
    "msedge-canary",
    "firefox",
    "firefox-beta",
    "firefox-dev-edition",
    "firefox-nightly",
]


class PlaywrightBrowserManager:
    """
    Manages Playwright instances and browsers. Now supports stake_id-based singleton instances.
    """

    _instances: Dict[str, "PlaywrightBrowserManager"] = {}
    _default_instance: Optional["PlaywrightBrowserManager"] = None
    _homepage = "about:blank"

    def __new__(
            cls, *args, stake_id: Optional[str] = None, **kwargs
    ) -> "PlaywrightBrowserManager":
        if stake_id is None:
            if cls._default_instance is None:
                instance = super().__new__(cls)
                instance.__initialized = False
                cls._default_instance = instance
                cls._instances["0"] = instance
                logger.debug(
                    "Created default PlaywrightBrowserManager instance with stake_id '0'"
                )
            return cls._default_instance

        if stake_id not in cls._instances:
            instance = super().__new__(cls)
            instance.__initialized = False
            cls._instances[stake_id] = instance
            logger.debug(
                f"Created new PlaywrightBrowserManager instance for stake_id '{stake_id}'"
            )
            if cls._default_instance is None:
                cls._default_instance = instance
        return cls._instances[stake_id]

    @classmethod
    def get_instance(cls, stake_id: Optional[str] = None) -> "PlaywrightBrowserManager":
        """Get PlaywrightBrowserManager instance for given stake_id, or default instance if none provided."""
        if stake_id is None:
            if cls._default_instance is None:
                return cls()
            return cls._default_instance
        if stake_id not in cls._instances:
            return cls(stake_id=stake_id)
        return cls._instances[stake_id]

    @classmethod
    async def close_instance(cls, stake_id: Optional[str] = None) -> None:
        """Close and remove a specific PlaywrightBrowserManager instance."""
        target_id = stake_id if stake_id is not None else "0"
        if target_id in cls._instances:
            instance = cls._instances[target_id]
            await instance.stop_playwright()  # Await the stop_playwright call
            del cls._instances[target_id]
            if instance == cls._default_instance:
                cls._default_instance = None
                if cls._instances:
                    cls._default_instance = next(iter(cls._instances.values()))

    @classmethod
    async def close_all_instances(cls) -> None:
        """Close all PlaywrightBrowserManager instances."""
        # Use asyncio.gather to close instances concurrently
        await asyncio.gather(*[cls.close_instance(stake_id) for stake_id in list(cls._instances.keys())])

    def __init__(
            self,
            # ----------------------
            # FALLBACKS via CONF
            # ----------------------
            browser_type: Optional[str] = None,
            browser_channel: Optional[BROWSER_CHANNELS] = None,
            browser_path: Optional[str] = None,
            browser_version: Optional[str] = None,
            headless: Optional[bool] = None,
            gui_input_mode: bool = False,
            stake_id: Optional[str] = None,
            screenshots_dir: Optional[str] = None,
            take_screenshots: Optional[bool] = None,
            cdp_config: Optional[Dict] = None,
            cdp_reuse_tabs: Optional[bool] = False,
            cdp_navigate_on_connect: Optional[bool] = True,
            record_video: Optional[bool] = None,
            video_dir: Optional[str] = None,
            log_requests_responses: Optional[bool] = None,
            request_response_log_file: Optional[str] = None,
            # --- Emulation-specific args ---
            device_name: Optional[str] = None,
            viewport: Optional[Tuple[int, int]] = None,
            locale: Optional[str] = None,
            timezone: Optional[str] = None,
            geolocation: Optional[Dict[str, float]] = None,
            color_scheme: Optional[str] = None,
            allow_all_permissions: bool = True,
            log_console: Optional[bool] = None,
            console_log_file: Optional[str] = None,
            take_bounding_box_screenshots: Optional[bool] = None,
    ):
        if hasattr(self, "_PlaywrightBrowserManager__initialized") and self.__initialized:
            return

        self.__initialized = True

        self.allow_all_permissions = allow_all_permissions
        self.stake_id = stake_id or "0"

        self._record_video = record_video if record_video is not None else get_global_conf().should_record_video()
        self._latest_video_path: Optional[str] = None
        self._video_dir: Optional[str] = None

        proof_path = get_global_conf().get_proof_path(test_id=self.stake_id)

        self.browser_type = browser_type or get_global_conf().get_browser_type() or "chromium"
        self.browser_channel = browser_channel or get_global_conf().get_browser_channel()
        self.browser_path = browser_path or get_global_conf().get_browser_path()
        self.browser_version = browser_version or get_global_conf().get_browser_version()
        self.isheadless = headless if headless is not None else get_global_conf().should_run_headless()
        self.cdp_config = cdp_config or get_global_conf().get_cdp_config()

        config = get_global_conf()
        self.cdp_reuse_tabs = cdp_reuse_tabs if cdp_reuse_tabs is not None else getattr(config, "cdp_reuse_tabs", True)
        self.cdp_navigate_on_connect = cdp_navigate_on_connect if cdp_navigate_on_connect is not None else getattr(
            config, "cdp_navigate_on_connect", False)

        self._take_screenshots_enabled = take_screenshots if take_screenshots is not None else get_global_conf().should_take_screenshots()
        self._take_bounding_box_screenshots = take_bounding_box_screenshots if take_bounding_box_screenshots is not None else get_global_conf().should_take_bounding_box_screenshots()

        self._screenshots_dir = proof_path + "/screenshots"
        self._video_dir = proof_path + "/videos"
        self.request_response_log_file = proof_path + "/network_logs.json"
        self.console_log_file = proof_path + "/console_logs.json"
        self._enable_tracing = get_global_conf().should_enable_tracing()
        self._trace_dir = None
        if self._enable_tracing:
            proof_path = get_global_conf().get_proof_path(test_id=self.stake_id)
            self._trace_dir = os.path.join(proof_path, "traces")
            logger.info(f"Tracing enabled. Traces will be saved to: {self._trace_dir}")

        self.log_requests_responses = log_requests_responses if log_requests_responses is not None else get_global_conf().should_capture_network()
        self.request_response_logs: List[Dict] = []

        self._playwright: Optional[Playwright] = None
        self._browser_context: Optional[BrowserContext] = None
        self._browser_instance: Any = None  # New: to store the browser instance itself
        self._current_page: Optional[Page] = None  # New: Store current active page
        self.__async_initialize_done = False
        self._latest_screenshot_bytes: Optional[bytes] = None

        self._extension_cache_dir = os.path.join(
            ".", ".cache", "browser", self.browser_type, "extension"
        )
        self._extension_path: Optional[str] = None

        device_name = device_name or get_global_conf().get_run_device()
        self.device_name = device_name
        conf_res_str = get_global_conf().get_resolution() or "1280,720"
        cw, ch = conf_res_str.split(",")
        conf_viewport = (int(cw), int(ch))
        self.user_viewport = viewport or conf_viewport

        self.user_locale = locale or get_global_conf().get_locale()
        self.user_timezone = timezone or get_global_conf().get_timezone()
        self.user_geolocation = geolocation or get_global_conf().get_geolocation()
        self.user_color_scheme = color_scheme or get_global_conf().get_color_scheme()

        self.browser_cookies = get_global_conf().get_browser_cookies()

        if self.device_name and "iphone" in self.device_name.lower():
            logger.info(f"Detected iPhone in device_name='{self.device_name}'; forcing browser_type=webkit.")
            self.browser_type = "webkit"

        self.log_console = log_console if log_console is not None else True

        logger.debug(
            f"PlaywrightBrowserManager init - "
            f"browser_type={self.browser_type}, headless={self.isheadless}, "
            f"device={self.device_name}, viewport={self.user_viewport}, "
            f"locale={self.user_locale}, timezone={self.user_timezone}, "
            f"geolocation={self.user_geolocation}, color_scheme={self.user_color_scheme}"
        )

    async def async_initialize(self) -> None:
        if self.__async_initialize_done:
            return

        os.makedirs(self._screenshots_dir, exist_ok=True)
        os.makedirs(self._video_dir, exist_ok=True)
        if self._enable_tracing:
            os.makedirs(self._trace_dir, exist_ok=True)
        # Ensure log directories exist for internal logging
        os.makedirs(os.path.dirname(self.request_response_log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.console_log_file), exist_ok=True)
        os.makedirs(get_global_conf().get_source_log_folder_path(), exist_ok=True)  # For get_page_text's log

        await self.start_playwright()
        await self.ensure_browser_context()
        await self.go_to_homepage()
        self.__async_initialize_done = True

    async def get_current_page(self) -> Page:
        """Returns the currently active Playwright Page object."""
        if self._current_page is None:
            await self.ensure_browser_context()
            # If still None, create a new page in the existing context
            if not self._current_page:
                self._current_page = await self._browser_context.new_page()
                logger.info("Created a new page as no current page was set.")
        return self._current_page

    async def get_browser_context(self) -> BrowserContext:
        """Returns the current browser context, ensuring it's initialized."""
        if self._browser_context is None:
            await self.ensure_browser_context()  # This will create it if not present
        if self._browser_context is None:  # Should not happen if ensure_browser_context works
            raise RuntimeError("Browser context could not be initialized.")
        return self._browser_context

    async def switch_to_tab(self,tab_index: int) -> None:
        browser_context = await self.get_browser_context()
        pages = browser_context.pages
        # if len(pages) > tab_index:
        self._current_page  = pages[tab_index]
        print(await self._current_page.title())
            # await page.bring_to_front()
        # else:
        #     print(f"Tab {tab_index} does not exist")

    async def press_key_combinationnew(self,
            key_combination: Annotated[str, "key to press, e.g., Enter, PageDown etc"],
    ) -> str:
        logger.info(f"Executing press_key_combination with key combo: {key_combination}")
        # Create and use the PlaywrightManager

        page = await self.get_current_page()

        if page is None:  # type: ignore
            raise ValueError("No active page found. OpenURL command opens a new page.")

        # Split the key combination if it's a combination of keys
        keys = key_combination.split("+")

        dom_changes_detected = None

        def detect_dom_changes(changes: str):  # type: ignore
            nonlocal dom_changes_detected
            dom_changes_detected = changes  # type: ignore

        subscribe(detect_dom_changes)
        # If it's a combination, hold down the modifier keys
        for key in keys[:-1]:  # All keys except the last one are considered modifier keys
            await page.keyboard.down(key)

        # Press the last key in the combination
        await page.keyboard.press(keys[-1])

        # Release the modifier keys
        for key in keys[:-1]:
            await page.keyboard.up(key)
        await asyncio.sleep(300)  # sleep for 100ms to allow the mutation observer to detect changes
        unsubscribe(detect_dom_changes)

        await self.wait_for_load_state_if_enabled(page=page)

        await self.take_screenshots("press_key_combination_end", page)
        if dom_changes_detected:
            return f"Key {key_combination} executed successfully.\n As a consequence of this action, new elements have appeared in view:{dom_changes_detected}. This means that the action is not yet executed and needs further interaction. Get all_fields DOM to complete the interaction."

        return f"Key {key_combination} executed successfully"



    async def custom_fill_elementnew(self,page: Page, selector: str, text_to_enter: str) -> None:
        selector = f"{selector}"  # Ensures the selector is treated as a string
        try:
            js_code = """(inputParams) => {
                /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                const selector = inputParams.selector;
                let text_to_enter = inputParams.text_to_enter.trim();

                // Start by searching in the regular document (DOM)
                const element = findElementInShadowDOMAndIframes(document, selector);

                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }

                // Set the value for the element
                element.value = "";
                element.value = text_to_enter;
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));

                return `Value set for ${selector}`;
            }"""

            result = await page.evaluate(
                get_js_with_element_finder(js_code),
                {"selector": selector, "text_to_enter": text_to_enter},
            )
            logger.debug(f"custom_fill_element result: {result}")
        except Exception as e:

            traceback.print_exc()
            logger.error(f"Error in custom_fill_element, Selector: {selector}, Text: {text_to_enter}. Error: {str(e)}")
            raise

    async def do_entertextnew(self,page: Page, selector: str, text_to_enter: str, use_keyboard_fill: bool = True) -> dict[
        str, str]:
        try:
            logger.debug(f"Looking for selector {selector} to enter text: {text_to_enter}")

            elem = await self.find_element(selector, page, element_name="entertext")

            # Initialize selector logger with proof path
            selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())

            if not elem:
                # Log failed selector interaction
                await selector_logger.log_selector_interaction(
                    tool_name="entertext",
                    selector=selector,
                    action="input",
                    selector_type="css" if "md=" in selector else "custom",
                    success=False,
                    error_message=f"Error: Selector {selector} not found. Unable to continue.",
                )
                error = f"Error: Selector {selector} not found. Unable to continue."
                return {"summary_message": error, "detailed_message": error}
            else:
                # Get element properties to determine the best selection strategy
                tag_name = await elem.evaluate("el => el.tagName.toLowerCase()")
                element_role = await elem.evaluate("el => el.getAttribute('role') || ''")
                element_type = await elem.evaluate("el => el.type || ''")
                input_roles = ["combobox", "listbox", "dropdown", "spinner", "select"]
                input_types = [
                    "range",
                    "combobox",
                    "listbox",
                    "dropdown",
                    "spinner",
                    "select",
                    "option",
                ]
                logger.info(f"element_role: {element_role}, element_type: {element_type}")
                if element_role in input_roles or element_type in input_types:
                    properties = {
                        "tag_name": tag_name,
                        "element_role": element_role,
                        "element_type": element_type,
                        "element_outer_html": await self.get_element_outer_html(elem, page),
                        "alternative_selectors": await selector_logger.get_alternative_selectors(elem, page),
                        "element_attributes": await selector_logger.get_element_attributes(elem),
                        "selector_logger": selector_logger,
                    }
                    return await self.interact_with_element_select_type(page, elem, selector, text_to_enter, properties)

            logger.info(f"Found selector {selector} to enter text")
            element_outer_html = await self.get_element_outer_html(elem, page)

            # Initialize selector logger with proof path
            selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
            # Get alternative selectors and element attributes for logging
            alternative_selectors = await selector_logger.get_alternative_selectors(elem, page)
            element_attributes = await selector_logger.get_element_attributes(elem)

            if use_keyboard_fill:
                await elem.focus()
                await asyncio.sleep(0.01)
                await self.press_key_combinationnew("Control+A")
                await asyncio.sleep(0.01)
                await self.press_key_combinationnew("Delete")
                await asyncio.sleep(0.01)
                logger.debug(f"Focused element with selector {selector} to enter text")
                await page.keyboard.type(text_to_enter, delay=1)
            else:
                await self.custom_fill_element( selector, text_to_enter)

            await elem.focus()
            await self.wait_for_load_state_if_enabled(page=page)

            # Log successful selector interaction
            await selector_logger.log_selector_interaction(
                tool_name="entertext",
                selector=selector,
                action="input",
                selector_type="css" if "md=" in selector else "custom",
                alternative_selectors=alternative_selectors,
                element_attributes=element_attributes,
                success=True,
                additional_data={
                    "text_entered": text_to_enter,
                    "input_method": "keyboard" if use_keyboard_fill else "javascript",
                },
            )

            logger.info(f'Success. Text "{text_to_enter}" set successfully in the element with selector {selector}')
            success_msg = f'Success. Text "{text_to_enter}" set successfully in the element with selector {selector}'
            return {
                "summary_message": success_msg,
                "detailed_message": f"{success_msg} and outer HTML: {element_outer_html}.",
            }

        except Exception as e:

            traceback.print_exc()
            # Initialize selector logger with proof path
            selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
            # Log failed selector interaction
            await selector_logger.log_selector_interaction(
                tool_name="entertext",
                selector=selector,
                action="input",
                selector_type="css" if "md=" in selector else "custom",
                success=False,
                error_message=str(e),
            )

            traceback.print_exc()
            error = f"Error entering text in selector {selector}."
            return {"summary_message": error, "detailed_message": f"{error} Error: {e}"}

    async def entertextnew(self,
            entry: Annotated[
                tuple[str, str],
                "tuple containing 'selector' and 'value_to_fill' in ('selector', 'value_to_fill') format, selector is md attribute value of the dom element to interact, md is an ID and 'value_to_fill' is the value or text of the option to select",
            ],
    ) -> Annotated[str, "Text entry result"]:

        logger.info(f"Entering text: {entry}")

        selector: str = entry[0]
        text_to_enter: str = entry[1]

        # if "md=" not in selector:
        #     selector = f"[md='{selector}']"

        # Create and use the PlaywrightManager
        page = await self.get_current_page()
        # await page.route("**/*", block_ads)
        if page is None:  # type: ignore
            return "Error: No active page found. OpenURL command opens a new page."

        function_name = inspect.currentframe().f_code.co_name  # type: ignore

        await self.take_screenshots(f"{function_name}_start", page)

        # await browser_manager.highlight_element(selector)

        dom_changes_detected = None

        def detect_dom_changes(changes: str):  # type: ignore
            nonlocal dom_changes_detected
            dom_changes_detected = changes  # type: ignore

        subscribe(detect_dom_changes)

        await page.evaluate(
            get_js_with_element_finder(
                """
            (selector) => {
                /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                const element = findElementInShadowDOMAndIframes(document, selector);
                if (element) {
                    element.value = '';
                } else {
                    console.error('Element not found:', selector);
                }
            }
            """
            ),
            selector,
        )
        await page.wait_for_load_state('networkidle', timeout=60000)
        result = await self.do_entertextnew(page, selector, text_to_enter)
        await asyncio.sleep(
            300)  # sleep to allow the mutation observer to detect changes
        unsubscribe(detect_dom_changes)

        await self.wait_for_load_state_if_enabled(page=page)

        await self.take_screenshots(f"{function_name}_end", page)

        if dom_changes_detected:
            return f"{result['detailed_message']}.\n As a consequence of this action, new elements have appeared in view: {dom_changes_detected}. This means that the action of entering text {text_to_enter} is not yet executed and needs further interaction. Get all_fields DOM to complete the interaction."
        return result["detailed_message"]




    async def perform_javascript_click(
            self, page: Page, selector: str, type_of_click: str
    ) -> str:
        js_code = """(params) => {
            /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
            const selector = params[0];
            const type_of_click = params[1];

            let element = findElementInShadowDOMAndIframes(document, selector);
            if (!element) {
                console.log(`perform_javascript_click: Element with selector ${selector} not found`);
                return `perform_javascript_click: Element with selector ${selector} not found`;
            }

            if (element.tagName.toLowerCase() === "a") {
                element.target = "_self";
            }

            let ariaExpandedBeforeClick = element.getAttribute('aria-expanded');

            // Get the element's bounding rectangle for mouse events
            const rect = element.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            // Check if we're in Salesforce
            const isSalesforce = window.location.href.includes('lightning/') || 
                                window.location.href.includes('force.com') || 
                                document.querySelector('.slds-dropdown, lightning-base-combobox') !== null;

            // Check if element is SVG or SVG child
            const isSvgElement = element.tagName.toLowerCase() === 'svg' || 
                                element.ownerSVGElement !== null ||
                                element.namespaceURI === 'http://www.w3.org/2000/svg';

            // Common mouse move event
            const mouseMove = new MouseEvent('mousemove', {
                bubbles: true,
                cancelable: true,
                clientX: centerX,
                clientY: centerY,
                view: window
            });
            element.dispatchEvent(mouseMove);

            // Handle different click types
            switch(type_of_click) {
                case 'right_click':
                    const contextMenuEvent = new MouseEvent('contextmenu', {
                        bubbles: true,
                        cancelable: true,
                        clientX: centerX,
                        clientY: centerY,
                        button: 2,
                        view: window
                    });
                    element.dispatchEvent(contextMenuEvent);
                    break;

                case 'double_click':
                    const dblClickEvent = new MouseEvent('dblclick', {
                        bubbles: true,
                        cancelable: true,
                        clientX: centerX,
                        clientY: centerY,
                        button: 0,
                        view: window
                    });
                    element.dispatchEvent(dblClickEvent);
                    break;

                case 'middle_click':
                    const middleClickEvent = new MouseEvent('click', {
                        bubbles: true,
                        cancelable: true,
                        button: 1,
                        view: window
                    });
                    element.dispatchEvent(middleClickEvent);
                    break;

                default: // normal click
                    // For SVG elements or Salesforce, use event sequence approach
                    if (isSvgElement || isSalesforce) {
                        // SVG elements need full event sequence
                        // Create and dispatch mousedown event first
                        const mouseDown = new MouseEvent('mousedown', {
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            clientX: centerX,
                            clientY: centerY,
                            button: 0
                        });
                        element.dispatchEvent(mouseDown);

                        const mouseUpEvent = new MouseEvent('mouseup', {
                            bubbles: true,
                            cancelable: true,
                            clientX: centerX,
                            clientY: centerY,
                            button: 0,
                            view: window
                        });
                        element.dispatchEvent(mouseUpEvent);

                        const clickEvent = new MouseEvent('click', {
                            bubbles: true,
                            cancelable: true,
                            clientX: centerX,
                            clientY: centerY,
                            button: 0,
                            view: window
                        });
                        element.dispatchEvent(clickEvent);
                    } else {
                        // For regular HTML elements, try direct click first, fallback to event sequence
                        try {
                            // Try the native click method first
                            element.click();
                        } catch (error) {
                            console.log('Native click failed, using event sequence');
                            // Fallback to event sequence
                            const mouseDown = new MouseEvent('mousedown', {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                clientX: centerX,
                                clientY: centerY,
                                button: 0
                            });
                            element.dispatchEvent(mouseDown);

                            const mouseUpEvent = new MouseEvent('mouseup', {
                                bubbles: true,
                                cancelable: true,
                                clientX: centerX,
                                clientY: centerY,
                                button: 0,
                                view: window
                            });
                            element.dispatchEvent(mouseUpEvent);

                            const clickEvent = new MouseEvent('click', {
                                bubbles: true,
                                cancelable: true,
                                clientX: centerX,
                                clientY: centerY,
                                button: 0,
                                view: window
                            });
                            element.dispatchEvent(clickEvent);

                            // If it's a link and click wasn't prevented, handle navigation
                            if (element.tagName.toLowerCase() === 'a' && element.href) {
                                window.location.href = element.href;
                            }
                        }
                    }
                    break;
            }

            const ariaExpandedAfterClick = element.getAttribute('aria-expanded');
            if (ariaExpandedBeforeClick === 'false' && ariaExpandedAfterClick === 'true') {
                return "Executed " + type_of_click + " on element with selector: " + selector + 
                    ". Very important: As a consequence, a menu has appeared where you may need to make further selection. " +
                    "Very important: Get all_fields DOM to complete the action." + " The click is best effort, so verify the outcome.";
            }
            return "Executed " + type_of_click + " on element with selector: " + selector + " The click is best effort, so verify the outcome.";
        }"""

        try:
            logger.info(
                f"Executing JavaScript '{type_of_click}' on element with selector: {selector}"
            )
            result: str = await page.evaluate(
                get_js_with_element_finder(js_code), (selector, type_of_click)
            )
            logger.debug(
                f"Executed JavaScript '{type_of_click}' on element with selector: {selector}"
            )
            return result
        except Exception as e:

            traceback.print_exc()
            logger.error(
                f"Error executing JavaScript '{type_of_click}' on element with selector: {selector}. Error: {e}"
            )
            traceback.print_exc()
            return f"Error executing JavaScript '{type_of_click}' on element with selector: {selector}"

    async def is_element_present(
            self, selector: str, page: Optional[Page] = None
    ) -> bool:
        """Check if an element is present in DOM/Shadow DOM/iframes."""
        if page is None:
            page = await self.get_current_page()

        # Try regular DOM first
        element = await page.query_selector(selector)
        if element:
            return True

        # Check Shadow DOM and iframes
        js_code = """(selector) => {
            /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
            return findElementInShadowDOMAndIframes(document, selector) !== null;
        }"""

        return await page.evaluate_handle(get_js_with_element_finder(js_code), selector)

    async def ensure_browser_context(self) -> None:
        if self._browser_context is None:
            await self.create_browser_context()
        # Ensure we have a current page after context is created
        if self._current_page is None and self._browser_context:
            if self._browser_context.pages:
                self._current_page = self._browser_context.pages[0]
                logger.info(f"Using existing page in context: {self._current_page.url}")
            else:
                self._current_page = await self._browser_context.new_page()
                logger.info("Created new page in existing context.")

        if self._current_page:
            # Set up handlers for the current page
            await self.set_page_handlers(self._current_page)

    async def set_page_handlers(self, page: Page) -> None:
        """Sets up handlers for a given page, including request/response logging and console logging."""
        if self.log_requests_responses:
            page.on("request", lambda request: asyncio.create_task(self._log_request(request)))
            page.on("response", lambda response: asyncio.create_task(self._log_response(response)))
        if self.log_console:
            page.on("console", lambda msg: logger.info(f"[Console {msg.type.upper()}]: {msg.text}"))
            page.on("pageerror", lambda err: logger.error(f"[Page Error]: {err}"))

    async def start_playwright(self) -> None:
        if not self._playwright:
            self._playwright = await playwright().start()
            logger.info("Playwright started.")

    async def stop_playwright(self) -> None:
        if self._browser_context is not None:
            await self.close_browser_context()
        if self._browser_instance is not None:  # Close the browser instance if it was launched
            await self._browser_instance.close()
            self._browser_instance = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
            logger.info("Playwright stopped.")

    async def prepare_extension(self) -> None:
        if not get_global_conf().should_enable_ublock_extension():
            logger.info("uBlock extension is disabled in config. Skipping installation.")
            return

        if os.name == "nt":
            logger.info("Skipping extension preparation on Windows.")
            return

        extension_url = ""
        extension_file_name = ""
        extension_dir_name = ""

        if self.browser_type == "chromium":
            extension_url = "https://github.com/gorhill/uBlock/releases/download/1.61.0/uBlock0_1.61.0.chromium.zip"
            extension_file_name = "uBlock0_1.61.0.chromium.zip"
            extension_dir_name = "uBlock0_1.61.0.chromium"
        elif self.browser_type == "firefox":
            extension_url = "https://addons.mozilla.org/firefox/downloads/file/4359936/ublock_origin-1.60.0.xpi"
            extension_file_name = "uBlock0_1.60.0.firefox.xpi"
            extension_dir_name = "uBlock0_1.60.0.firefox"  # This will be the unzipped name for Chrome, for Firefox just file.
        else:
            logger.error(f"Unsupported browser type for extension: {self.browser_type}")
            return

        extension_dir = self._extension_cache_dir
        extension_file_path = os.path.join(extension_dir, extension_file_name)

        os.makedirs(extension_dir, exist_ok=True)

        if not os.path.exists(extension_file_path):
            logger.info(f"Downloading extension from {extension_url}")
            try:
                async with httpx.AsyncClient(follow_redirects=True) as client:
                    response = await client.get(extension_url, timeout=60.0)
                    response.raise_for_status()  # Raise an exception for 4xx/5xx responses
                    await asyncio.to_thread(lambda: open(extension_file_path, "wb").write(response.content))
                    logger.info(f"Extension downloaded and saved to {extension_file_path}")
            except Exception as e:
                logger.error(f"Failed to download extension: {e}")
                return

        if self.browser_type == "chromium":
            extension_unzip_dir = os.path.join(extension_dir, extension_dir_name)
            if not os.path.exists(extension_unzip_dir):
                logger.info(f"Unzipping extension to {extension_unzip_dir}")
                await asyncio.to_thread(
                    lambda: zipfile.ZipFile(extension_file_path, "r").extractall(extension_unzip_dir))
            self._extension_path = os.path.join(extension_unzip_dir,
                                                "uBlock0.chromium")  # Correct path within unzipped dir
        elif self.browser_type == "firefox":
            self._extension_path = extension_file_path  # For Firefox, it's the .xpi file itself

    async def _start_tracing(self, context_type: str = "browser") -> None:
        if not self._enable_tracing:
            return

        try:
            if self._browser_context:
                await self._browser_context.tracing.start(
                    screenshots=True,
                    snapshots=True,
                    sources=True,
                )
                logger.info(f"Tracing started for {context_type} context")
            else:
                logger.warning("Cannot start tracing: browser context is not initialized.")
        except Exception as e:
            logger.error(f"Failed to start tracing for {context_type} context: {e}")
            traceback.print_exc()

    async def _stop_tracing(self) -> None:
        if not self._enable_tracing or not self._browser_context:
            return

        try:
            if self._trace_dir:
                trace_file_path = os.path.join(self._trace_dir, f"trace-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip")
                await self._browser_context.tracing.stop(path=trace_file_path)
                logger.info(f"Tracing stopped. Trace saved to: {trace_file_path}")
            else:
                await self._browser_context.tracing.stop()  # Stop without saving if path not set
                logger.info("Tracing stopped (path not set).")
        except Exception as e:
            logger.error(f"Failed to stop tracing: {e}")
            traceback.print_exc()

    async def create_browser_context(self) -> None:
        user_dir: str = os.environ.get("BROWSER_STORAGE_DIR", "")
        if not user_dir:  # Use a temporary directory if not specified
            user_dir = tempfile.mkdtemp(prefix="playwright-user-data-")
            self._temp_user_data_dir = Path(user_dir)  # Store for cleanup

        disable_args = [
            "--disable-session-crashed-bubble", "--disable-notifications", "--no-sandbox",
            "--disable-blink-features=AutomationControlled", "--disable-infobars",
            "--disable-background-timer-throttling", "--disable-popup-blocking",
            "--disable-backgrounding-occluded-windows", "--disable-renderer-backgrounding",
            "--disable-window-activation", "--disable-focus-on-load", "--no-first-run",
            "--no-default-browser-check", "--window-position=0,0",
            "--disable-web-security", "--disable-features=IsolateOrigins,site-per-process",
        ]

        if not self.device_name:
            w, h = self.user_viewport
            disable_args.append(f"--window-size={w},{h}")

        if self.cdp_config:
            logger.info("Connecting over CDP with provided configuration.")
            endpoint_url = self.cdp_config.get("endpoint_url")
            if not endpoint_url:
                raise ValueError("CDP config must include 'endpoint_url'.")

            browser_type_launcher = getattr(self._playwright, self.browser_type)

            self._browser_instance = await browser_type_launcher.connect_over_cdp(endpoint_url, timeout=120000)

            context_options = self._build_emulation_context_options()
            if self._record_video:
                context_options["record_video_dir"] = self._video_dir
                logger.info("Recording video in CDP mode.")

            self._browser_context = None
            if self._browser_instance.contexts and self.cdp_reuse_tabs:
                logger.info("Reusing existing browser context.")
                self._browser_context = self._browser_instance.contexts[0]

            if not self._browser_context:
                logger.info("Creating new browser context.")
                self._browser_context = await self._browser_instance.new_context(**context_options)

            # Page selection logic - ensures a current page is set
            self._current_page = None
            if self._browser_context.pages:
                self._current_page = next(
                    (p for p in self._browser_context.pages if p.url not in ["about:blank", "chrome://newtab/"]),
                    self._browser_context.pages[0])
                logger.info(f"Reusing existing page with URL: {self._current_page.url}")
                try:
                    await self._current_page.bring_to_front()
                    logger.info("Brought reused page to front")
                except Exception as e:
                    logger.warning(f"Failed to bring page to front: {e}")

            if not self._current_page:  # If no usable page found or created
                logger.info(
                    "No usable existing pages found or creating new page as no existing pages found. Creating new page.")
                self._current_page = await self._browser_context.new_page()

            if self.cdp_navigate_on_connect:
                logger.info("Navigating to Google as specified in configuration.")
                await self._current_page.goto("https://www.google.com", timeout=120000)
            elif self._current_page.url in ["about:blank", "chrome://newtab/"]:
                logger.info("Setting minimal HTML content for empty tab")
                await self._current_page.set_content(
                    "<html><body><h1>Connected via PlaywrightBrowserManager</h1><p>Tab is ready for automation.</p></body></html>")

            await self._add_cookies_if_provided()

        else:  # Standard Playwright launch
            browser_type_launcher = getattr(self._playwright, self.browser_type)
            await self.prepare_extension()

            launch_options = {
                "headless": self.isheadless,
                "args": disable_args,
                "channel": self.browser_channel,
                "executable_path": self.browser_path,
                "timeout": 120000,  # Increased timeout for browser launch
            }

            if self.browser_type == "chromium" and self._extension_path:
                launch_options["args"].extend([
                    f"--disable-extensions-except={self._extension_path}",
                    f"--load-extension={self._extension_path}"
                ])
            elif self.browser_type == "firefox":
                # Firefox specific preferences (might need adjustment for extensions)
                firefox_prefs = {
                    "app.update.auto": False, "browser.shell.checkDefaultBrowser": False,
                    "media.navigator.permission.disabled": True, "permissions.default.screen": 1,
                    "media.getusermedia.window.enabled": True,
                }
                if get_global_conf().should_auto_accept_screen_sharing():
                    firefox_prefs.update({
                        "permissions.default.camera": 1, "permissions.default.microphone": 1,
                        "permissions.default.desktop-notification": 1, "media.navigator.streams.fake": True,
                        "media.getusermedia.screensharing.enabled": True, "media.getusermedia.browser.enabled": True,
                        "dom.disable_beforeunload": True, "media.autoplay.default": 0,
                        "media.autoplay.enabled": True, "privacy.webrtc.legacyGlobalIndicator": False,
                        "privacy.webrtc.hideGlobalIndicator": True, "permissions.default.desktop": 1,
                    })
                launch_options["firefox_user_prefs"] = firefox_prefs

            # Persistent context allows using existing user data dir
            context_options = self._build_emulation_context_options()
            if self._record_video:
                context_options["record_video_dir"] = self._video_dir
                context_options["record_video_size"] = {"width": self.user_viewport[0], "height": self.user_viewport[1]}

            self._browser_context = await browser_type_launcher.launch_persistent_context(
                user_data_dir=user_dir,
                **launch_options,
                **context_options  # Merge context options here
            )
            self._current_page = self._browser_context.pages[
                0] if self._browser_context.pages else await self._browser_context.new_page()
            await self._add_cookies_if_provided()

        await self._start_tracing()
        await self.set_page_handlers(await self.get_current_page())  # Set handlers for the active page

    def _build_emulation_context_options(self) -> Dict[str, Any]:
        context_options = {}
        if self.device_name:
            device = self._playwright.devices.get(self.device_name)
            if device:
                context_options.update(device)
            else:
                logger.warning(f"Device '{self.device_name}' not found in Playwright devices.")
        else:
            context_options["viewport"] = {"width": self.user_viewport[0], "height": self.user_viewport[1]}

        if self.user_locale: context_options["locale"] = self.user_locale
        if self.user_timezone: context_options["timezone_id"] = self.user_timezone
        if self.user_geolocation: context_options["geolocation"] = self.user_geolocation
        if self.user_color_scheme: context_options["color_scheme"] = self.user_color_scheme

        if self.allow_all_permissions:
            permissions = ALL_POSSIBLE_PERMISSIONS.copy()
            if self.browser_type == "chromium": permissions.extend(CHROMIUM_PERMISSIONS)
            context_options["permissions"] = permissions

        return context_options

    async def _add_cookies_if_provided(self) -> None:
        if self.browser_cookies and self._browser_context:
            try:
                await self._browser_context.add_cookies(self.browser_cookies)
                logger.info("Added browser cookies to context.")
            except Exception as e:
                logger.error(f"Failed to add browser cookies: {e}")

    async def close_browser_context(self) -> None:
        if self._browser_context is not None:
            await self._stop_tracing()
            # If a temporary user data directory was created, clean it up
            if hasattr(self, '_temp_user_data_dir') and self._temp_user_data_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, self._temp_user_data_dir)
                    logger.info(f"Cleaned up temporary user data directory: {self._temp_user_data_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary user data directory {self._temp_user_data_dir}: {e}")

            await self._browser_context.close()
            self._browser_context = None
            self._current_page = None
            logger.info("Browser context closed.")

    async def go_to_homepage(self) -> None:
        if self._current_page and self._homepage != "about:blank":
            try:
                await self._current_page.goto(self._homepage)
                logger.info(f"Navigated to homepage: {self._homepage}")
            except Exception as e:
                logger.error(f"Failed to navigate to homepage {self._homepage}: {e}")

    async def _log_request(self, request) -> None:
        try:
            self.request_response_logs.append({
                "type": "request",
                "timestamp": datetime.now().isoformat(),
                "url": request.url,
                "method": request.method,
                "headers": request.headers,
                "post_data": request.post_data,
            })
            # logger.debug(f"Logged request: {request.method} {request.url}")
        except Exception as e:
            logger.error(f"Error logging request: {e}")

    async def _log_response(self, response) -> None:
        try:
            status = response.status
            url = response.url
            headers = response.headers
            # Attempt to get response body (can be complex for large responses or redirects)
            body_preview = None
            try:
                body_bytes = await response.body()
                body_preview = body_bytes.decode('utf-8', errors='ignore')[:500] + (
                    '...' if len(body_bytes) > 500 else '')
            except Exception:
                pass  # Body might not be available or too large

            self.request_response_logs.append({
                "type": "response",
                "timestamp": datetime.now().isoformat(),
                "url": url,
                "status": status,
                "headers": headers,
                "body_preview": body_preview,
            })
            # logger.debug(f"Logged response: {status} {url}")
        except Exception as e:
            logger.error(f"Error logging response: {e}")

    async def reuse_or_create_tab(self, force_new_tab: bool = False) -> Page:
        """
        Reuses an existing page in the browser context or creates a new one.
        If force_new_tab is True, always creates a new page.
        """
        await self.ensure_browser_context()  # Ensure context is ready

        if force_new_tab:
            new_page = await self._browser_context.new_page()
            logger.info("Forced new tab creation.")
            self._current_page = new_page  # Set this as the active page
            await self.set_page_handlers(self._current_page)  # Set handlers for the new page
            return new_page
        elif self._current_page and not self._current_page.is_closed():
            logger.info(f"Reusing existing page: {self._current_page.url}")
            return self._current_page
        elif self._browser_context.pages:
            # If current_page is closed or None, but context has other pages
            page_to_reuse = self._browser_context.pages[0]
            logger.info(f"Reusing first available page in context: {page_to_reuse.url}")
            self._current_page = page_to_reuse  # Set this as the active page
            await self.set_page_handlers(self._current_page)  # Set handlers for the reused page
            return page_to_reuse
        else:
            # No existing pages, create a new one
            new_page = await self._browser_context.new_page()
            logger.info("No existing pages found, created a new page.")
            self._current_page = new_page  # Set this as the active page
            await self.set_page_handlers(self._current_page)  # Set handlers for the new page
            return new_page

    async def wait_for_page_and_frames_load(self) -> None:
        """
        Waits for the current page and its frames to reach a network idle state.
        This is a simplified version for this example.
        """
        page = await self.get_current_page()
        try:
            # Wait for 'networkidle' on the main frame
            await page.wait_for_load_state('networkidle', timeout=60000)
            logger.info(f"Main frame network idle for {page.url}")

            # Optionally, iterate through frames and wait for network idle if needed
            # For simplicity, we'll only wait for the main frame here.
            # In a real app, you might want a more robust solution for all frames.
            for frame in page.frames:
                if frame != page.main_frame:
                    try:
                        await frame.wait_for_load_state('networkidle', timeout=30000)
                        logger.debug(f"Frame network idle: {frame.url}")
                    except PlaywrightTimeoutError:
                        logger.warning(f"Frame network idle timeout for {frame.url}")
                    except Exception as e:
                        logger.error(f"Error waiting for frame network idle {frame.url}: {e}")

        except PlaywrightTimeoutError:
            logger.warning(f"Main page network idle timeout for {page.url}")
        except Exception as e:
            logger.error(f"Error waiting for page and frames load for {page.url}: {e}")

    async def take_screenshots(self, base_name: str, page: Optional[Page] = None) -> List[str]:
        """
        Takes screenshots. If _take_bounding_box_screenshots is enabled, it takes
        an additional screenshot with bounding boxes.
        Returns a list of paths to the saved screenshots.
        """
        if not self._take_screenshots_enabled:
            return []

        if page is None:
            page = await self.get_current_page()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        screenshot_paths = []

        try:
            # Main screenshot
            main_path = os.path.join(self._screenshots_dir, f"{base_name}-{timestamp}.png")
            await page.screenshot(path=main_path, full_page=True)
            screenshot_paths.append(main_path)
            logger.info(f"Main screenshot saved to {main_path}")

            if self._take_bounding_box_screenshots:
                # This part requires more advanced DOM analysis/drawing which is outside
                # the scope of simple Playwright methods and typically requires injecting
                # JavaScript to draw on a canvas or getting element positions and drawing
                # them on the screenshot using PIL.
                # For this example, I'll provide a placeholder or a very simplified version.
                # A proper implementation would involve:
                # 1. Getting all interactive elements (buttons, inputs, links).
                # 2. Getting their bounding boxes.
                # 3. Loading the screenshot into PIL.
                # 4. Drawing rectangles on the image using PIL.
                # 5. Saving the new image.

                # Simplified bounding box screenshot (just a copy for demonstration)
                # In a real scenario, this would be a more complex operation.
                bounding_box_path = os.path.join(self._screenshots_dir, f"{base_name}-{timestamp}-bbox.png")
                shutil.copy(main_path, bounding_box_path)  # Placeholder: just copy main screenshot
                screenshot_paths.append(bounding_box_path)
                logger.info(f"Bounding box screenshot (placeholder) saved to {bounding_box_path}")

        except PlaywrightError as e:
            logger.error(f"Failed to take screenshots for {base_name}: {e}")
            traceback.print_exc()
        except Exception as e:
            logger.error(f"An unexpected error occurred during screenshotting: {e}")
            traceback.print_exc()

        return screenshot_paths

    # --- Public methods for browser interaction (to be exposed as Autogen tools) ---

    async def navigate_to_url(self, url: str) -> Dict[str, Any]:
        """Navigates the browser to the specified URL."""
        page = await self.get_current_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            success = True
            message = f"Navigated to {url}"
            current_url = page.url
        except PlaywrightError as e:
            success = False
            message = f"Failed to navigate to {url}: {e}"
            current_url = page.url if page else "N/A"
            logger.error(message)
        return {"success": success, "message": message, "current_url": current_url}

    async def click_element(self, selector: Optional[str] = None, text: Optional[str] = None) -> Dict[str, Any]:
        """Clicks on an element identified by a CSS selector or visible text.
        Prefer text if both are provided."""
        page = await self.get_current_page()
        try:
            if text:
                locator = page.get_by_text(text, exact=True)
                action_desc = f"element with text '{text}'"
            elif selector:
                locator = page.locator(selector)
                action_desc = f"element with selector '{selector}'"
            else:
                return {"success": False, "error": "Either selector or text must be provided."}

            await locator.click(timeout=30000)
            success = True
            message = f"Clicked {action_desc}."
        except PlaywrightError as e:
            success = False
            message = f"Failed to click {action_desc}: {e}"
            logger.error(message)
        return {"success": success, "message": message}

    async def get_element_outer_html(self,element: ElementHandle, page: Page,
                                     element_tag_name: str | None = None) -> str:
        """
        Constructs the opening tag of an HTML element along with its attributes.

        Args:
            element (ElementHandle): The element to retrieve the opening tag for.
            page (Page): The page object associated with the element.
            element_tag_name (str, optional): The tag name of the element. Defaults to None. If not passed, it will be retrieved from the element.

        Returns:
            str: The opening tag of the HTML element, including a select set of attributes.
        """
        tag_name: str = element_tag_name if element_tag_name else await page.evaluate(
            "element => element.tagName.toLowerCase()", element)

        attributes_of_interest: list[str] = [
            "id",
            "name",
            "aria-label",
            "placeholder",
            "href",
            "src",
            "aria-autocomplete",
            "role",
            "type",
            "data-testid",
            "value",
            "selected",
            "aria-labelledby",
            "aria-describedby",
            "aria-haspopup",
            "title",
            "aria-controls",
        ]
        opening_tag: str = f"<{tag_name}"

        for attr in attributes_of_interest:
            value: str = await element.get_attribute(attr)  # type: ignore
            if value:
                opening_tag += f' {attr}="{value}"'
        opening_tag += ">"

        return opening_tag

    async def custom_fill_element(self, selector: str, text_to_enter: str) -> None:
        selector = f"{selector}"  # Ensures the selector is treated as a string
        try:
            js_code = """(inputParams) => {
                /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                const selector = inputParams.selector;
                let text_to_enter = inputParams.text_to_enter.trim();

                // Start by searching in the regular document (DOM)
                const element = findElementInShadowDOMAndIframes(document, selector);

                if (!element) {
                    throw new Error(`Element not found: ${selector}`);
                }

                // Set the value for the element
                element.value = "";
                element.value = text_to_enter;
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));

                return `Value set for ${selector}`;
            }"""
            page = await self.get_current_page()
            result = await page.evaluate(
                get_js_with_element_finder(js_code),
                {"selector": selector, "text_to_enter": text_to_enter},
            )
            logger.debug(f"custom_fill_element result: {result}")
        except Exception as e:

            traceback.print_exc()
            logger.error(
                f"Error in custom_fill_element, Selector: {selector}, Text: {text_to_enter}. Error: {str(e)}")
            raise

    async def find_element(
            self,
            selector: str,
            page: Optional[Page] = None,
            element_name: Optional[str] = None,
    ) -> Optional[ElementHandle]:
        """Find element in DOM/Shadow DOM/iframes and return ElementHandle.
        Also captures a screenshot with the element's bounding box and metadata overlay if enabled.

        Args:
            selector: The selector to find the element
            page: Optional page instance to search in
            element_name: Optional friendly name for the element (used in screenshot naming)
        """
        if page is None:
            page = await self.get_current_page()

        # Try regular DOM first
        element = await page.query_selector(selector)
        # if element:
        #     if self._take_bounding_box_screenshots:
        #         await self._capture_element_with_bbox(
        #             element, page, selector, element_name
        #         )
        #     return element

        # Check Shadow DOM and iframes
        js_code = """(selector) => {
            /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
            return findElementInShadowDOMAndIframes(document, selector);
        }"""

        element = await page.evaluate_handle(
            get_js_with_element_finder(js_code), selector
        )
        if element:
            element_handle = element.as_element()
            if element_handle:
                if self._take_bounding_box_screenshots:
                    await self._capture_element_with_bbox(
                        element_handle, page, selector, element_name
                    )
            return element_handle

        return None

    async def select_option(self,
            entry: Annotated[
                tuple[str, str],
                (
                        "tuple containing 'selector' and 'value_to_fill' in "
                        "('selector', 'value_to_fill') format. Selector is the md attribute value "
                        "of the DOM element and value_to_fill is the option text/value to select."
                ),
            ],
    ) -> Annotated[str, "Explanation of the outcome of dropdown/spinner selection."]:
        # add_event(EventType.INTERACTION, EventData(detail="SelectOption"))
        logger.info(f"Selecting option: {entry}")
        selector: str = entry[0]
        option_value: str = entry[1]

        # If the selector doesn't contain md=, wrap it accordingly.
        # if "md=" not in selector:
        #     selector = f"[md='{selector}']"


        page = await self.get_current_page()

        if page is None:
            return "Error: No active page found. OpenURL command opens a new page."

        function_name = inspect.currentframe().f_code.co_name  # type: ignore
        await self.take_screenshots(f"{function_name}_start", page)
        # await self.highlight_element(selector)

        dom_changes_detected = None

        def detect_dom_changes(changes: str) -> None:
            nonlocal dom_changes_detected
            dom_changes_detected = changes

        subscribe(detect_dom_changes)
        await page.wait_for_load_state('networkidle', timeout=60000)
        result = await self.do_select_option(page, selector, option_value)
        # Wait for page to stabilize after selection
        await self.wait_for_load_state_if_enabled(page=page)
        unsubscribe(detect_dom_changes)

        await self.wait_for_load_state_if_enabled(page=page)
        await self.take_screenshots(f"{function_name}_end", page)

        # Simply return the detailed message
        return result["detailed_message"]

    async def do_select_option(self,page:Page, selector: str, option_value: str) -> dict[str, str]:
        """
        Simplified approach to select an option in a dropdown using the element's properties.
        Uses find_element to get the element and then determines the best strategy based on
        the element's role, type, and tag name.
        """
        try:
            # page = await self.get_current_page()
            logger.info(f"Looking for selector {selector} to select option: {option_value}")

            # Part 1: Find the element and get its properties
            element, properties = await self.find_element_select_type(page, selector)
            if not element:
                error = f"Error: Selector '{selector}' not found. Unable to continue."
                return {"summary_message": error, "detailed_message": error}
            logger.info(f"Looking for selector {element} to select option: {properties}")
            # Part 2: Interact with the element to select the option
            return await self.interact_with_element_select_type(page, element, selector, option_value, properties)

        except Exception as e:

            traceback.print_exc()
            selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
            await selector_logger.log_selector_interaction(
                tool_name="select_option",
                selector=selector,
                action="select",
                selector_type="css" if "md=" in selector else "custom",
                success=False,
                error_message=str(e),
            )
            traceback.print_exc()
            error = f"Error selecting option in selector '{selector}'."
            return {"summary_message": error, "detailed_message": f"{error} Error: {e}"}

    async def find_element_select_type(self, page: Page,selector: str) -> tuple[Optional[ElementHandle], dict]:
        """
        Internal function to find the element and gather its properties.
        Returns the element and a dictionary of its properties.
        """
        element = await self.find_element(selector, page, element_name="select_option")

        if not element:
            selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
            await selector_logger.log_selector_interaction(
                tool_name="select_option",
                selector=selector,
                action="select",
                selector_type="css" if "md=" in selector else "custom",
                success=False,
                error_message=f"Error: Selector '{selector}' not found. Unable to continue.",
            )
            return None, {}

        logger.info(f"Found selector '{selector}' to select option")
        selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
        alternative_selectors = await selector_logger.get_alternative_selectors(element, page)
        element_attributes = await selector_logger.get_element_attributes(element)

        # Get element properties to determine the best selection strategy
        tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
        element_role = await element.evaluate("el => el.getAttribute('role') || ''")
        element_type = await element.evaluate("el => el.type || ''")
        element_outer_html = await self.get_element_outer_html(element,page)

        properties = {
            "tag_name": tag_name,
            "element_role": element_role,
            "element_type": element_type,
            "element_outer_html": element_outer_html,
            "alternative_selectors": alternative_selectors,
            "element_attributes": element_attributes,
            "selector_logger": selector_logger,
        }

        return element, properties

    async def get_latest_screenshot_stream(self) -> Optional[BytesIO]:
        if not self._latest_screenshot_bytes:
            # Take a new screenshot if none exists
            page = await self.get_current_page()
            await self.take_screenshots("latest_screenshot", page)

        if self._latest_screenshot_bytes:
            return BytesIO(self._latest_screenshot_bytes)
        else:
            logger.warning("Failed to take screenshot.")
            return None


    async def _capture_element_with_bbox(
            self,
            element: ElementHandle,
            page: Page,
            selector: str,
            element_name: Optional[str] = None,
    ) -> None:
        """Capture screenshot with bounding box and metadata overlay."""
        try:
            # Get element's bounding box
            bbox = await element.bounding_box()
            if not bbox:
                return

            # Get element's accessibility info
            accessibility_info = await element.evaluate(
                """element => {
                return {
                    ariaLabel: element.getAttribute('aria-label'),
                    role: element.getAttribute('role'),
                    name: element.getAttribute('name'),
                    title: element.getAttribute('title')
                }
            }"""
            )

            # Use the first non-empty value from accessibility info
            element_identifier = next(
                (
                    val
                    for val in [
                    accessibility_info.get("ariaLabel"),
                    accessibility_info.get("role"),
                    accessibility_info.get("name"),
                    accessibility_info.get("title"),
                ]
                    if val
                ),
                "element",  # default if no accessibility info found
            )

            # Construct screenshot name
            screenshot_name = f"{element_identifier}_{element_name or selector}_bbox_{int(datetime.now().timestamp())}"

            # Take screenshot using existing method
            await self.take_screenshots(
                name=screenshot_name,
                page=page,
                full_page=True,
                include_timestamp=False,
            )

            # Get the latest screenshot using get_latest_screenshot_stream
            screenshot_stream = await self.get_latest_screenshot_stream()
            if not screenshot_stream:
                logger.error("Failed to get screenshot for bounding box overlay")
                return

            image = Image.open(screenshot_stream)
            draw = ImageDraw.Draw(image)

            # Draw bounding box
            draw.line(
                [
                    (bbox["x"], bbox["y"]),
                    (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                ],
                width=4,
            )

            # Prepare metadata text
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            url = page.url
            test_name = self.stake_id or "default"
            element_info = f"Element: {element_identifier} by {element_name}"

            # Create metadata text block with word wrapping
            metadata = [
                f"Timestamp: {current_time}",
                f"URL: {url}",
                f"Test: {test_name}",
                element_info,
            ]

            # Calculate text position and size
            try:
                font = ImageFont.truetype("Arial", 14)
            except Exception as e:
                logger.error(f"Failed to load font: {e}")
                font = ImageFont.load_default()

            # Increase text padding by 10%
            text_padding = 11  # Original 10 + 10%
            line_height = 22  # Original 20 + 10%

            # Calculate text dimensions with word wrapping
            max_width = min(
                image.width * 0.4, 400
            )  # Reduced from 500px to 400px for better wrapping
            wrapped_lines = []

            for text in metadata:
                if text.startswith("URL: "):
                    # Special handling for URLs - break into chunks
                    url_prefix = "URL: "
                    url_text = text[len(url_prefix):]
                    current_line = url_prefix

                    # Break URL into segments of reasonable length
                    segment_length = (
                        40  # Adjust this value to control URL segment length
                    )
                    start = 0
                    while start < len(url_text):
                        end = start + segment_length
                        if end < len(url_text):
                            # Look for a good breaking point
                            break_chars = ["/", "?", "&", "-", "_", "."]
                            for char in break_chars:
                                pos = url_text[start: end + 10].find(char)
                                if pos != -1:
                                    end = start + pos + 1
                                    break
                        else:
                            end = len(url_text)

                        segment = url_text[start:end]
                        if start == 0:
                            wrapped_lines.append(url_prefix + segment)
                        else:
                            wrapped_lines.append(" " * len(url_prefix) + segment)
                        start = end
                else:
                    # Normal text wrapping for non-URL lines
                    words = text.split()
                    current_line = words[0]
                    for word in words[1:]:
                        test_line = current_line + " " + word
                        test_width = draw.textlength(test_line, font=font)
                        if test_width <= max_width:
                            current_line = test_line
                        else:
                            wrapped_lines.append(current_line)
                            current_line = word
                    wrapped_lines.append(current_line)

            # Calculate background dimensions with some extra padding
            bg_width = max_width + (text_padding * 2)
            bg_height = (line_height * len(wrapped_lines)) + (text_padding * 2)

            # Draw background rectangle for metadata
            bg_x = image.width - bg_width - text_padding
            bg_y = text_padding

            # Draw semi-transparent background
            bg_color = (0, 0, 0, 128)
            bg_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_layer)
            bg_draw.rectangle(
                [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color
            )

            # Composite the background onto the main image
            image = Image.alpha_composite(image.convert("RGBA"), bg_layer)
            draw = ImageDraw.Draw(image)

            # Draw wrapped text
            current_y = bg_y + text_padding
            for line in wrapped_lines:
                draw.text(
                    (bg_x + text_padding, current_y), line, fill="white", font=font
                )
                current_y += line_height

            # Save the modified screenshot
            screenshot_path = os.path.join(
                self.get_screenshots_dir(), f"{screenshot_name}.png"
            )

            # Convert back to RGB before saving as PNG
            image = image.convert("RGB")
            image.save(screenshot_path, "PNG")

            logger.debug(f"Saved bounding box screenshot: {screenshot_path}")

            # Get browser logger instance
            browser_logger = get_browser_loggernew(self.get_screenshots_dir())

            # Get element attributes and alternative selectors for logging
            element_attributes = await browser_logger.get_element_attributes(element)
            alternative_selectors = await browser_logger.get_alternative_selectors(
                element, page
            )

            # Log the screenshot interaction
            await browser_logger.log_browser_interaction(
                tool_name="find_element",
                action="capture_bounding_box_screenshot",
                interaction_type="screenshot",
                selector=selector,
                selector_type="custom",
                alternative_selectors=alternative_selectors,
                element_attributes=element_attributes,
                success=True,
                additional_data={
                    "screenshot_name": f"{screenshot_name}.png",
                    "screenshot_path": screenshot_path,
                    "element_identifier": element_identifier,
                    "bounding_box": bbox,
                    "url": url,
                    "timestamp": current_time,
                    "test_name": test_name,
                    "element_name": element_name,
                },
            )

        except Exception as e:
            logger.error(f"Failed to capture element with bounding box: {e}")
            traceback.print_exc()

            # Log failure in browser logger
            browser_logger = get_browser_loggernew(self.get_screenshots_dir())
            await browser_logger.log_browser_interaction(
                tool_name="find_element",
                action="capture_bounding_box_screenshot",
                interaction_type="screenshot",
                selector=selector,
                success=False,
                error_message=str(e),
                additional_data={
                    "element_name": element_name,
                },
            )
    # async def interact_with_element_select_type(self,
    #         page:Page,
    #         element: ElementHandle,
    #         selector: str,
    #         option_value: str,
    #         properties: dict,
    # ) -> dict[str, str]:
    #     """
    #     Internal function to interact with the element to select the option.
    #     """
    #     # page = await self.get_current_page()
    #     tag_name = properties["tag_name"]
    #     element_role = properties["element_role"]
    #     element_type = properties["element_type"]
    #     element_outer_html = properties["element_outer_html"]
    #     alternative_selectors = properties["alternative_selectors"]
    #     element_attributes = properties["element_attributes"]
    #     selector_logger = properties["selector_logger"]
    #     logger.info(f"tag name details {tag_name}")
    #     # Strategy 1: Standard HTML select element
    #     if tag_name == "select":
    #         await element.select_option(value=option_value)
    #         await page.wait_for_load_state("domcontentloaded", timeout=1000)
    #         await selector_logger.log_selector_interaction(
    #             tool_name="select_option",
    #             selector=selector,
    #             action="select",
    #             selector_type="css" if "md=" in selector else "custom",
    #             alternative_selectors=alternative_selectors,
    #             element_attributes=element_attributes,
    #             success=True,
    #             additional_data={
    #                 "element_type": "select",
    #                 "selected_value": option_value,
    #             },
    #         )
    #         success_msg = f"Success. Option '{option_value}' selected in the dropdown with selector '{selector}'"
    #         return {
    #             "summary_message": success_msg,
    #             "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
    #         }
    #
    #     # Strategy 2: Input elements (text, number, etc.)
    #     elif tag_name in ["input", "button"]:
    #         input_roles = ["combobox", "listbox", "dropdown", "spinner", "select"]
    #         input_types = [
    #             "number",
    #             "range",
    #             "combobox",
    #             "listbox",
    #             "dropdown",
    #             "spinner",
    #             "select",
    #             "option",
    #         ]
    #
    #         if element_type in input_types or element_role in input_roles:
    #             await element.click()
    #             try:
    #                 await element.fill(option_value)
    #             except Exception as e:
    #
    #                 # traceback.print_exc()
    #                 logger.warning(f"Error filling input: {str(e)}, trying type instead")
    #                 await element.type(option_value)
    #
    #             if "lwc" in str(element) and "placeholder" in str(element):
    #                 logger.info("Crazy LWC element detected")
    #                 await asyncio.sleep(0.5)
    #                 # await press_key_combination("ArrowDown+Enter")
    #             else:
    #                 await element.press("Enter")
    #
    #             await page.wait_for_load_state("domcontentloaded", timeout=1000)
    #
    #             await selector_logger.log_selector_interaction(
    #                 tool_name="select_option",
    #                 selector=selector,
    #                 action="input",
    #                 selector_type="css" if "md=" in selector else "custom",
    #                 alternative_selectors=alternative_selectors,
    #                 element_attributes=element_attributes,
    #                 success=True,
    #                 additional_data={
    #                     "element_type": "input",
    #                     "input_type": element_type,
    #                     "value": option_value,
    #                 },
    #             )
    #             success_msg = f"Success. Value '{option_value}' set in the input with selector '{selector}'"
    #             return {
    #                 "summary_message": success_msg,
    #                 "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
    #             }
    #
    #     # Strategy 3: Generic click and select approach for all other elements
    #     # Click to open the dropdown
    #
    #     logger.info(f"taking worst case scenario of selecting option for {element}, properties: {properties}")
    #     await element.click()
    #     await page.wait_for_timeout(300)  # Short wait for dropdown to appear
    #
    #     # Try to find and click the option by text content
    #     try:
    #         # Use a simple text-based selector that works in most cases
    #         option_selector = f"text={option_value}"
    #         await page.click(option_selector, timeout=2000)
    #         await page.wait_for_load_state("domcontentloaded", timeout=1000)
    #
    #         await selector_logger.log_selector_interaction(
    #             tool_name="select_option",
    #             selector=selector,
    #             action="click_by_text",
    #             selector_type="css" if "md=" in selector else "custom",
    #             alternative_selectors=alternative_selectors,
    #             element_attributes=element_attributes,
    #             success=True,
    #             additional_data={
    #                 "element_type": tag_name,
    #                 "selected_value": option_value,
    #                 "method": "text_content",
    #             },
    #         )
    #         success_msg = f"Success. Option '{option_value}' selected by text content"
    #         return {
    #             "summary_message": success_msg,
    #             "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
    #         }
    #     except Exception as e:
    #
    #         traceback.print_exc()
    #         logger.debug(f"Text-based selection failed: {str(e)}")
    #
    #         # If all attempts fail, report failure
    #         await selector_logger.log_selector_interaction(
    #             tool_name="select_option",
    #             selector=selector,
    #             action="select",
    #             selector_type="css" if "md=" in selector else "custom",
    #             alternative_selectors=alternative_selectors,
    #             element_attributes=element_attributes,
    #             success=False,
    #             error_message=f"Could not find option '{option_value}' in the dropdown with any selection method.",
    #         )
    #         error = f"Error: Option '{option_value}' not found in the element with selector '{selector}'. Try clicking the element first and then select the option."
    #         return {"summary_message": error, "detailed_message": error}

    async def interact_with_element_select_type(
            self,
            page: Page,
            element: ElementHandle,
            selector: str,  # This selector is primarily for logging purposes, not for re-locating
            option_value: str,
            properties: dict,
    ) -> dict[str, str]:
        """
        Internal function to interact with the element to select the option.
        """
        tag_name = properties.get("tag_name")
        element_role = properties.get("element_role")
        element_type = properties.get("element_type")
        element_outer_html = properties.get(
            "outer_html")  # Changed from element_outer_html to outer_html to match _get_element_and_properties
        alternative_selectors = properties.get("alternative_selectors", [])
        element_attributes = properties.get("element_attributes", {})
        selector_logger = properties.get("selector_logger")

        if not selector_logger:
            logger.error("selector_logger not provided in properties. Cannot log interaction.")
            selector_logger = get_browser_logger(get_global_conf().get_proof_path())  # Fallback

        # Strategy 1: Standard HTML select element
        if tag_name == "select":
            try:
                await element.select_option(value=option_value)
                await page.wait_for_load_state("domcontentloaded", timeout=1000)
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if selector.startswith("#") or selector.startswith(
                        ".") or "[" in selector else "custom",  # More robust selector type check
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=True,
                    additional_data={
                        "element_type": "select",
                        "selected_value": option_value,
                    },
                )
                success_msg = f"Success. Option '{option_value}' selected in the dropdown with selector '{selector}'"
                return {
                    "summary_message": success_msg,
                    "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
                }
            except Exception as e:
                error_msg = f"Failed to select option '{option_value}' in standard select element '{selector}': {e}"
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if selector.startswith("#") or selector.startswith(
                        ".") or "[" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=False,
                    error_message=error_msg,
                )
                logger.error(error_msg)
                return {"summary_message": error_msg, "detailed_message": error_msg}


        # Strategy 2: Input elements (text, number, etc.) that behave like dropdowns
        elif tag_name in ["input", "button"]:
            input_roles = ["combobox", "listbox", "dropdown", "spinner", "select"]
            input_types = [
                "number",
                "range",
                # These types are usually for non-standard dropdowns
                "combobox",
                "listbox",
                "dropdown",
                "spinner",
                "select",
                "option",
            ]

            if element_type in input_types or element_role in input_roles:
                try:
                    await element.click()
                    try:
                        await element.fill(option_value)
                    except Exception as e:
                        logger.warning(f"Error filling input: {str(e)}, trying type instead")
                        await element.type(option_value)

                    # This LWC specific logic should probably be more generic or configurable
                    if "lwc" in str(element_outer_html) and "placeholder" in str(element_outer_html):
                        logger.info("Potential LWC element detected, attempting ArrowDown+Enter.")
                        await asyncio.sleep(0.5)  # Give time for dropdown to appear
                        await self.press_key_combinationnew("ArrowDown+Enter")
                    else:
                        await element.press("Enter")

                    await page.wait_for_load_state("domcontentloaded", timeout=1000)

                    await selector_logger.log_selector_interaction(
                        tool_name="select_option",
                        selector=selector,
                        action="input",
                        selector_type="css" if selector.startswith("#") or selector.startswith(
                            ".") or "[" in selector else "custom",
                        alternative_selectors=alternative_selectors,
                        element_attributes=element_attributes,
                        success=True,
                        additional_data={
                            "element_type": "input",
                            "input_type": element_type,
                            "value": option_value,
                        },
                    )
                    success_msg = f"Success. Value '{option_value}' set in the input with selector '{selector}'"
                    return {
                        "summary_message": success_msg,
                        "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
                    }
                except Exception as e:
                    error_msg = f"Failed to interact with input/button element '{selector}' (type: {element_type}, role: {element_role}): {e}"
                    await selector_logger.log_selector_interaction(
                        tool_name="select_option",
                        selector=selector,
                        action="input",
                        selector_type="css" if selector.startswith("#") or selector.startswith(
                            ".") or "[" in selector else "custom",
                        alternative_selectors=alternative_selectors,
                        element_attributes=element_attributes,
                        success=False,
                        error_message=error_msg,
                    )
                    logger.error(error_msg)
                    return {"summary_message": error_msg, "detailed_message": error_msg}

        # Strategy 3: Generic click and select approach for all other elements (e.g., custom dropdowns built with divs/spans)
        logger.info(f"Attempting generic click-and-select for element: {selector}, properties: {properties}")
        try:
            await element.click()
            await page.wait_for_timeout(300)  # Short wait for dropdown to appear

            # Try to find and click the option by text content
            option_locator = page.get_by_text(option_value, exact=True)
            await option_locator.click(timeout=5000)  # Increased timeout for clicking option
            await page.wait_for_load_state("domcontentloaded", timeout=1000)

            await selector_logger.log_selector_interaction(
                tool_name="select_option",
                selector=selector,
                action="click_by_text",
                selector_type="css" if selector.startswith("#") or selector.startswith(
                    ".") or "[" in selector else "custom",  # More robust selector type check
                alternative_selectors=alternative_selectors,
                element_attributes=element_attributes,
                success=True,
                additional_data={
                    "element_type": tag_name,
                    "selected_value": option_value,
                    "method": "text_content_click",
                },
            )
            success_msg = f"Success. Option '{option_value}' selected by text content for element '{selector}'"
            return {
                "summary_message": success_msg,
                "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
            }
        except Exception as e:
            error_msg = f"Could not find or click option '{option_value}' in the element '{selector}' using generic click-and-select method: {e}"
            logger.error(error_msg)
            traceback.print_exc()

            # If all attempts fail, report failure
            await selector_logger.log_selector_interaction(
                tool_name="select_option",
                selector=selector,
                action="select",
                selector_type="css" if selector.startswith("#") or selector.startswith(
                    ".") or "[" in selector else "custom",
                alternative_selectors=alternative_selectors,
                element_attributes=element_attributes,
                success=False,
                error_message=error_msg,
            )
            return {"summary_message": error_msg, "detailed_message": error_msg}

    async def fill_field(self, selector: str, value: str) -> Dict[str, Any]:
        """Fills a text input field identified by a CSS selector with a given value."""
        page = await self.get_current_page()
        try:
            await page.fill(selector, value, timeout=30000)
            success = True
            message = f"Filled field '{selector}' with value: '{value}'."
        except PlaywrightError as e:
            success = False
            message = f"Failed to fill field '{selector}': {e}"
            logger.error(message)
        return {"success": success, "message": message}

    async def get_page_content(self) -> Dict[str, Any]:
        """Retrieves the full HTML content of the current page."""
        page = await self.get_current_page()
        try:
            content = await page.content()
            return {"success": True, "content": content}
        except PlaywrightError as e:
            logger.error(f"Failed to get page content: {e}")
            return {"success": False, "error": str(e)}

    async def get_text_from_xpath(self, xpath: str):
        page = await self.get_current_page()
        element = await page.query_selector(f"xpath={xpath}")
        if element:
            return {"success": True, "text": await element.inner_text()}
        else:
            return {"success": False, "text": "Element not found"}

    async def get_text_from_selector(self, selector: str) -> Dict[str, Any]:
        """Retrieves the text content of an element identified by a CSS selector."""
        page = await self.get_current_page()
        try:
            text_content = await page.locator(selector).text_content(timeout=1000)
            if text_content is None:
                return {"success": False, "error": f"Element with selector '{selector}' found but has no text content."}
            return {"success": True, "text": text_content.strip()}
        except PlaywrightError as e:
            logger.error(f"Failed to get text from selector '{selector}': {e}")
            return {"success": False, "error": f"Failed to get text from selector '{selector}': {e}"}

    async def take_screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Takes a screenshot of the current page and saves it to a specified path.
        Returns the path to the screenshot."""
        if not self._take_screenshots_enabled:
            return {"success": False, "error": "Screenshots are disabled in configuration."}

        page = await self.get_current_page()
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(self._screenshots_dir, f"screenshot-{timestamp}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            await page.screenshot(path=path)
            logger.info(f"Screenshot saved to {path}")
            return {"success": True, "path": path}
        except PlaywrightError as e:
            logger.error(f"Failed to take screenshot: {e}")
            return {"success": False, "error": str(e)}

    async def scroll_page(self, direction: Literal["up", "down"] = "down", pixels: int = 500) -> Dict[str, Any]:
        """Scrolls the page up or down by a specified number of pixels."""
        page = await self.get_current_page()
        try:
            if direction == "down":
                await page.evaluate(f"window.scrollBy(0, {pixels})")
            elif direction == "up":
                await page.evaluate(f"window.scrollBy(0, -{pixels})")
            else:
                return {"success": False, "error": "Direction must be 'up' or 'down'."}
            return {"success": True, "message": f"Scrolled {direction} by {pixels} pixels."}
        except PlaywrightError as e:
            logger.error(f"Failed to scroll page: {e}")
            return {"success": False, "error": str(e)}

    async def selectoptionnew(self, selector: Optional[str] = None, text: Optional[str] = None,
                            option_text: Optional[str] = None, option_value: Optional[str] = None) -> Dict[str, Any]:
        """Selects an option in a dropdown element identified by a CSS selector or visible text.
        Prefer text if both are provided for the element.
        Specify either option_text or option_value for the option to select."""

        page = await self.get_current_page()
        try:
            if text:
                locator = page.get_by_text(text, exact=True)
                action_desc = f"element with text '{text}'"
            elif selector:
                locator = page.locator(selector)
                action_desc = f"element with selector '{selector}'"
            else:
                return {"success": False, "error": "Either selector or text must be provided."}

            if option_text:
                option_desc = f"option with text '{option_text}'"
                await locator.select_option(label=option_text, timeout=30000)
            elif option_value:
                option_desc = f"option with value '{option_value}'"
                await locator.select_option(value=option_value, timeout=30000)
            else:
                return {"success": False, "error": "Either option_text or option_value must be provided."}

            success = True
            message = f"Selected {option_desc} in {action_desc}."
        except PlaywrightError as e:
            success = False
            message = f"Failed to select {option_desc} in {action_desc}: {e}"
            logger.error(message)
        return {"success": success, "message": message}

    async def get_element_attribute(self, selector: str, attribute_name: str) -> Dict[str, Any]:
        """Retrieves the value of a specific attribute from an element identified by a CSS selector."""
        page = await self.get_current_page()
        try:
            element = page.locator(selector)
            if await element.count() == 0:
                return {"success": False, "error": f"Element with selector '{selector}' not found."}

            attribute_value = await element.first.get_attribute(attribute_name)
            if attribute_value is None:
                return {"success": False,
                        "error": f"Attribute '{attribute_name}' not found or has no value for selector '{selector}'."}

            return {"success": True, "attribute_value": attribute_value}
        except PlaywrightError as e:
            logger.error(f"Failed to get attribute '{attribute_name}' from selector '{selector}': {e}")
            return {"success": False, "error": str(e)}

    async def wait_for_selector(self, selector: str, timeout: int = 30000,
                                state: Literal["attached", "detached", "visible", "hidden"] = "visible") -> Dict[
        str, Any]:
        """Waits for an element matching the selector to satisfy a state condition."""
        page = await self.get_current_page()
        try:
            await page.wait_for_selector(selector, timeout=timeout, state=state)
            return {"success": True, "message": f"Element '{selector}' is now '{state}'."}
        except PlaywrightError as e:
            logger.error(f"Failed to wait for selector '{selector}': {e}")
            return {"success": False, "error": str(e)}

    async def wait_for_load_state_if_enabled(self, page: Page, state: Literal[
        "load", "domcontentloaded", "networkidle"] = "load") -> None:
        """
        Waits for the page to reach a specified load state.
        """
        try:
            await page.wait_for_load_state(state, timeout=10000)  # Wait for 'load' event
            logger.info(f"Page load state '{state}' reached for {page.url}")
        except PlaywrightTimeoutError as e:
            logger.warning(f"Timeout waiting for page load state '{state}': {e}")
        except PlaywrightError as e:
            logger.warning(f"Failed to wait for page load state: {e}")

    async def get_page_text_content(self) -> Dict[str, Any]:
        """
        Retrieves, filters, and cleans visible text content from the current page.
        It saves the raw filtered text to a log file and returns the cleaned text.
        """
        logger.info(f"Executing get_page_text_content")
        start_time = time.time()

        page = await self.get_current_page()

        # Wait for page to be in a stable state
        await self.wait_for_load_state_if_enabled(page=page, state="networkidle")  # Use networkidle for more stability
        await asyncio.sleep(1)  # Give a moment for page to render after load state

        if page is None:
            return {"success": False, "error": "No active page found to retrieve text."}

        try:
            logger.debug("Fetching DOM for text_only via get_filtered_text_content")
            # Execute the JavaScript to get filtered content
            text_content = await get_filtered_text_content(page)
            cleaned_text = clean_text(text_content)

            # Save to file using the global config path
            log_folder_path = get_global_conf().get_source_log_folder_path()
            os.makedirs(log_folder_path, exist_ok=True)  # Ensure directory exists
            log_file_path = os.path.join(log_folder_path, "text_only_dom.txt")

            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logger.info(f"Page text saved to {log_file_path}")

            extracted_data = cleaned_text
        except Exception as e:
            logger.error(f"Error extracting filtered text content: {e}")
            extracted_data = f"Error extracting page text: {e}"
            return {"success": False, "error": extracted_data}

        elapsed_time = time.time() - start_time
        logger.info(f"Get DOM Command executed in {elapsed_time:.2f} seconds")

        final_response = extracted_data if extracted_data else "Its Empty, try something else"
        return {"success": True, "text_content": final_response}


# --- Helper functions for get_page_text and openurl ---
def clean_text(text_content: str) -> str:
    """
    Cleans and normalizes text content by stripping lines, collapsing whitespace,
    and removing duplicate lines.
    """
    lines = text_content.splitlines()
    seen_lines = set()
    cleaned_lines = []

    for line in lines:
        cleaned_line = " ".join(line.strip().replace("\t", " ").split())
        cleaned_line = cleaned_line.strip()
        if cleaned_line and cleaned_line not in seen_lines:
            cleaned_lines.append(cleaned_line)
            seen_lines.add(cleaned_line)  # Add to seen set
    return "\n".join(cleaned_lines)


async def get_filtered_text_content(page: Page) -> str:
    text_content = await page.evaluate(
        """
        () => {
        const selectorsToFilter = ['#hercules-overlay'];
        const originalStyles = [];

        /**
        * Hide elements by setting their visibility to "hidden".
        */
        function hideElements(root, selector) {
            if (!root) return;
            const elements = root.querySelectorAll(selector);
            elements.forEach(element => {
            originalStyles.push({
                element,
                originalStyle: element.style.visibility
            });
            element.style.visibility = 'hidden';
            });
        }

        /**
        * Recursively hide elements in shadow DOM.
        */
        function processElementsInShadowDOM(root, selector) {
            if (!root) return;
            hideElements(root, selector);

            const allNodes = root.querySelectorAll('*');
            allNodes.forEach(node => {
            if (node.shadowRoot) {
                processElementsInShadowDOM(node.shadowRoot, selector);
            }
            });
        }

        /**
        * Recursively hide elements in iframes.
        */
        function processElementsInIframes(root, selector) {
            if (!root) return;
            const iframes = root.querySelectorAll('iframe');
            iframes.forEach(iframe => {
            try {
                const iframeDoc = iframe.contentDocument;
                if (iframeDoc) {
                processElementsInShadowDOM(iframeDoc, selector);
                processElementsInIframes(iframeDoc, selector);
                }
            } catch (err) {
                console.log('Error accessing iframe content:', err);
            }
            });
        }

        /**
        * Create a TreeWalker that:
        * - Visits text nodes and element nodes
        * - Skips (<script> and <style>) elements entirely
        */
        function createSkippingTreeWalker(root) {
            return document.createTreeWalker(
            root,
            NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
            {
                acceptNode(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const tag = node.tagName.toLowerCase();
                    if (tag === 'script' || tag === 'style') {
                    return NodeFilter.FILTER_REJECT; // skip <script> / <style>
                    }
                }
                return NodeFilter.FILTER_ACCEPT;
                }
            }
            );
        }

        /**
        * Gets text by walking the DOM, but skipping <script> and <style>.
        * Also recursively checks for shadow roots and iframes.
        */
        function getTextSkippingScriptsStyles(root) {
            if (!root) return '';

            let textContent = '';
            const walker = createSkippingTreeWalker(root);

            while (walker.nextNode()) {
            const node = walker.currentNode;

            // If it's a text node, accumulate text
            if (node.nodeType === Node.TEXT_NODE) {
                textContent += node.nodeValue;
            }
            // If it has a shadowRoot, recurse
            else if (node.shadowRoot) {
                textContent += getTextSkippingScriptsStyles(node.shadowRoot);
            }
            }

            return textContent;
        }

        /**
        * Recursively gather text from iframes, also skipping <script> & <style>.
        */
        function getTextFromIframes(root) {
            if (!root) return '';
            let iframeText = '';

            const iframes = root.querySelectorAll('iframe');
            iframes.forEach(iframe => {
            try {
                const iframeDoc = iframe.contentDocument;
                if (iframeDoc) {
                // Grab text from iframe body, docElement, plus nested iframes
                iframeText += getTextSkippingScriptsStyles(iframeDoc.body);
                iframeText += getTextSkippingScriptsStyles(iframeDoc.documentElement);
                iframeText += getTextFromIframes(iframeDoc);
                }
            } catch (err) {
                console.log('Error accessing iframe content:', err);
            }
            });

            return iframeText;
        }

        /**
        * Collect alt texts for images (this part can remain simpler, as alt text
        * won't appear in <script> or <style> tags anyway).
        */
        function getAltTextsFromShadowDOM(root) {
            if (!root) return [];
            let altTexts = Array.from(root.querySelectorAll('img')).map(img => img.alt);

            const allNodes = root.querySelectorAll('*');
            allNodes.forEach(node => {
            if (node.shadowRoot) {
                altTexts = altTexts.concat(getAltTextsFromShadowDOM(node.shadowRoot));
            }
            });
            return altTexts;
        }

        function getAltTextsFromIframes(root) {
            if (!root) return [];
            let iframeAltTexts = [];

            const iframes = root.querySelectorAll('iframe');
            iframes.forEach(iframe => {
            try {
                const iframeDoc = iframe.contentDocument;
                if (iframeDoc) {
                iframeAltTexts = iframeAltTexts.concat(getAltTextsFromShadowDOM(iframeDoc));
                iframeAltTexts = iframeAltTexts.concat(getAltTextsFromIframes(iframeDoc));
                }
            } catch (err) {
                console.log('Error accessing iframe content:', err);
            }
            });

            return iframeAltTexts;
        }

        // 1) Hide overlays
        selectorsToFilter.forEach(selector => {
            processElementsInShadowDOM(document, selector);
            processElementsInIframes(document, selector);
        });

        // 2) Collect text from the main document
        let textContent = getTextSkippingScriptsStyles(document.body);
        textContent += getTextSkippingScriptsStyles(document.documentElement);

        // 3) Collect text from iframes
        textContent += getTextFromIframes(document);

        // 4) Collect alt texts
        let altTexts = getAltTextsFromShadowDOM(document);
        altTexts = altTexts.concat(getAltTextsFromIframes(document));
        const altTextsString = 'Other Alt Texts in the page: ' + altTexts.join(' ');

        // 5) Restore hidden overlays
        originalStyles.forEach(entry => {
            entry.element.style.visibility = entry.originalStyle;
        });

        // 6) Return final text
        textContent = textContent + ' ' + altTextsString;

        // Optional: sanitize whitespace, if needed
        // const sanitizeString = (input) => input.replace(\\s+/g, ' ');
        // textContent = sanitizeString(textContent);

        return textContent;
        }
    """
    )
    return clean_text(text_content)

# async def get_filtered_text_content(page: Page) -> str:
#     """
#     Executes JavaScript on the page to extract filtered text content,
#     excluding scripts, styles, and specified overlay elements.
#     It also collects alt texts from images.
#     """
#     text_content = await page.evaluate(
#         """
#         () => {
#         const selectorsToFilter = ['#hercules-overlay']; // Example of elements to hide/filter
#         const originalStyles = [];
#
#         function hideElements(root, selector) {
#             if (!root) return;
#             const elements = root.querySelectorAll(selector);
#             elements.forEach(element => {
#             originalStyles.push({
#                 element,
#                 originalStyle: element.style.visibility
#             });
#             element.style.visibility = 'hidden';
#             });
#         }
#
#         function processElementsInShadowDOM(root, selector) {
#             if (!root) return;
#             hideElements(root, selector);
#
#             const allNodes = root.querySelectorAll('*');
#             allNodes.forEach(node => {
#             if (node.shadowRoot) {
#                 processElementsInShadowDOM(node.shadowRoot, selector);
#             }
#             });
#         }
#
#         function processElementsInIframes(root, selector) {
#             if (!root) return;
#             const iframes = root.querySelectorAll('iframe');
#             iframes.forEach(iframe => {
#             try {
#                 const iframeDoc = iframe.contentDocument;
#                 if (iframeDoc) {
#                 processElementsInShadowDOM(iframeDoc, selector);
#                 processElementsInIframes(iframeDoc, selector);
#                 }
#             } catch (err) {
#                 console.log('Error accessing iframe content:', err);
#             }
#             });
#         }
#
#         function createSkippingTreeWalker(root) {
#             return document.createTreeWalker(
#             root,
#             NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
#             {
#                 acceptNode(node) {
#                 if (node.nodeType === Node.ELEMENT_NODE) {
#                     const tag = node.tagName.toLowerCase();
#                     if (tag === 'script' || tag === 'style') {
#                     return NodeFilter.FILTER_REJECT; // skip <script> / <style>
#                     }
#                 }
#                 return NodeFilter.FILTER_ACCEPT;
#                 }
#             }
#             );
#         }
#
#         function getTextSkippingScriptsStyles(root) {
#             if (!root) return '';
#
#             let textContent = '';
#             const walker = createSkippingTreeWalker(root);
#
#             while (walker.nextNode()) {
#             const node = walker.currentNode;
#
#             if (node.nodeType === Node.TEXT_NODE) {
#                 textContent += node.nodeValue;
#             }
#             else if (node.shadowRoot) {
#                 textContent += getTextSkippingScriptsStyles(node.shadowRoot);
#             }
#             }
#             return textContent;
#         }
#
#         function getTextFromIframes(root) {
#             if (!root) return '';
#             let iframeText = '';
#
#             const iframes = root.querySelectorAll('iframe');
#             iframes.forEach(iframe => {
#             try {
#                 const iframeDoc = iframe.contentDocument;
#                 if (iframeDoc) {
#                 iframeText += getTextSkippingScriptsStyles(iframeDoc.body);
#                 iframeText += getTextSkippingScriptsStyles(iframeDoc.documentElement);
#                 iframeText += getTextFromIframes(iframeDoc);
#                 }
#             } catch (err) {
#                 console.log('Error accessing iframe content:', err);
#             }
#             });
#             return iframeText;
#         }
#
#         function getAltTextsFromShadowDOM(root) {
#             if (!root) return [];
#             let altTexts = Array.from(root.querySelectorAll('img')).map(img => img.alt);
#
#             const allNodes = root.querySelectorAll('*');
#             allNodes.forEach(node => {
#             if (node.shadowRoot) {
#                 altTexts = altTexts.concat(getAltTextsFromShadowDOM(node.shadowRoot));
#             }
#             });
#             return altTexts.filter(alt => alt && alt.trim() !== ''); // Filter out empty alt texts
#         }
#
#         function getAltTextsFromIframes(root) {
#             if (!root) return [];
#             let iframeAltTexts = [];
#
#             const iframes = root.querySelectorAll('iframe');
#             iframes.forEach(iframe => {
#             try {
#                 const iframeDoc = iframe.contentDocument;
#                 if (iframeDoc) {
#                 iframeAltTexts = iframeAltTexts.concat(getAltTextsFromShadowDOM(iframeDoc));
#                 iframeAltTexts = iframeAltTexts.concat(getAltTextsFromIframes(iframeDoc));
#                 }
#             } catch (err) {
#                 console.log('Error accessing iframe content:', err);
#             }
#             });
#             return iframeAltTexts.filter(alt => alt && alt.trim() !== ''); // Filter out empty alt texts
#         }
#
#         // 1) Hide overlays
#         selectorsToFilter.forEach(selector => {
#             processElementsInShadowDOM(document, selector);
#             processElementsInIframes(document, selector);
#         });
#
#         // 2) Collect text from the main document and its documentElement
#         let mainTextContent = getTextSkippingScriptsStyles(document.body);
#         mainTextContent += getTextSkippingScriptsStyles(document.documentElement);
#
#         // 3) Collect text from iframes
#         let iframeTextContent = getTextFromIframes(document);
#
#         // 4) Collect alt texts
#         let altTexts = getAltTextsFromShadowDOM(document);
#         altTexts = altTexts.concat(getAltTextsFromIframes(document));
#
#         # Combine all text
#         let combinedText = mainTextContent + '\n' + iframeTextContent;
#         if (altTexts.length > 0) {
#             combinedText += '\nOther Alt Texts in the page: ' + altTexts.join(' ');
#         }
#
#         // 5) Restore hidden overlays
#         originalStyles.forEach(entry => {
#             entry.element.style.visibility = entry.originalStyle;
#         });
#
#         return combinedText;
#         }
#     """
#     )
#     return text_content


def ensure_protocol(url: str) -> str:
    """
    Ensures that a URL has a protocol (http:// or https://). If it doesn't have one,
    https:// is added by default.

    Special browser URLs like about:blank, chrome://, etc. are preserved as-is.
    """
    special_schemes = [
        "about:", "chrome:", "edge:", "brave:", "firefox:", "safari:",
        "data:", "file:", "view-source:",
    ]

    if any(url.startswith(scheme) for scheme in special_schemes):
        logger.debug(f"URL uses a special browser scheme, preserving as-is: {url}")
        return url

    if not url.startswith(("http://", "https://")):
        url = "https://" + url
        logger.info(f"Added 'https://' protocol to URL because it was missing. New URL is: {url}")
    return url


# Autogen imports


load_dotenv()  # Load environment variables for OpenAI API

azure_openai_api_key1 = "CtRZOhhcn6D6yotmbHPxRURAmMtbMtVouhLGT9S2KdttSABh7IFLJQQJ99BEACHYHv6XJ3w3AAAAACOGWbzD"
azure_openai_endpoint1 = "https://12240-mb0slb6w-eastus2.cognitiveservices.azure.com/"

config_list_openai = [
    {
        "model": "gpt-35-turbo",  # or "gpt-4", "gpt-4-vision-preview"
        "api_key": azure_openai_api_key1,
        "base_url": azure_openai_endpoint1,
        "api_type": "azure",
        "api_version": "2024-12-01-preview",  # Use a recent API version compatible with function calling
    }
]


# --- Autogen Tool Decorator ---
def tool(agent_names: list, name: str, description: str):
    """
    A decorator to attach metadata to functions for Autogen tool registration.
    """

    def decorator(func):
        func.__tool_metadata__ = {
            "agent_names": agent_names,
            "name": name,
            "description": description
        }
        return func

    return decorator


# --- BrowserNavAgent Class ---
class BrowserNavAgent:
    """
    Autogen agent for browser navigation and interaction using Playwright.
    """
    agent_name: str = "browser_nav_agent"
    prompt = dedent("""# Web Navigation Agent

You are a smart and specialized web navigation agent tasked with executing precise webpage interactions and retrieving information accurately using the provided browser tools.

## Capabilities
- Navigate to URLs (`openurl`).
- Interact with web elements (buttons, inputs, dropdowns, etc.) using selectors or visible text.
- Retrieve visible and relevant text content from the current page (`get_page_text`).
- Extract and summarize text content from web pages or specific elements.
- Take screenshots for visual verification.
- Scroll pages to bring elements into view.
- Get attributes from elements.
- Wait for elements to appear or change state.

## Core Rules

### TASK BOUNDARIES
1. Execute ONLY web navigation tasks using the available tools; never attempt other types of tasks.
2. Stay on the current page unless explicitly directed to navigate elsewhere.
3. Focus ONLY on elements within the ACTIVE interaction plane of the UI.

### ELEMENT IDENTIFICATION

**Element Identification:** When a step requires interacting with or verifying an element, use a robust strategy to locate it. Do NOT rely solely on XPath from a previous step or a single type of selector.
        *   **Prioritize Selectors:** Attempt to locate elements using the most reliable selectors first:
            *   ID (if available and unique)
            *   Name attribute
            *   CSS Selectors (preferable for readability and robustness over brittle XPaths)
            *   Link Text or Partial Link Text (for anchor tags)
            *   Button Text or Value
            *   XPath (use as a fallback, prioritize reliable, less brittle XPaths if possible, e.g., relative paths or paths using attributes).
        *   **Contextual Identification:** Use the text content, role, or other attributes mentioned or implied in the Gherkin step description to help identify the *correct* element among similar ones. For example, if the step is "When the user clicks the 'Submit' button", look for a button element containing the text "Submit".
        *   **Locate BEFORE Action/Verification:** Always attempt to locate the element successfully *before* performing an action (click, type) or verification on it.
        *   **Capture Detailed Element Information:** After locating an element but before interacting with it, use the "Get detailed element information" action to capture comprehensive details about the element including its ID, tag name, class name, XPaths (absolute and relative), and CSS selectors. This information is crucial for generating robust test scripts.

4. ALWAYS use precise CSS selectors or visible text for element identification.
5. If you cannot identify an element or need general page content for analysis, use the `get_page_text` tool to understand the page structure better, or ask for clarification.


### EXECUTION PROCESS
6. ALWAYS analyze the page (or its content using `get_page_text`) and the task BEFORE taking any action.
7. Plan and execute the optimal sequence of function/tool calls.
8. Execute ONE function/tool at a time.
9. Fully verify each result before proceeding to the next action.
10. PERSIST until the task is FULLY COMPLETED.
11. If a step fails, try to understand why from the error message and re-attempt (if possible) or report the failure.

### INTERACTION SPECIFICS
12. Use the most appropriate tool for the interaction (e.g., `click_element` for buttons/links, `fill_field` for input boxes).
13. Always provide all required parameters for tool calls. If a parameter is optional but useful, consider providing it.

## Response Format
### Success:
previous_step: [previous step assigned summary]
current_output: [Detailed description of actions performed and outcomes, often based on tool output]
##FLAG::SAVE_IN_MEM##
##TERMINATE TASK##

### Information Request Response:
previous_step: [previous step assigned summary]
current_output: [Detailed answer with specific information extracted from the page]
Data: [Relevant extracted information]
##TERMINATE TASK##

### Error or Uncertainty:
previous_step: [previous step assigned summary]
current_output: [Precise description of the issue encountered or why the task cannot be completed]
##TERMINATE TASK##

Available Test Data: $basic_test_information
""")

    def __init__(self
                 , model_config_list: list
                 , llm_config_params: dict[str, Any]
                 , nav_executor: autogen.UserProxyAgent
                 , playwright_manager: PlaywrightBrowserManager
                 , system_prompt: str | None = None
                 , agent_name: str | None = None
                 , agent_prompt: str | None = None
                 ):

        self.nav_executor = nav_executor
        self.playwright_manager = playwright_manager  # Store the Playwright manager instance

        self._agent_name = agent_name if agent_name is not None else self.agent_name

        effective_system_message = agent_prompt or self.prompt
        if system_prompt:
            if isinstance(system_prompt, list):
                effective_system_message = "\n".join(system_prompt)
            else:
                effective_system_message = system_prompt
            logger.info(f"Using custom system prompt for {self._agent_name}: {effective_system_message}")

        effective_system_message += f"\nCurrent timestamp is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        effective_system_message = Template(effective_system_message).safe_substitute(basic_test_information="")

        logger.info(f"Nav agent {self._agent_name} using model: {model_config_list[0]['model']}")

        self.agent = autogen.ConversableAgent(
            name=self._agent_name,
            system_message=effective_system_message,
            llm_config={
                "config_list": config_list_openai,  # Use the global config_list_openai
                **llm_config_params,
            },
            human_input_mode="NEVER",
            is_termination_msg=lambda x: x.get("content") and "TERMINATE" in x.get("content", "").upper(),
            max_consecutive_auto_reply=40,
        )

        self.register_tools()

    def get_agent_instance(self) -> autogen.ConversableAgent:
        return self.agent

    def register_tools(self) -> None:
        """
        Registers Playwright-based browser tools with the Autogen agents.
        """
        logger.info(f"Registering Playwright tools for {self._agent_name}...")

        # --- NEW TOOL: openurl ---
        @tool(
            agent_names=[self._agent_name],
            description="""Opens specified URL in browser. Returns new page URL or error message.
            Parameters:
            - url: URL to navigate to. Value must include the protocol (http:// or https://).
            - timeout: Additional wait time in seconds after initial load (default: 3).
            - force_new_tab: Force opening in a new tab instead of reusing existing ones (default: False).
            """,
            name="openurl",
        )
        async def openurl(
                url: Annotated[str, "URL to navigate to. Value must include the protocol (http:// or https://)."],
                timeout: Annotated[int, "Additional wait time in seconds after initial load."] = 3,
                force_new_tab: Annotated[bool, "Force opening in a new tab instead of reusing existing ones."] = False,
        ) -> Annotated[str, "Returns the result of this request in text form"]:
            logger.info(f"Executing openurl: {url} (force_new_tab={force_new_tab})")
            browser_logger = get_browser_logger(get_global_conf().get_proof_path())

            try:
                page = await self.playwright_manager.reuse_or_create_tab(force_new_tab=force_new_tab)
                logger.info(
                    f"{'Using new tab' if force_new_tab else 'Reusing existing tab when possible'} for navigation to {url}")

                special_browser_urls = [
                    "about:blank", "about:newtab", "chrome://newtab/", "edge://newtab/",
                ]

                if url.strip().lower() in special_browser_urls:
                    special_url = url.strip().lower()
                    logger.info(f"Navigating to special browser URL: {special_url}")
                    try:
                        await page.evaluate(f"window.location.href = '{special_url}'")
                        await page.wait_for_load_state("domcontentloaded")
                    except Exception as e:
                        logger.warning(
                            f"JavaScript navigation to {special_url} failed: {e}. Trying alternative method.")
                        try:
                            if special_url.startswith("about:"):
                                await page.goto(special_url, timeout=timeout * 1000)
                            else:
                                await page.set_content("<html><body></body></html>")
                                await page.evaluate(f"window.location.href = '{special_url}'")
                        except Exception as fallback_err:
                            logger.error(f"All navigation methods to {special_url} failed: {fallback_err}")

                    title = await page.title()
                    await browser_logger.log_browser_interaction(
                        tool_name="openurl", action="navigate", interaction_type="navigation", success=True,
                        additional_data={"url": special_url, "title": title, "from_cache": False, "status": "loaded",
                                         "force_new_tab": force_new_tab, },
                    )
                    return f"Navigated to {special_url}, Title: {title}"

                url = ensure_protocol(url)
                if page.url == url:
                    logger.info(f"Current page URL is the same as the new URL: {url}. No need to refresh.")
                    try:
                        title = await page.title()
                        await browser_logger.log_browser_interaction(
                            tool_name="openurl", action="navigate", interaction_type="navigation", success=True,
                            additional_data={"url": url, "title": title, "from_cache": True, "status": "already_loaded",
                                             "force_new_tab": force_new_tab, },
                        )
                        return f"Page already loaded: {url}, Title: {title}"
                    except Exception as e:
                        logger.error(
                            f"An error occurred while getting the page title: {e}, but will continue to load the page.")

                function_name = inspect.currentframe().f_code.co_name  # type: ignore
                await self.playwright_manager.take_screenshots(f"{function_name}_start", page)

                response = await page.goto(url, timeout=timeout * 10000)
                await self.playwright_manager.take_screenshots(f"{function_name}_end", page)

                title = await page.title()
                final_url = page.url
                status = response.status if response else None
                ok = response.ok if response else False

                await self.playwright_manager.wait_for_load_state_if_enabled(page=page, state="domcontentloaded")
                if timeout > 0:
                    await asyncio.sleep(timeout)
                await self.playwright_manager.wait_for_page_and_frames_load()

                await browser_logger.log_browser_interaction(
                    tool_name="openurl", action="navigate", interaction_type="navigation", success=True,
                    additional_data={
                        "url": url, "final_url": final_url, "title": title,
                        "status_code": status, "ok": ok, "from_cache": False, "force_new_tab": force_new_tab,
                    },
                )
                return f"Page loaded: {final_url}, Title: {title}"

            except PlaywrightTimeoutError as pte:
                await browser_logger.log_browser_interaction(
                    tool_name="openurl", action="navigate", interaction_type="navigation", success=False,
                    error_message=str(pte),
                    additional_data={"url": url, "error_type": "timeout", "timeout_seconds": timeout,
                                     "force_new_tab": force_new_tab, },
                )
                logger.warning(f"Initial navigation to {url} failed: {pte}. Will try to continue anyway.")
                return f"Timeout error opening URL: {url}"

            except Exception as e:
                await browser_logger.log_browser_interaction(
                    tool_name="openurl", action="navigate", interaction_type="navigation", success=False,
                    error_message=str(e),
                    additional_data={"url": url, "error_type": type(e).__name__, "force_new_tab": force_new_tab, },
                )
                logger.error(f"An error occurred while opening the URL: {url}. Error: {e}")
                traceback.print_exc()
                return f"Error opening URL: {url}"

        # Register existing tools (rest of the functions remain the same)
        @tool(agent_names=[self._agent_name], name="click_element",
              description="Clicks on an element identified by a CSS selector or its exact visible text.")
        async def click_element(selector: Annotated[
            Optional[str], "CSS selector of the element to click (e.g., 'button#submit')."] = None, text: Annotated[
            Optional[str], "Exact visible text of the element to click (e.g., 'Login button')."] = None) -> Annotated[
            str, "JSON string indicating success and message."]:
            result = await self.playwright_manager.click_element(selector=selector, text=text)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="switch_to_tab",
              description="Switch to a specific tab by index. Tab indices start from 0.")
        async def switch_to_tab(tab_index: Annotated[int, "The index of the tab to switch to."]) -> Annotated[
            str, "JSON string indicating success or failure."]:
            try:
                await self.playwright_manager.switch_to_tab(tab_index)
                return json.dumps({"success": True}, separators=(",", ":")).replace('"', "'")
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, separators=(",", ":")).replace('"', "'")


        @tool(agent_names=[self._agent_name], name="fill_field",
              description="Fills a text input field identified by a CSS selector with a given value.")
        async def fill_field(selector: Annotated[str, "CSS selector of the text field to fill."],
                             value: Annotated[str, "The value to fill into the text field."]) -> Annotated[
            str, "JSON string indicating success and message."]:
            result = await self.playwright_manager.fill_field(selector, value)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="get_page_content",
              description="Retrieves the full HTML content of the current page.")
        async def get_page_content() -> Annotated[str, "JSON string indicating success and the full HTML content."]:
            result = await self.playwright_manager.get_page_content()
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="get_text_from_selector",
              description="Retrieves the text content of an element identified by a CSS selector.")
        async def get_text_from_selector(
                selector: Annotated[str, "CSS selector of the element from which to extract text."]) -> Annotated[
            str, "JSON string indicating success and the extracted text."]:
            result = await self.playwright_manager.get_text_from_selector(selector)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        # @tool(
        #     agent_names=[self._agent_name],
        #     name="get_text_from_selector",
        #     description="""Retrieves the text content of an element using various locator strategies.
        #             Provide *one* of the following parameters:
        #             - selector: CSS selector of the element (e.g., 'div.my-class', '#my-id').
        #             - text: Exact visible text of the element (e.g., 'Submit Button').
        #             - role: ARIA role of the element (e.g., 'button', 'textbox', 'link').
        #             - test_id: Value of the 'data-testid' attribute (e.g., 'login-button').
        #             - placeholder: Placeholder text of an input field (e.g., 'Enter username').
        #             - label: Text of the label associated with an element (e.g., 'Username').
        #             - alt_text: Alt text of an image (e.g., 'Company Logo').
        #             - title_attr: Value of the 'title' attribute (e.g., 'Tooltip for button').
        #             """
        # )
        # async def get_text_from_selector(
        #         selector: Annotated[Optional[str], "CSS selector of the element."] = None,
        #         text: Annotated[Optional[str], "Exact visible text of the element."] = None,
        #         role: Annotated[Optional[str], "ARIA role of the element."] = None,
        #         test_id: Annotated[Optional[str], "Value of the 'data-testid' attribute."] = None,
        #         placeholder: Annotated[Optional[str], "Placeholder text of an input field."] = None,
        #         label: Annotated[Optional[str], "Text of the label associated with an element."] = None,
        #         alt_text: Annotated[Optional[str], "Alt text of an image."] = None,
        #         title_attr: Annotated[Optional[str], "Value of the 'title' attribute."] = None,
        # ) -> Annotated[str, "JSON string indicating success and the extracted text."]:
        #     result = await self.playwright_manager.get_text_from_selector(
        #         selector=selector,
        #         text=text,
        #         role=role,
        #         test_id=test_id,
        #         placeholder=placeholder,
        #         label=label,
        #         alt_text=alt_text,
        #         title_attr=title_attr,
        #     )
        #     return json.dumps(result, separators=(",", ":")).replace('"', "'")

        # @tool(agent_names=[self._agent_name], name="get_text_from_selector",
        #       description="Retrieves the text content of an element identified by a CSS selector or XPath.")
        # async def get_text_from_selector(
        #         selector: Annotated[str, "Selector of the element from which to extract text."],
        #         selector_type: Annotated[str, "Type of selector (css or xpath). Default is css."] = "css"
        # ) -> Annotated[str, "JSON string indicating success and the extracted text."]:
        #     if selector_type.lower() == "css":
        #         result = await self.playwright_manager.get_text_from_selector(selector)
        #     elif selector_type.lower() == "xpath":
        #         result = await self.playwright_manager.get_text_from_xpath(selector)
        #     else:
        #         raise ValueError("Invalid selector type. Supported types are 'css' and 'xpath'.")
        #
        #     return json.dumps(result, separators=(",", ":")).replace('"', "'")

        # @tool(
        #     agent_names=[self._agent_name],
        #     name="get_text_from_selector",
        #     description="""Retrieves the text content of an element using various locator strategies.
        #             Prioritizes ID, Name, CSS selectors over text, role, etc., and XPath as a last resort.
        #             Provide *one or more* of the following parameters to locate the element:
        #             - id_attr: ID attribute of the element (e.g., 'productName').
        #             - name_attr: Name attribute of the element (e.g., 'itemDescription').
        #             - css_selector: CSS selector of the element (e.g., 'div.price').
        #             - text: Exact visible text of the element (e.g., 'Product Title').
        #             - role: ARIA role of the element (e.g., 'heading', 'listitem').
        #             - test_id: Value of the 'data-testid' attribute.
        #             - placeholder: Placeholder text of an input field.
        #             - label: Text of the label associated with an element.
        #             - alt_text: Alt text of an image.
        #             - title_attr: Value of the 'title' attribute.
        #             - xpath: XPath of the element (use as a fallback).
        #             """
        # )
        # async def get_text_from_selector(
        #         id_attr: Annotated[Optional[str], "ID attribute of the element (e.g., 'productName')."] = None,
        #         name_attr: Annotated[Optional[str], "Name attribute of the element (e.g., 'itemDescription')."] = None,
        #         css_selector: Annotated[Optional[str], "CSS selector of the element (e.g., 'div.price')."] = None,
        #         text: Annotated[Optional[str], "Exact visible text of the element (e.g., 'Product Title')."] = None,
        #         role: Annotated[Optional[str], "ARIA role of the element (e.g., 'heading', 'listitem')."] = None,
        #         test_id: Annotated[Optional[str], "Value of the 'data-testid' attribute."] = None,
        #         placeholder: Annotated[Optional[str], "Placeholder text of an input field."] = None,
        #         label: Annotated[Optional[str], "Text of the label associated with an element."] = None,
        #         alt_text: Annotated[Optional[str], "Alt text of an image."] = None,
        #         title_attr: Annotated[Optional[str], "Value of the 'title' attribute."] = None,
        #         xpath: Annotated[Optional[str], "XPath of the element (use as a fallback)."] = None,
        # ) -> Annotated[str, "JSON string indicating success and the extracted text."]:
        #     result = await self.playwright_manager.get_text_from_selector(
        #         id_attr=id_attr, name_attr=name_attr, css_selector=css_selector,
        #         text=text, role=role, test_id=test_id, placeholder=placeholder,
        #         label=label, alt_text=alt_text, title_attr=title_attr, xpath=xpath,
        #     )
        #     return json.dumps(result, separators=(",", ":")).replace('"', "'")
        @tool(
            agent_names=["browser_nav_agent"],
            description="""Executes key press on page (Enter, PageDown, ArrowDown, etc.).""",
            name="press_key_combination",
        )
        async def press_key_combination(
                key_combination: Annotated[str, "key to press, e.g., Enter, PageDown etc"],
        ) -> str:
            logger.info(f"Executing press_key_combination with key combo: {key_combination}")
            # Create and use the PlaywrightManager
            browser_manager = self.playwright_manager()
            page = await browser_manager.get_current_page()

            if page is None:  # type: ignore
                raise ValueError("No active page found. OpenURL command opens a new page.")

            # Split the key combination if it's a combination of keys
            keys = key_combination.split("+")

            dom_changes_detected = None

            def detect_dom_changes(changes: str):  # type: ignore
                nonlocal dom_changes_detected
                dom_changes_detected = changes  # type: ignore

            subscribe(detect_dom_changes)
            # If it's a combination, hold down the modifier keys
            for key in keys[:-1]:  # All keys except the last one are considered modifier keys
                await page.keyboard.down(key)

            # Press the last key in the combination
            await page.keyboard.press(keys[-1])

            # Release the modifier keys
            for key in keys[:-1]:
                await page.keyboard.up(key)
            await asyncio.sleep(
                get_global_conf().get_delay_time())  # sleep for 100ms to allow the mutation observer to detect changes
            unsubscribe(detect_dom_changes)

            await browser_manager.wait_for_load_state_if_enabled(page=page)

            await browser_manager.take_screenshots("press_key_combination_end", page)
            if dom_changes_detected:
                return f"Key {key_combination} executed successfully.\n As a consequence of this action, new elements have appeared in view:{dom_changes_detected}. This means that the action is not yet executed and needs further interaction. Get all_fields DOM to complete the interaction."

            return f"Key {key_combination} executed successfully"


        def get_js_with_element_finder(action_js_code: str) -> str:
            """
            Combines the element finder code with specific action code.

            Args:
                action_js_code: JavaScript code that uses findElementInShadowDOMAndIframes

            Returns:
                Combined JavaScript code
            """
            pattern = "/*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/"
            if pattern in action_js_code:
                return action_js_code.replace(pattern, TEMPLATES["FIND_ELEMENT_IN_SHADOW_DOM"])
            else:
                return action_js_code

        FIND_ELEMENT_IN_SHADOW_DOM = """
        const findElementInShadowDOMAndIframes = (parent, selector) => {
            // Try to find the element in the current context
            let element = parent.querySelector(selector);
            if (element) {
                return element; // Element found in the current context
            }

            // Search inside shadow DOMs and iframes
            const elements = parent.querySelectorAll('*');
            for (const el of elements) {
                // Search inside shadow DOMs
                if (el.shadowRoot) {
                    element = findElementInShadowDOMAndIframes(el.shadowRoot, selector);
                    if (element) {
                        return element; // Element found in shadow DOM
                    }
                }
                // Search inside iframes
                if (el.tagName.toLowerCase() === 'iframe') {
                    let iframeDocument;
                    try {
                        // Access the iframe's document if it's same-origin
                        iframeDocument = el.contentDocument || el.contentWindow.document;
                    } catch (e) {
                        // Cannot access cross-origin iframe; skip to the next element
                        continue;
                    }
                    if (iframeDocument) {
                        element = findElementInShadowDOMAndIframes(iframeDocument, selector);
                        if (element) {
                            return element; // Element found inside iframe
                        }
                    }
                }
            }
            return null; // Element not found
        };
        """

        TEMPLATES = {"FIND_ELEMENT_IN_SHADOW_DOM": FIND_ELEMENT_IN_SHADOW_DOM}

        async def custom_fill_element(page: Page, selector: str, text_to_enter: str) -> None:
            selector = f"{selector}"  # Ensures the selector is treated as a string
            try:
                js_code = """(inputParams) => {
                    /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                    const selector = inputParams.selector;
                    let text_to_enter = inputParams.text_to_enter.trim();

                    // Start by searching in the regular document (DOM)
                    const element = findElementInShadowDOMAndIframes(document, selector);

                    if (!element) {
                        throw new Error(`Element not found: ${selector}`);
                    }

                    // Set the value for the element
                    element.value = "";
                    element.value = text_to_enter;
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true }));

                    return `Value set for ${selector}`;
                }"""

                result = await page.evaluate(
                    get_js_with_element_finder(js_code),
                    {"selector": selector, "text_to_enter": text_to_enter},
                )
                logger.debug(f"custom_fill_element result: {result}")
            except Exception as e:

                traceback.print_exc()
                logger.error(
                    f"Error in custom_fill_element, Selector: {selector}, Text: {text_to_enter}. Error: {str(e)}")
                raise

        async def entertext(
                entry: Annotated[
                    tuple[str, str],
                    "tuple containing 'selector' and 'value_to_fill' in ('selector', 'value_to_fill') format, selector is md attribute value of the dom element to interact, md is an ID and 'value_to_fill' is the value or text of the option to select",
                ],
        ) -> Annotated[str, "Text entry result"]:

            logger.info(f"Entering text: {entry}")

            selector: str = entry[0]
            text_to_enter: str = entry[1]

            if "md=" not in selector:
                selector = f"[md='{selector}']"

            # Create and use the PlaywrightManager
            browser_manager = self.playwright_manager()
            page = await browser_manager.get_current_page()
            # await page.route("**/*", block_ads)
            if page is None:  # type: ignore
                return "Error: No active page found. OpenURL command opens a new page."

            function_name = inspect.currentframe().f_code.co_name  # type: ignore

            await browser_manager.take_screenshots(f"{function_name}_start", page)

            # await browser_manager.highlight_element(selector)

            dom_changes_detected = None

            def detect_dom_changes(changes: str):  # type: ignore
                nonlocal dom_changes_detected
                dom_changes_detected = changes  # type: ignore

            subscribe(detect_dom_changes)

            await page.evaluate(
                get_js_with_element_finder(
                    """
                (selector) => {
                    /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                    const element = findElementInShadowDOMAndIframes(document, selector);
                    if (element) {
                        element.value = '';
                    } else {
                        console.error('Element not found:', selector);
                    }
                }
                """
                ),
                selector,
            )

            result = await do_entertext(page, selector, text_to_enter)
            await asyncio.sleep(
                get_global_conf().get_delay_time())  # sleep to allow the mutation observer to detect changes
            unsubscribe(detect_dom_changes)

            await browser_manager.wait_for_load_state_if_enabled(page=page)

            await browser_manager.take_screenshots(f"{function_name}_end", page)

            if dom_changes_detected:
                return f"{result['detailed_message']}.\n As a consequence of this action, new elements have appeared in view: {dom_changes_detected}. This means that the action of entering text {text_to_enter} is not yet executed and needs further interaction. Get all_fields DOM to complete the interaction."
            return result["detailed_message"]

        async def get_element_outer_html(element: ElementHandle, page: Page,
                                         element_tag_name: str | None = None) -> str:
            """
            Constructs the opening tag of an HTML element along with its attributes.

            Args:
                element (ElementHandle): The element to retrieve the opening tag for.
                page (Page): The page object associated with the element.
                element_tag_name (str, optional): The tag name of the element. Defaults to None. If not passed, it will be retrieved from the element.

            Returns:
                str: The opening tag of the HTML element, including a select set of attributes.
            """
            tag_name: str = element_tag_name if element_tag_name else await page.evaluate(
                "element => element.tagName.toLowerCase()", element)

            attributes_of_interest: list[str] = [
                "id",
                "name",
                "aria-label",
                "placeholder",
                "href",
                "src",
                "aria-autocomplete",
                "role",
                "type",
                "data-testid",
                "value",
                "selected",
                "aria-labelledby",
                "aria-describedby",
                "aria-haspopup",
                "title",
                "aria-controls",
            ]
            opening_tag: str = f"<{tag_name}"

            for attr in attributes_of_interest:
                value: str = await element.get_attribute(attr)  # type: ignore
                if value:
                    opening_tag += f' {attr}="{value}"'
            opening_tag += ">"

            return opening_tag

        async def do_entertext(page: Page, selector: str, text_to_enter: str, use_keyboard_fill: bool = True) -> dict[
            str, str]:
            try:
                logger.debug(f"Looking for selector {selector} to enter text: {text_to_enter}")

                browser_manager = self.playwright_manager()
                elem = await browser_manager.find_element(selector, page, element_name="entertext")

                # Initialize selector logger with proof path
                selector_logger = get_browser_logger(get_global_conf().get_proof_path())

                if not elem:
                    # Log failed selector interaction
                    await selector_logger.log_selector_interaction(
                        tool_name="entertext",
                        selector=selector,
                        action="input",
                        selector_type="css" if "md=" in selector else "custom",
                        success=False,
                        error_message=f"Error: Selector {selector} not found. Unable to continue.",
                    )
                    error = f"Error: Selector {selector} not found. Unable to continue."
                    return {"summary_message": error, "detailed_message": error}
                else:
                    # Get element properties to determine the best selection strategy
                    tag_name = await elem.evaluate("el => el.tagName.toLowerCase()")
                    element_role = await elem.evaluate("el => el.getAttribute('role') || ''")
                    element_type = await elem.evaluate("el => el.type || ''")
                    input_roles = ["combobox", "listbox", "dropdown", "spinner", "select"]
                    input_types = [
                        "range",
                        "combobox",
                        "listbox",
                        "dropdown",
                        "spinner",
                        "select",
                        "option",
                    ]
                    logger.info(f"element_role: {element_role}, element_type: {element_type}")
                    if element_role in input_roles or element_type in input_types:
                        properties = {
                            "tag_name": tag_name,
                            "element_role": element_role,
                            "element_type": element_type,
                            "element_outer_html": await get_element_outer_html(elem, page),
                            "alternative_selectors": await selector_logger.get_alternative_selectors(elem, page),
                            "element_attributes": await selector_logger.get_element_attributes(elem),
                            "selector_logger": selector_logger,
                        }
                        return await interact_with_element_select_type(page, elem, selector, text_to_enter, properties)

                logger.info(f"Found selector {selector} to enter text")
                element_outer_html = await get_element_outer_html(elem, page)

                # Initialize selector logger with proof path
                selector_logger = get_browser_logger(get_global_conf().get_proof_path())
                # Get alternative selectors and element attributes for logging
                alternative_selectors = await selector_logger.get_alternative_selectors(elem, page)
                element_attributes = await selector_logger.get_element_attributes(elem)

                if use_keyboard_fill:
                    await elem.focus()
                    await asyncio.sleep(0.01)
                    await press_key_combination("Control+A")
                    await asyncio.sleep(0.01)
                    await press_key_combination("Delete")
                    await asyncio.sleep(0.01)
                    logger.debug(f"Focused element with selector {selector} to enter text")
                    await page.keyboard.type(text_to_enter, delay=1)
                else:
                    await custom_fill_element(page, selector, text_to_enter)

                await elem.focus()
                await browser_manager.wait_for_load_state_if_enabled(page=page)

                # Log successful selector interaction
                await selector_logger.log_selector_interaction(
                    tool_name="entertext",
                    selector=selector,
                    action="input",
                    selector_type="css" if "md=" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=True,
                    additional_data={
                        "text_entered": text_to_enter,
                        "input_method": "keyboard" if use_keyboard_fill else "javascript",
                    },
                )

                logger.info(f'Success. Text "{text_to_enter}" set successfully in the element with selector {selector}')
                success_msg = f'Success. Text "{text_to_enter}" set successfully in the element with selector {selector}'
                return {
                    "summary_message": success_msg,
                    "detailed_message": f"{success_msg} and outer HTML: {element_outer_html}.",
                }

            except Exception as e:

                traceback.print_exc()
                # Initialize selector logger with proof path
                selector_logger = get_browser_logger(get_global_conf().get_proof_path())
                # Log failed selector interaction
                await selector_logger.log_selector_interaction(
                    tool_name="entertext",
                    selector=selector,
                    action="input",
                    selector_type="css" if "md=" in selector else "custom",
                    success=False,
                    error_message=str(e),
                )

                traceback.print_exc()
                error = f"Error entering text in selector {selector}."
                return {"summary_message": error, "detailed_message": f"{error} Error: {e}"}

        page_data_store = {}

        def set_page_data(page: Any, data: Any) -> None:
            page_data_store[page] = data

        # Function to get data
        def get_page_data(page: Any) -> dict[str, Any]:
            data = page_data_store.get(page)
            return data if data is not None else {}

        @tool(
            agent_names=["browser_nav_agent"],
            description="""Clicks element by md attribute. Returns success/failure status.""",
            name="click",
        )
        async def click(
                selector: Annotated[str, """selector using md attribute, just give the md ID value"""],
                user_input_dialog_response: Annotated[str, "Dialog input value"] = "",
                expected_message_of_dialog: Annotated[str, "Expected dialog message"] = "",
                action_on_dialog: Annotated[str, "Dialog action: 'DISMISS' or 'ACCEPT'"] = "",
                type_of_click: Annotated[str, "Click type: click/right_click/double_click/middle_click"] = "click",
                wait_before_execution: Annotated[float, "Wait time before click"] = 0.0,
        ) -> Annotated[str, "Click action result"]:
            query_selector = selector

            # if "md=" not in query_selector:
            #     query_selector = f"[md='{query_selector}']"

            logger.info(f'Executing ClickElement with "{query_selector}" as the selector')

            # Initialize PlaywrightManager and get the active browser page
            browser_manager = self.playwright_manager
            page = await browser_manager.get_current_page()
            # await page.route("**/*", block_ads)
            action_on_dialog = action_on_dialog.lower() if action_on_dialog else ""
            type_of_click = type_of_click.lower() if type_of_click else "click"

            async def handle_dialog(dialog: Any) -> None:
                try:
                    await asyncio.sleep(0.5)
                    data = get_page_data(page)
                    user_input_dialog_response = data.get("user_input_dialog_response", "")
                    expected_message_of_dialog = data.get("expected_message_of_dialog", "")
                    action_on_dialog = data.get("action_on_dialog", "")
                    if action_on_dialog:
                        action_on_dialog = action_on_dialog.lower().strip()
                    dialog_message = dialog.message if dialog.message is not None else ""
                    logger.info(f"Dialog message: {dialog_message}")

                    # Check if the dialog message matches the expected message (if provided)
                    if expected_message_of_dialog and dialog_message != expected_message_of_dialog:
                        logger.error(
                            f"Dialog message does not match the expected message: {expected_message_of_dialog}")
                        if action_on_dialog == "accept":
                            if dialog.type == "prompt":
                                await dialog.accept(user_input_dialog_response)
                            else:
                                await dialog.accept()
                        elif action_on_dialog == "dismiss":
                            await dialog.dismiss()
                        else:
                            await dialog.dismiss()  # Dismiss if the dialog message doesn't match
                    elif user_input_dialog_response:
                        await dialog.accept(user_input_dialog_response)
                    else:
                        await dialog.dismiss()

                except Exception as e:

                    traceback.print_exc()
                    logger.info(f"Error handling dialog: {e}")

            if page is None:  # type: ignore
                raise ValueError("No active page found. OpenURL command opens a new page.")

            function_name = inspect.currentframe().f_code.co_name  # type: ignore

            await browser_manager.take_screenshots(f"{function_name}_start", page)

            # await browser_manager.highlight_element(query_selector)

            dom_changes_detected = None

            def detect_dom_changes(changes: str):  # type: ignore
                nonlocal dom_changes_detected
                dom_changes_detected = changes  # type: ignore

            subscribe(detect_dom_changes)
            set_page_data(
                page,
                {
                    "user_input_dialog_response": user_input_dialog_response,
                    "expected_message_of_dialog": expected_message_of_dialog,
                    "action_on_dialog": action_on_dialog,
                    "type_of_click": type_of_click,
                },
            )

            page = await browser_manager.get_current_page()
            page.on("dialog", handle_dialog)
            result = await do_click(page, query_selector, wait_before_execution, type_of_click)

            await asyncio.sleep(
                1000)  # sleep to allow the mutation observer to detect changes
            unsubscribe(detect_dom_changes)

            await browser_manager.wait_for_load_state_if_enabled(page=page)

            await browser_manager.take_screenshots(f"{function_name}_end", page)

            if dom_changes_detected:
                return f"Success: {result['summary_message']}.\n As a consequence of this action, new elements have appeared in view: {dom_changes_detected}. This means that the action to click {query_selector} is not yet executed and needs further interaction. Get all_fields DOM to complete the interaction."
            return result["detailed_message"]

        async def do_click(page: Page, selector: str, wait_before_execution: float, type_of_click: str) -> dict[
            str, str]:
            logger.info(
                f'Executing ClickElement with "{selector}" as the selector. Wait time before execution: {wait_before_execution} seconds.')

            # Wait before execution if specified
            if wait_before_execution > 0:
                await asyncio.sleep(wait_before_execution)

            # Wait for the selector to be present and ensure it's attached and visible. If timeout, try JavaScript click
            try:
                logger.info(
                    f'Executing ClickElement with "{selector}" as the selector. Waiting for the element to be attached and visible.')

                # Attempt to find the element on the main page or in iframes
                browser_manager = self.playwright_manager
                element = await browser_manager.find_element(selector, page, element_name="click")
                if element is None:
                    # Initialize selector logger with proof path
                    selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
                    # Log failed selector interaction
                    await selector_logger.log_selector_interaction(
                        tool_name="click",
                        selector=selector,
                        action=type_of_click,
                        selector_type="css" if "md=" in selector else "custom",
                        success=False,
                        error_message=f'Element with selector: "{selector}" not found',
                    )
                    raise ValueError(f'Element with selector: "{selector}" not found')

                logger.info(f'Element with selector: "{selector}" is attached. Scrolling it into view if needed.')
                try:
                    await element.scroll_into_view_if_needed(timeout=2000)
                    logger.info(
                        f'Element with selector: "{selector}" is attached and scrolled into view. Waiting for the element to be visible.')
                except Exception as e:

                    traceback.print_exc()
                    logger.exception(f"Error scrolling element into view: {e}")
                    # If scrollIntoView fails, just move on, not a big deal
                    pass

                if not await element.is_visible():
                    return {
                        "summary_message": f'Element with selector: "{selector}" is not visible, Try another element',
                        "detailed_message": f'Element with selector: "{selector}" is not visible, Try another element',
                    }

                element_tag_name = await element.evaluate("element => element.tagName.toLowerCase()")
                element_outer_html = await get_element_outer_html(element, page, element_tag_name)

                # Initialize selector logger with proof path
                selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
                # Get alternative selectors and element attributes for logging
                alternative_selectors = await selector_logger.get_alternative_selectors(element, page)
                element_attributes = await selector_logger.get_element_attributes(element)

                # hack for aura component in salesforce
                element_title = (await element.get_attribute("title") or "").lower()
                if "upload" in element_title:
                    return {
                        "summary_message": "Use the click_and_upload_file tool to upload files",
                        "detailed_message": "Use the click_and_upload_file tool to upload files",
                    }

                if element_tag_name == "option":
                    element_value = await element.get_attribute("value")
                    parent_element = await element.evaluate_handle("element => element.parentNode")
                    await parent_element.select_option(value=element_value)  # type: ignore

                    # Log successful selector interaction for option selection
                    await selector_logger.log_selector_interaction(
                        tool_name="click",
                        selector=selector,
                        action="select_option",
                        selector_type="css" if "md=" in selector else "custom",
                        alternative_selectors=alternative_selectors,
                        element_attributes=element_attributes,
                        success=True,
                        additional_data={"selected_value": element_value},
                    )

                    logger.info(f'Select menu option "{element_value}" selected')

                    return {
                        "summary_message": f'Select menu option "{element_value}" selected',
                        "detailed_message": f'Select menu option "{element_value}" selected. The select element\'s outer HTML is: {element_outer_html}.',
                    }

                input_type = await element.evaluate("(el) => el.type")

                # Determine if it's checkable
                if element_tag_name == "input" and input_type in ["radio"]:
                    await element.check()
                    msg = f'Checked element with selector: "{selector}"'
                elif element_tag_name == "input" and input_type in ["checkbox"]:
                    await element.type(" ")
                    msg = f'Checked element with selector: "{selector}"'
                else:
                    # Perform the click based on the type_of_click
                    if type_of_click == "right_click":
                        await element.click(button="right")
                        msg = f'Right-clicked element with selector: "{selector}"'
                    elif type_of_click == "double_click":
                        await element.dblclick()
                        msg = f'Double-clicked element with selector: "{selector}"'
                    elif type_of_click == "middle_click":
                        await element.click(button="middle")
                        msg = f'Middle-clicked element with selector: "{selector}"'
                    else:  # Default to regular click
                        await element.click()
                        msg = f'Clicked element with selector: "{selector}"'

                # Log successful selector interaction
                await selector_logger.log_selector_interaction(
                    tool_name="click",
                    selector=selector,
                    action=type_of_click,
                    selector_type="css" if "md=" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=True,
                    additional_data={"click_type": type_of_click},
                )

                return {
                    "summary_message": msg,
                    "detailed_message": f"{msg} The clicked element's outer HTML is: {element_outer_html}.",
                }  # type: ignore
            except Exception as e:
                # Try a JavaScript fallback click before giving up

                traceback.print_exc()
                try:
                    logger.info(f'Standard click failed for "{selector}". Attempting JavaScript fallback click.')
                    browser_manager = self.playwright_manager
                    msg = await browser_manager.perform_javascript_click(page, selector, type_of_click)

                    if msg:
                        # Initialize selector logger with proof path
                        selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
                        # Log successful JavaScript fallback click
                        await selector_logger.log_selector_interaction(
                            tool_name="click",
                            selector=selector,
                            action=f"js_fallback_{type_of_click}",
                            selector_type="css" if "md=" in selector else "custom",
                            success=True,
                            additional_data={"click_type": "javascript_fallback"},
                        )

                        return {
                            "summary_message": msg,
                            "detailed_message": f"{msg}.",
                        }
                except Exception as js_error:

                    traceback.print_exc()
                    logger.error(f"JavaScript fallback click also failed: {js_error}")
                    # Both standard and fallback methods failed, proceed with original error handling
                    pass

                # Initialize selector logger with proof path
                selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
                # Log failed selector interaction
                await selector_logger.log_selector_interaction(
                    tool_name="click",
                    selector=selector,
                    action=type_of_click,
                    selector_type="css" if "md=" in selector else "custom",
                    success=False,
                    error_message=str(e),
                )

                logger.error(f'Unable to click element with selector: "{selector}". Error: {e}')
                traceback.print_exc()
                msg = f'Unable to click element with selector: "{selector}" since the selector is invalid. Proceed by retrieving DOM again.'
                return {"summary_message": msg, "detailed_message": f"{msg}. Error: {e}"}

        @tool(
            agent_names=["browser_nav_agent"],
            name="bulk_enter_text",
            description="Enters text into multiple text fields and textarea elements using a bulk operation based on detected fields details. An dict containing'selector' (selector query using md attribute e.g. [md='114'] md is ID) and 'text' (text to enter on the element)",
        )
        async def bulk_enter_text(
                entries: Annotated[
                    List[List[str]],
                    "List of tuple containing 'selector' and 'value_to_fill' in [('selector', 'value_to_fill'), ..] format, selector is md attribute value of the dom element to interact, md is an ID and 'value_to_fill' is the value or text",
                ],
        ) -> Annotated[
            List[Dict[str, str]],
            "List of dictionaries, each containing 'selector' and the result of the operation.",
        ]:

            results: List[Dict[str, str]] = []
            logger.info("Executing bulk set input value command")
            for entry in entries:
                if len(entry) != 2:
                    logger.error(f"Invalid entry format: {entry}. Expected [selector, value]")
                    continue
                result = await self.playwright_manager.entertextnew((entry[0], entry[1]))  # Create tuple with explicit values
                results.append({"selector": entry[0], "result": result})

            return results

        async def select_option(
                entry: Annotated[
                    tuple[str, str],
                    (
                            "tuple containing 'selector' and 'value_to_fill' in "
                            "('selector', 'value_to_fill') format. Selector is the md attribute value "
                            "of the DOM element and value_to_fill is the option text/value to select."
                    ),
                ],
        ) -> Annotated[str, "Explanation of the outcome of dropdown/spinner selection."]:

            logger.info(f"Selecting option: {entry}")
            selector: str = entry[0]
            option_value: str = entry[1]

            # If the selector doesn't contain md=, wrap it accordingly.
            # if "md=" not in selector:
            #     selector = f"[md='{selector}']"

            browser_manager = self.playwright_manager
            page = await browser_manager.get_current_page()
            if page is None:
                return "Error: No active page found. OpenURL command opens a new page."

            function_name = inspect.currentframe().f_code.co_name  # type: ignore
            await browser_manager.take_screenshots(f"{function_name}_start", page)
            # await browser_manager.highlight_element(selector)

            dom_changes_detected = None

            def detect_dom_changes(changes: str) -> None:
                nonlocal dom_changes_detected
                dom_changes_detected = changes

            subscribe(detect_dom_changes)
            result = await do_select_option(page, selector, option_value)
            # Wait for page to stabilize after selection
            await browser_manager.wait_for_load_state_if_enabled(page=page)
            unsubscribe(detect_dom_changes)

            await browser_manager.wait_for_load_state_if_enabled(page=page)
            await browser_manager.take_screenshots(f"{function_name}_end", page)

            # Simply return the detailed message
            return result["detailed_message"]

        async def do_select_option(page: Page, selector: str, option_value: str) -> dict[str, str]:
            """
            Simplified approach to select an option in a dropdown using the element's properties.
            Uses find_element to get the element and then determines the best strategy based on
            the element's role, type, and tag name.
            """
            try:
                logger.debug(f"Looking for selector {selector} to select option: {option_value}")

                # Part 1: Find the element and get its properties
                element, properties = await find_element_select_type(page, selector)
                if not element:
                    error = f"Error: Selector '{selector}' not found. Unable to continue."
                    return {"summary_message": error, "detailed_message": error}

                # Part 2: Interact with the element to select the option
                return await interact_with_element_select_type(page, element, selector, option_value, properties)

            except Exception as e:

                traceback.print_exc()
                selector_logger = get_browser_loggernew(get_global_conf().get_proof_path())
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if "md=" in selector else "custom",
                    success=False,
                    error_message=str(e),
                )
                traceback.print_exc()
                error = f"Error selecting option in selector '{selector}'."
                return {"summary_message": error, "detailed_message": f"{error} Error: {e}"}

        async def find_element(
                self,
                selector: str,
                page: Optional[Page] = None,
                element_name: Optional[str] = None,
        ) -> Optional[ElementHandle]:
            """Find element in DOM/Shadow DOM/iframes and return ElementHandle.
            Also captures a screenshot with the element's bounding box and metadata overlay if enabled.

            Args:
                selector: The selector to find the element
                page: Optional page instance to search in
                element_name: Optional friendly name for the element (used in screenshot naming)
            """
            if page is None:
                page = await self.get_current_page()

            # Try regular DOM first
            element = await page.query_selector(selector)
            if element:
                if self._take_bounding_box_screenshots:
                    await self._capture_element_with_bbox(
                        element, page, selector, element_name
                    )
                return element

            # Check Shadow DOM and iframes
            js_code = """(selector) => {
                /*INJECT_FIND_ELEMENT_IN_SHADOW_DOM*/
                return findElementInShadowDOMAndIframes(document, selector);
            }"""

            element = await page.evaluate_handle(
                get_js_with_element_finder(js_code), selector
            )
            if element:
                element_handle = element.as_element()
                if element_handle:
                    if self._take_bounding_box_screenshots:
                        await self._capture_element_with_bbox(
                            element_handle, page, selector, element_name
                        )
                return element_handle

            return None

        async def find_element_select_type(page: Page, selector: str) -> tuple[Optional[ElementHandle], dict]:
            """
            Internal function to find the element and gather its properties.
            Returns the element and a dictionary of its properties.
            """
            browser_manager = self.playwright_manager()
            element = await browser_manager.find_element(selector, page, element_name="select_option")

            if not element:
                selector_logger = get_browser_logger(get_global_conf().get_proof_path())
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if "md=" in selector else "custom",
                    success=False,
                    error_message=f"Error: Selector '{selector}' not found. Unable to continue.",
                )
                return None, {}

            logger.info(f"Found selector '{selector}' to select option")
            selector_logger = get_browser_logger(get_global_conf().get_proof_path())
            alternative_selectors = await selector_logger.get_alternative_selectors(element, page)
            element_attributes = await selector_logger.get_element_attributes(element)

            # Get element properties to determine the best selection strategy
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            element_role = await element.evaluate("el => el.getAttribute('role') || ''")
            element_type = await element.evaluate("el => el.type || ''")
            element_outer_html = await get_element_outer_html(element, page)

            properties = {
                "tag_name": tag_name,
                "element_role": element_role,
                "element_type": element_type,
                "element_outer_html": element_outer_html,
                "alternative_selectors": alternative_selectors,
                "element_attributes": element_attributes,
                "selector_logger": selector_logger,
            }

            return element, properties

        async def interact_with_element_select_type(
                page: Page,
                element: ElementHandle,
                selector: str,
                option_value: str,
                properties: dict,
        ) -> dict[str, str]:
            """
            Internal function to interact with the element to select the option.
            """
            tag_name = properties["tag_name"]
            element_role = properties["element_role"]
            element_type = properties["element_type"]
            element_outer_html = properties["element_outer_html"]
            alternative_selectors = properties["alternative_selectors"]
            element_attributes = properties["element_attributes"]
            selector_logger = properties["selector_logger"]

            # Strategy 1: Standard HTML select element
            if tag_name == "select":
                await element.select_option(value=option_value)
                await page.wait_for_load_state("domcontentloaded", timeout=1000)
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if "md=" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=True,
                    additional_data={
                        "element_type": "select",
                        "selected_value": option_value,
                    },
                )
                success_msg = f"Success. Option '{option_value}' selected in the dropdown with selector '{selector}'"
                return {
                    "summary_message": success_msg,
                    "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
                }

            # Strategy 2: Input elements (text, number, etc.)
            elif tag_name in ["input", "button"]:
                input_roles = ["combobox", "listbox", "dropdown", "spinner", "select"]
                input_types = [
                    "number",
                    "range",
                    "combobox",
                    "listbox",
                    "dropdown",
                    "spinner",
                    "select",
                    "option",
                ]

                if element_type in input_types or element_role in input_roles:
                    await element.click()
                    try:
                        await element.fill(option_value)
                    except Exception as e:

                        # traceback.print_exc()
                        logger.warning(f"Error filling input: {str(e)}, trying type instead")
                        await element.type(option_value)

                    if "lwc" in str(element) and "placeholder" in str(element):
                        logger.info("Crazy LWC element detected")
                        await asyncio.sleep(0.5)
                        await press_key_combination("ArrowDown+Enter")
                    else:
                        await element.press("Enter")

                    await page.wait_for_load_state("domcontentloaded", timeout=1000)

                    await selector_logger.log_selector_interaction(
                        tool_name="select_option",
                        selector=selector,
                        action="input",
                        selector_type="css" if "md=" in selector else "custom",
                        alternative_selectors=alternative_selectors,
                        element_attributes=element_attributes,
                        success=True,
                        additional_data={
                            "element_type": "input",
                            "input_type": element_type,
                            "value": option_value,
                        },
                    )
                    success_msg = f"Success. Value '{option_value}' set in the input with selector '{selector}'"
                    return {
                        "summary_message": success_msg,
                        "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
                    }

            # Strategy 3: Generic click and select approach for all other elements
            # Click to open the dropdown

            logger.info(f"taking worst case scenario of selecting option for {element}, properties: {properties}")
            await element.click()
            await page.wait_for_timeout(300)  # Short wait for dropdown to appear

            # Try to find and click the option by text content
            try:
                # Use a simple text-based selector that works in most cases
                option_selector = f"text={option_value}"
                await page.click(option_selector, timeout=2000)
                await page.wait_for_load_state("domcontentloaded", timeout=1000)

                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="click_by_text",
                    selector_type="css" if "md=" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=True,
                    additional_data={
                        "element_type": tag_name,
                        "selected_value": option_value,
                        "method": "text_content",
                    },
                )
                success_msg = f"Success. Option '{option_value}' selected by text content"
                return {
                    "summary_message": success_msg,
                    "detailed_message": f"{success_msg}. Outer HTML: {element_outer_html}",
                }
            except Exception as e:

                traceback.print_exc()
                logger.debug(f"Text-based selection failed: {str(e)}")

                # If all attempts fail, report failure
                await selector_logger.log_selector_interaction(
                    tool_name="select_option",
                    selector=selector,
                    action="select",
                    selector_type="css" if "md=" in selector else "custom",
                    alternative_selectors=alternative_selectors,
                    element_attributes=element_attributes,
                    success=False,
                    error_message=f"Could not find option '{option_value}' in the dropdown with any selection method.",
                )
                error = f"Error: Option '{option_value}' not found in the element with selector '{selector}'. Try clicking the element first and then select the option."
                return {"summary_message": error, "detailed_message": error}



        @tool(
            agent_names=["browser_nav_agent"],
            name="bulk_select_option",
            description=(
            "Used to select/search an options in multiple picklist/listbox/combobox/dropdowns/spinners in a single attempt. " "Each entry is a tuple of (selector, value_to_fill)."),
        )
        async def bulk_select_option(
                entries: Annotated[
                    List[List[str]],
                    (
                            "List of tuples containing 'selector' and 'value_to_fill' in the format "
                            "[('selector', 'value_to_fill'), ...]. 'selector' is the md attribute value and 'value_to_fill' is the option to select."
                    ),
                ],
        ) -> Annotated[
            List[Dict[str, str]],
            "List of dictionaries, each containing 'selector' and the result of the operation.",
        ]:

            results: List[Dict[str, str]] = []
            logger.info("Executing bulk select option command")

            for entry in entries:
                if len(entry) != 2:
                    logger.error(f"Invalid entry format: {entry}. Expected [selector, value]")
                    continue
                result = await self.playwright_manager.select_option((entry[0], entry[1]))
                if isinstance(result, str):
                    if "new elements have appeared in view" in result and "success" in result.lower():
                        success_part = result.split(".\nAs a consequence")[0]
                        results.append({"selector": entry[0], "result": success_part})
                    else:
                        results.append({"selector": entry[0], "result": result})
                else:
                    results.append({"selector": entry[0], "result": str(result)})
            return results

        @tool(agent_names=[self._agent_name], name="take_screenshot",
              description="Takes a screenshot of the current page. Returns the path to the saved screenshot.")
        async def take_screenshot(path: Annotated[Optional[
            str], "Optional path to save the screenshot (e.g., 'my_screenshot.png'). If not provided, a default path will be used."] = None) -> \
        Annotated[str, "JSON string indicating success and the screenshot file path."]:
            result = await self.playwright_manager.take_screenshot(path)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="scroll_page",
              description="Scrolls the page up or down by a specified number of pixels.")
        async def scroll_page(
                direction: Annotated[Literal["up", "down"], "Direction to scroll ('up' or 'down')."] = "down",
                pixels: Annotated[int, "Number of pixels to scroll (default is 500)."] = 500) -> Annotated[
            str, "JSON string indicating success and message."]:
            result = await self.playwright_manager.scroll_page(direction=direction, pixels=pixels)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="get_element_attribute",
              description="Retrieves the value of a specific attribute from an element identified by a CSS selector.")
        async def get_element_attribute(selector: Annotated[str, "CSS selector of the element."],
                                        attribute_name: Annotated[
                                            str, "The name of the attribute to retrieve (e.g., 'href', 'id', 'class')."]) -> \
        Annotated[str, "JSON string indicating success and the attribute value."]:
            result = await self.playwright_manager.get_element_attribute(selector, attribute_name)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="wait_for_selector",
              description="Waits for an element matching the selector to satisfy a state condition.")
        async def wait_for_selector(selector: Annotated[str, "CSS selector of the element to wait for."],
                                    timeout: Annotated[
                                        int, "Maximum time to wait in milliseconds (default: 30000)."] = 30000,
                                    state: Annotated[Literal[
                                        "attached", "detached", "visible", "hidden"], "State to wait for ('attached', 'detached', 'visible', 'hidden')."] = "visible") -> \
        Annotated[str, "JSON string indicating success and message."]:
            result = await self.playwright_manager.wait_for_selector(selector, timeout=timeout, state=state)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        async def wait_for_non_loading_dom_state(page: Page, max_wait_seconds: int) -> None:
            end_time = asyncio.get_event_loop().time() + max_wait_seconds
            while asyncio.get_event_loop().time() < end_time:
                all_frames_ready = True
                for frame in page.frames:
                    dom_state = await frame.evaluate("document.readyState")
                    if dom_state == "loading":
                        all_frames_ready = False
                        break  # Exit the loop if any frame is still loading
                if all_frames_ready:
                    logger.debug("All frames have DOM state not 'loading'")
                    break  # Exit the outer loop if all frames are ready
                await asyncio.sleep(0.1)

        async def __inject_attributes(page: Page) -> None:
            """
            Injects 'md' and 'aria-keyshortcuts' into all DOM elements. If an element already has an 'aria-keyshortcuts',
            it renames it to 'orig-aria-keyshortcuts' before injecting the new 'aria-keyshortcuts'
            This will be captured in the accessibility tree and thus make it easier to reconcile the tree with the DOM.
            'aria-keyshortcuts' is choosen because it is not widely used aria attribute.
            """

            last_md = await page.evaluate(
                """() => {
                    // A recursive function to handle elements in DOM, shadow DOM, and iframes
                    const processElements = (elements, idCounter) => {
                        elements.forEach(element => {
                            // If the element has a shadowRoot, process its children too
                            if (element.shadowRoot) {
                                idCounter = processElements(element.shadowRoot.querySelectorAll('*'), idCounter);
                            }

                            // If the element is an iframe, process its contentDocument if accessible
                            if (element.tagName.toLowerCase() === 'iframe') {
                                let iframeDocument;
                                try {
                                    // Access the iframe's document if it's same-origin
                                    iframeDocument = element.contentDocument || element.contentWindow.document;
                                } catch (e) {
                                    // Cannot access cross-origin iframe; skip to the next element
                                    return;
                                }
                                if (iframeDocument) {
                                    const iframeElements = iframeDocument.querySelectorAll('*');
                                    idCounter = processElements(iframeElements, idCounter);
                                }
                            }

                            // Check if the element is interactive (buttons, inputs, etc.)
                            if (isInteractiveElement(element)) {
                                const origAriaAttribute = element.getAttribute('aria-keyshortcuts');
                                const md = `${++idCounter}`;
                                element.setAttribute('md', md);
                                element.setAttribute('aria-keyshortcuts', md);

                                // Preserve the original aria-keyshortcuts if it exists
                                if (origAriaAttribute) {
                                    element.setAttribute('orig-aria-keyshortcuts', origAriaAttribute);
                                }
                            }
                        });
                        return idCounter;
                    };
                    function isInteractiveElement(element) {
                        // Immediately return false for body tag
                        if (element.tagName.toLowerCase() === 'body') {
                            return false;
                        }

                        // Base interactive elements and roles
                        const interactiveElements = new Set([
                            'a', 'button', 'details', 'embed', 'input', 'label',
                            'menu', 'menuitem', 'object', 'select', 'textarea', 'summary'
                        ]);

                        const interactiveRoles = new Set([
                            'button', 'menu', 'menuitem', 'link', 'checkbox', 'radio',
                            'slider', 'tab', 'tabpanel', 'textbox', 'combobox', 'grid',
                            'listbox', 'option', 'progressbar', 'scrollbar', 'searchbox',
                            'switch', 'tree', 'treeitem', 'spinbutton', 'tooltip', 'a-button-inner', 'a-dropdown-button', 'click', 
                            'menuitemcheckbox', 'menuitemradio', 'a-button-text', 'button-text', 'button-icon', 'button-icon-only', 'button-text-icon-only', 'dropdown', 'combobox'
                        ]);

                        const tagName = element.tagName.toLowerCase();
                        const role = element.getAttribute('role');
                        const ariaRole = element.getAttribute('aria-role');
                        const tabIndex = element.getAttribute('tabindex');

                        // Add check for specific class
                        const hasAddressInputClass = element.classList.contains('address-input__container__input');

                        // Basic role/attribute checks
                        const hasInteractiveRole = hasAddressInputClass ||
                            interactiveElements.has(tagName) ||
                            interactiveRoles.has(role) ||
                            interactiveRoles.has(ariaRole) ||
                            (tabIndex !== null && tabIndex !== '-1' && element.parentElement?.tagName.toLowerCase() !== 'body') ||
                            element.getAttribute('data-action') === 'a-dropdown-select' ||
                            element.getAttribute('data-action') === 'a-dropdown-button';

                        if (hasInteractiveRole) return true;

                        // Get computed style
                        const style = window.getComputedStyle(element);

                        if (
                            style.cursor === 'pointer' || 
                            style.cursor === 'hand' ||
                            style.cursor === 'move' ||
                            style.cursor === 'grab' ||
                            style.cursor === 'grabbing'
                        ) {
                            return true;
                        }

                        // Check for event listeners
                        const hasClickHandler = element.onclick !== null ||
                            element.getAttribute('onclick') !== null ||
                            element.hasAttribute('ng-click') ||
                            element.hasAttribute('@click') ||
                            element.hasAttribute('v-on:click');

                        // Helper function to safely get event listeners
                        function getEventListeners(el) {
                            try {
                                // Try to get listeners using Chrome DevTools API
                                return window.getEventListeners?.(el) || {};
                            } catch (e) {
                                // Fallback: check for common event properties
                                const listeners = {};

                                // List of common event types to check
                                const eventTypes = [
                                    'click', 'mousedown', 'mouseup',
                                    'touchstart', 'touchend',
                                    'keydown', 'keyup', 'focus', 'blur'
                                ];

                                for (const type of eventTypes) {
                                    const handler = el[`on${type}`];
                                    if (handler) {
                                        listeners[type] = [{
                                            listener: handler,
                                            useCapture: false
                                        }];
                                    }
                                }

                                return listeners;
                            }
                        }

                        // Check for click-related events on the element itself
                        const listeners = getEventListeners(element);
                        const hasClickListeners = listeners && (
                            listeners.click?.length > 0 ||
                            listeners.mousedown?.length > 0 ||
                            listeners.mouseup?.length > 0 ||
                            listeners.touchstart?.length > 0 ||
                            listeners.touchend?.length > 0
                        );

                        // Check for ARIA properties that suggest interactivity
                        const hasAriaProps = element.hasAttribute('aria-expanded') ||
                            element.hasAttribute('aria-pressed') ||
                            element.hasAttribute('aria-selected') ||
                            element.hasAttribute('aria-checked');

                        // Check for form-related functionality
                        const isFormRelated = element.form !== undefined ||
                            element.hasAttribute('contenteditable') ||
                            style.userSelect !== 'none';

                        // Check if element is draggable
                        const isDraggable = element.draggable ||
                            element.getAttribute('draggable') === 'true';

                        // Additional check to prevent body from being marked as interactive
                        if (element.tagName.toLowerCase() === 'body' || element.parentElement?.tagName.toLowerCase() === 'body') {
                            return false;
                        }

                        return hasAriaProps ||
                            // hasClickStyling ||
                            hasClickHandler ||
                            hasClickListeners ||
                            // isFormRelated ||
                            isDraggable;
                    }

                    // Helper function to determine if an element is interactive
                    // const isInteractiveElement = (element) => {
                    //     const interactiveTags = ['button', 'a', 'input', 'select', 'textarea'];
                    //     return interactiveTags.includes(element.tagName.toLowerCase()) || element.hasAttribute('tabindex');
                    // };

                    // Start processing the DOM
                    const allElements = document.querySelectorAll('*');
                    let id = processElements(allElements, 0);

                    return id;
                };
                """
            )
            logger.debug(f"Added MD into {last_md} elements")

        async def do_get_accessibility_info(page: Page, only_input_fields: bool = False) -> dict[str, Any] | None:
            """
            Retrieves the accessibility information of a web page and saves it as JSON files.

            Args:
                page (Page): The page object representing the web page.
                only_input_fields (bool, optional): If True, only retrieves accessibility information for input fields.
                    Defaults to False.

            Returns:
                dict[str, Any] or None: The enhanced accessibility tree as a dictionary, or None if an error occurred.
            """
            await __inject_attributes(page)
            # accessibility_tree: dict[str, Any] = await page.accessibility.snapshot(interesting_only=True)  # type: ignore
            js_code = """
                    () => {
                        function generateAccessibilityTree(rootElement, level) {
                            const requiredAriaAttributesByRole = {
                                'alert': [],
                                'button': [],
                                'checkbox': ['aria-checked'],
                                'combobox': ['aria-expanded', 'aria-controls'],
                                'dialog': [],
                                'gridcell': [],
                                'link': [],
                                'listbox': ['aria-multiselectable'],
                                'menuitemcheckbox': ['aria-checked'],
                                'menuitemradio': ['aria-checked'],
                                'option': ['aria-selected'],
                                'progressbar': ['aria-valuenow'],
                                'radio': ['aria-checked'],
                                'scrollbar': ['aria-controls', 'aria-valuenow', 'aria-valuemin', 'aria-valuemax', 'aria-orientation'],
                                'searchbox': [],
                                'slider': ['aria-valuenow', 'aria-valuemin', 'aria-valuemax', 'aria-orientation'],
                                'spinbutton': ['aria-valuenow', 'aria-valuemin', 'aria-valuemax'],
                                'tab': ['aria-selected'],
                                'tabpanel': [],
                                'textbox': ['aria-multiline'],
                                'treeitem': ['aria-expanded']
                            };

                            function isElementHidden(element) {
                                const style = window.getComputedStyle(element);
                                return (
                                    style.display === 'none' ||
                                    style.visibility === 'hidden' ||
                                    element.getAttribute('aria-hidden') === 'true'
                                );
                            }

                            function getAccessibleName(element) {
                                try {
                                    // Chromium-based accessibility API
                                    if (window.getComputedAccessibleNode) {
                                        const accessibilityInfo = window.getComputedAccessibleNode(element);
                                        console.log('accessibilityInfo:', accessibilityInfo);
                                        if (accessibilityInfo?.name) {
                                            return cleanName(accessibilityInfo.name);
                                        }
                                    }
                                } catch (error) {
                                    console.warn("Chromium accessibility API failed, falling back to manual method:", error);
                                }

                                // Existing manual accessibility extraction
                                let name = element.getAttribute('aria-label');
                                if (name) return cleanName(name);

                                const labelledby = element.getAttribute('aria-labelledby');
                                if (labelledby) {
                                    const labelElement = document.getElementById(labelledby);
                                    if (labelElement) return cleanName(labelElement.innerText);
                                }

                                if (element.alt) return cleanName(element.alt);
                                if (element.title) return cleanName(element.title);
                                if (element.placeholder) return cleanName(element.placeholder);

                                if (
                                    element.value &&
                                    (element.tagName.toLowerCase() === 'input' || element.tagName.toLowerCase() === 'textarea')
                                ) {
                                    return cleanName(element.value);
                                }

                                if (element.innerText) return cleanName(element.innerText);
                                return '';
                            }

                            function cleanName(name) {
                                if (typeof name !== 'string') {
                                    console.warn('Expected a string, but received:', name);
                                    return '';
                                }
                                const firstLine = name.split('\\n')[0];
                                return firstLine.trim();
                            }

                            function getRole(element) {
                                const role = element.getAttribute('role');
                                if (role) return role;

                                const tagName = element.tagName.toLowerCase();
                                if (tagName === 'button') return 'button';
                                if (tagName === 'a' && element.hasAttribute('href')) return 'link';
                                if (tagName === 'input') {
                                    const type = element.type.toLowerCase();
                                    switch (type) {
                                        case 'button':
                                        case 'submit':
                                        case 'reset':
                                        case 'image':
                                            return 'button';
                                        case 'checkbox':
                                            return 'checkbox';
                                        case 'radio':
                                            return 'radio';
                                        case 'range':
                                            return 'slider';
                                        case 'number':
                                            return 'spinbutton';
                                        case 'search':
                                            return 'searchbox';
                                        case 'file':
                                            return 'button';
                                        case 'color':
                                        case 'date':
                                        case 'datetime-local':
                                        case 'month':
                                        case 'time':
                                        case 'week':
                                            return 'combobox';
                                        case 'email':
                                        case 'tel':
                                        case 'url':
                                        case 'password':
                                        case 'text':
                                            return 'textbox';
                                        case 'hidden':
                                            return '';
                                        default:
                                            return 'textbox';
                                    }
                                }

                                if (tagName === 'select') return 'listbox';
                                if (tagName === 'textarea') return 'textbox';
                                return '';
                            }

                            function getRequiredAriaAttributes(element) {
                                const role = getRole(element);
                                return requiredAriaAttributesByRole[role] || [];
                            }

                            function processElement(element, level) {
                                if (isElementHidden(element)) return null;

                                const node = {};

                                const md = element.getAttribute('md');
                                if (md) node.md = md;

                                node.tag = element.tagName.toLowerCase();

                                const role = getRole(element);
                                if (role) node.role = role;

                                const name = getAccessibleName(element);
                                if (name) node.name = name;

                                const title = element.getAttribute('title');
                                if (title) node.title = cleanName(title);

                                if (level) node.level = level;

                                node.children = [];

                                if (element.shadowRoot) {
                                    for (const child of element.shadowRoot.children) {
                                        const childNode = processElement(child, level + 1);
                                        if (childNode) node.children.push(childNode);
                                    }
                                }

                                for (const child of element.children) {
                                    if (child.tagName.toLowerCase() === 'iframe') {
                                        try {
                                            const iframeDoc = child.contentDocument || child.contentWindow.document;
                                            if (iframeDoc && iframeDoc.body) {
                                                const iframeTree = generateAccessibilityTree(iframeDoc.body, level + 1);
                                                if (iframeTree) {
                                                    const iframeNode = {
                                                        tag: 'iframe',
                                                        role: 'document',
                                                        level: level + 1,
                                                        children: [iframeTree]
                                                    };
                                                    const iframeMd = child.getAttribute('md');
                                                    if (iframeMd) iframeNode.md = iframeMd;
                                                    node.children.push(iframeNode);
                                                }
                                            }
                                        } catch (e) {
                                            // Handle cross-origin policy restrictions
                                        }
                                    } else {
                                        const childNode = processElement(child, level + 1);
                                        if (childNode) node.children.push(childNode);
                                    }
                                }

                                if (!node.md && !node.name && node.children.length === 0 && !node.role) {
                                    return null;
                                }

                                const requiredAriaAttrs = getRequiredAriaAttributes(element);
                                for (const attr of requiredAriaAttrs) {
                                    if (!element.hasAttribute(attr)) {
                                        // Handle missing required ARIA attribute warning
                                    }
                                }

                                return node;
                            }

                            return processElement(rootElement || document.body, level || 1);
                        }

                        return generateAccessibilityTree();
                    }
            """
            accessibility_tree = await page.evaluate(js_code)
        @tool(
            agent_names=["browser_nav_agent"],
            description="""DOM Type dict Retrieval Tool, giving all interactive elements on page.
        Notes: [Elements ordered as displayed, Consider ordinal/numbered item positions, List ordinal represent z-index on page]""",
            name="get_interactive_elements",
        )
        async def get_interactive_elements() -> Annotated[str, "DOM type dict giving all interactive elements on page"]:

            start_time = time.time()
            # Create and use the PlaywrightManager
            browser_manager = self.playwright_manager()
            await browser_manager.wait_for_page_and_frames_load()
            page = await browser_manager.get_current_page()

            await browser_manager.wait_for_load_state_if_enabled(page=page)

            if page is None:  # type: ignore
                raise ValueError("No active page found. OpenURL command opens a new page.")

            extracted_data = ""
            await wait_for_non_loading_dom_state(page, 1)

            extracted_data = await do_get_accessibility_info(page, only_input_fields=False)

            # Flatten the hierarchy into a list of elements
            def flatten_elements(node: dict, parent_name: str = "", parent_title: str = "") -> list[dict]:
                elements = []
                interactive_roles = {
                    "button",
                    "link",
                    "checkbox",
                    "radio",
                    "textbox",
                    "combobox",
                    "listbox",
                    "menuitem",
                    "menuitemcheckbox",
                    "menuitemradio",
                    "option",
                    "slider",
                    "spinbutton",
                    "switch",
                    "tab",
                    "treeitem",
                }

                if "children" in node:
                    # Get current node's name and title for passing to children
                    current_name = node.get("name", parent_name)
                    current_title = node.get("title", parent_title)

                    for child in node["children"]:
                        # If child doesn't have name/title, it will use parent's values
                        if "name" not in child and current_name:
                            child["name"] = current_name
                        if "title" not in child and current_title:
                            child["title"] = current_title
                        elements.extend(flatten_elements(child, current_name, current_title))

                # Include elements with interactive roles or clickable/focusable elements
                if "md" in node and (
                        node.get("r", "").lower() in interactive_roles
                        or node.get("tag", "").lower() in {"a", "button", "input", "select", "textarea"}
                        or node.get("clickable", False)
                        or node.get("focusable", False)
                ):
                    new_node = node.copy()
                    new_node.pop("children", None)
                    elements.append(new_node)
                return elements

            flattened_data = flatten_elements(extracted_data) if isinstance(extracted_data, dict) else []

            elapsed_time = time.time() - start_time
            logger.info(f"Get DOM Command executed in {elapsed_time} seconds")

            # Count elements
            rr = 0
            if isinstance(extracted_data, (dict, list)):
                rr = len(extracted_data)


            #     if isinstance(extracted_data, dict):
            #         extracted_data = await rename_children(extracted_data)

            #     extracted_data = json.dumps(extracted_data, separators=(",", ":"))
            #     extracted_data_legend = """Key legend:
            # t: tag
            # r: role
            # c: children
            # n: name
            # tl: title
            # Dict >>
            # """
            #     extracted_data = extracted_data_legend + extracted_data
            extracted_data = json.dumps(extracted_data, separators=(",", ":"))
            return extracted_data or "Its Empty, try something else"  # type: ignore

        @tool(agent_names=[self._agent_name],
              description="Retrieve visible and relevant text content from the current page, excluding scripts, styles, and certain overlays. Useful for comprehensive page analysis.",
              name="get_page_text")
        async def get_page_text() -> Annotated[
            str, "JSON string containing the extracted page text, indicating success and the text content. Returns 'Its Empty, try something else' if no text is found."]:
            result = await self.playwright_manager.get_page_text_content()
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        # Register all the tools with the Assistant Agent's LLM and User Proxy Agent's execution map
        for func in [
            openurl, click_element,
            click, fill_field, get_page_content,
            take_screenshot, scroll_page,
            get_element_attribute, wait_for_selector,get_interactive_elements,
            get_page_text,bulk_select_option,bulk_enter_text,switch_to_tab
        ]:
            tool_metadata = func.__tool_metadata__
            self.agent.register_for_llm(name=tool_metadata["name"], description=tool_metadata["description"])(func)
            self.nav_executor.register_for_execution(name=tool_metadata["name"])(func)
            logger.info(f"Registered tool: {tool_metadata['name']}")


# --- Main Execution Workflow ---
async def run_autogen_browser_automation(user_task: str):
    """
    Orchestrates the Autogen agents and Playwright for browser automation tasks.
    """
    # Initialize PlaywrightBrowserManager
    playwright_manager = PlaywrightBrowserManager(headless=False, record_video=True)
    await playwright_manager.async_initialize()

    # --- User Proxy Agent (Executor) ---
    browser_executor = autogen.UserProxyAgent(
        name="Browser_Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=40,
        is_termination_msg=lambda x: x.get("content") and "TERMINATE" in x.get("content", "").upper(),
        function_map={},  # Tools will be dynamically registered by BrowserNavAgent
        code_execution_config={"use_docker": False}
    )

    # --- BrowserNavAgent (Assistant) ---
    browser_assistant_instance = BrowserNavAgent(
        model_config_list=config_list_openai,
        llm_config_params={"temperature": 0.1},
        nav_executor=browser_executor,
        playwright_manager=playwright_manager,  # Pass the Playwright manager instance
    )
    browser_assistant = browser_assistant_instance.get_agent_instance()

    overall_success = False
    try:
        print(f"\n--- Running Browser Automation Task: {user_task} ---")
        await browser_executor.a_initiate_chat(
            browser_assistant,
            message=user_task,
        )
        print("\n--- Browser Automation Task Complete ---")
        overall_success = True
    except Exception as e:
        logger.error(f"Error during Autogen browser automation: {e}")
        traceback.print_exc()
        overall_success = False
    finally:
        # Ensure Playwright is stopped gracefully
        await PlaywrightBrowserManager.close_all_instances()

    return {"success": overall_success, "message": "Browser automation process concluded."}


# --- Main entry point ---
if __name__ == '__main__':
    # Apply nest_asyncio at the very beginning of the script if running in environments
    # where the event loop might already be running (e.g., Jupyter, Colab, certain IDEs).
    nest_asyncio.apply()

    if not azure_openai_api_key or not azure_openai_endpoint:
        print("Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file.")
        sys.exit(1)


    browser_task_1 = "Go to 'https://www.google.com' using the `openurl` tool. Then use `get_page_text` tool to retrieve the visible text content of the page. Take a screenshot."
    browser_task_2 = "Open 'https://www.wikipedia.org' using `openurl`. Then use the `get_page_text` tool to retrieve the visible text content. Once retrieved, please summarize the text and confirm if it contains the word 'encyclopedia'. Take a screenshot."
    browser_task_3 = "Go to 'https://www.demoblaze.com/' using `openurl`. Click on 'Laptops' category using text 'Laptops'. Then click on 'MacBook air' using text 'MacBook air'. Add it to cart using text 'Add to cart'. Verify that 'Product added.' is present on the page using `get_page_text`. Take a screenshot after each step."
    browser_task_4 = "Go to 'https://www.demoblaze.com/' using `openurl`. Scroll page down 500 using 'scroll_page'. Click on 'Monitors'. Verify that 'ASUS Full HD' is present on the page. Click on 'ASUS Full HD'. and then take a screenshot."
    browser_task_5 = "Go to 'https://www.pwc.com/' using `openurl`. Click on 'Search' category using text 'Search'. Enter Text 'Audit Services' text 'searchfield'."
    browser_task_6 ="""
    1. Navigate to the PwC website successfully.
    2. User Clicks on 'Global' link by aria-label "Select a PwC territory site".
    3. User Click on 'United States'.
    4. Click on the 'Search' category.
    5. Enter Text 'Audit Services' text by text 'searchfield' 
    6. Click on the 'Audit Services' text by text 'Audit Services'.
    7. Click on the 'Search entry level opportunities' link.
    8. Switch to tab index 1
    9. Enter text "Senior Associate" in text box by name "select[name='k']" using "bulk_enter_text"
    10. Verify text "Search jobs" on the page
    11. Wait for selector by name "custom_fields.FieldofStudy"
    12. Select text "Austin" from by name "select[name='City']" by using "bulk_select_option"
    """

    browser_task_7 = """
     a user is on the URL as https://practice.expandtesting.com/jqueryui/menu#
     user scroll down to the bottom.
     the user hover on enabled menu. and then a new menu opens.
     user scroll down to the bottom.
     the user clicks on "Back to jQuery UI link" option, if nothing happens then click again
     the user should be navigated to "https://practice.expandtesting.com/jqueryui" page
     """
    browser_task_8 = """
    a user is on the URL as https://practice.expandtesting.com/redirector
    the user clicks on the "here" link, if nothing happens then click again
    the user should be able to navigated to URL as "https://practice.expandtesting.com/status-codes"
    """
    browser_task_9 = """
    a user is on the URL as https://practice.expandtesting.com/js-dialogs
    the user clicks on the "Js Prompt" button.
    then enter text "hello" in the prompt of opened dialog box and accept.
    then user scroll down to the bottom.
    the Dialogue Response field on the page should say "hello", if not found then try again.
    """
    browser_task_10 = """
    a user is on the URL as https://practice.expandtesting.com/windows
    the user clicks on the "Click Here" link, if nothing happens then click again
    the user should be navigated to a new window with the text as "Example of a new window page for Automation Testing Practice"
    """
    browser_task_11 = """
    a user is on the URL as https://wrangler.in
    the user clicks on 'Search' image
    the user enters text as "Rainbow sweater" in search text box
    the user filters Turtle Neck within Neck filter section.
    only one product should be displayed as the result, regardless of type of product.
    """

    browser_task_12 = """
        a user is on the URL as https://practice.expandtesting.com/windows
        the user clicks on the "Demos" link, if nothing happens then click again
         the user clicks on the "Examples" link
         the user clicks on "dismiss" link by id "dismiss-button" in iframe by using "click_element"
         the user clicks on "Web inputs" link
        """

    browser_task_13 = """
    Navigate to the PwC website successfully. 
    Click on territory link by aria-label "Select a PwC territory site" and select "United States"
    Click on Search category and Enter "Audit Services" in the search text box and submit the search 
    """

    browser_task_14 ="""
    a user is on the URL as https://www.saucedemo.com/inventory.html
    Enter username as "standard_user" and password as "secret_sauce"
    Click on Login button and verify text as "Sauce Labs Backpack" in the page
    Click on the "Open Menu" button
    Click on "About" Text
    Click on "Pricing" Text
    Verify text "Platform for Test"
    """
    browser_task_15 = """
        a user is on the URL as https://www.saucedemo.com/inventory.html using `openurl`
        Enter by data-test "username" as "standard_user" 
        Enter by data-test "password" as "secret_sauce"
        Click on Login button and verify text as "Sauce Labs Backpack" in the page
        Click on the "Sauce Labs Backpack" link
        Click on "Add to cart" Button
        Click on "shopping_cart_container" by id
        Click on "Checkout" button
        Fill Random data in First name, Last name and Zip
        Click on "continue" button
        Click on "Finish" button
        Verify text "Thank you for your order!"
        Take a screenshot after each step.
        """
    browser_task_16="""
    Open @https://www.saucedemo.com/ 
    Login with Username and password
    Click On Login button
    Select option "Price (low to high)" from "product-sort-container" by using "bulk_select_option"
    Click Product "Sauce Labs Backpack" link and Add product Into the cart
    Open the cart 
    Click on Checkout button
    Fill Random data in First name , Last name and Zip
    Click on continue button
    Click on finish button
    Verify message "Thank you for your order!"
    """

    browser_task_17 = """
        Open @https://www.demoblaze.com/index.html
        Click On Contact text
        Fill random data Contact Email, Contact Name, Message
        Click on button "Send message" text
        """

    browser_task_18 = """
          Open @https://www.demoblaze.com/index.html
          Click On link Monitors text
          Click on link "Apple monitor 24" text
          click on link "Add to cart" text
          click on link "Cart" text
          Click on button "Place Order" text
          Fill random data name , country , city
          """

    browser_task_19 = """
    Open @https://practice.expandtesting.com/shadowdom
    Click on button "Here's a basic button example." text
    Click on button "This button is inside a Shadow DOM." text
    """


    browser_task_20 = """
    Open @https://practice.expandtesting.com/dropdown
    Scroll down to the bottom
    Select option "Option 1" from "dropdown" by using "bulk_select_option"
    """

    browser_task_21 = """
    open @https://support.xbox.com/en-US/
    Fill "How do I reset my console?" text in textbox by id "SearchTerms"
    Click on button by id "SearchBox_GoButton"
    Verify "Search results" text on the page
    Click on link "Xbox status" text
    Verify "Xbox status" text on the page
    Click on button "Help topics" text
    Click on link by aria-label "Account & profile" text
    Verify "Account & profile" text on the page
    Click on button "Most popular" text
    Click on link "Sign in to Xbox" text
    Verify "Sign in to Xbox" text on the page
    """

    browser_task_22 = """
        open @https://xboxdesignlab.xbox.com/
        Click on button "Accept" text
        Click on button "Shop" text
        Click on link by aria-label "Xbox Wireless Controller"
        Scroll down
        Verify "Xbox Wireless Controller" text on the page
        """

    task_to_run = browser_task_14

    asyncio.run(run_autogen_browser_automation(task_to_run))

    print("\nScript Finished.")
