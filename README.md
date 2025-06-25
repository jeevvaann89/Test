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
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable,Annotated

import httpx
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import BrowserContext, BrowserType, ElementHandle
from playwright.async_api import Error as PlaywrightError  # for exception handling
from playwright.async_api import Page, Playwright
from playwright.async_api import async_playwright as playwright
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

# --- Simplified / Placeholder imports from testzeus_hercules ---
# In a real scenario, you'd properly import or redefine these.
# For this example, I'm providing minimal implementations.

class DummyConfig:
    def get_global_conf(self):
        return self

    def should_record_video(self): return False

    def get_proof_path(self, test_id): return f"./proofs/{test_id}"

    def get_browser_type(self): return "chromium"

    def get_browser_channel(self): return None

    def get_browser_path(self): return None

    def get_browser_version(self): return None

    def should_run_headless(self): return True

    def get_cdp_config(self): return None

    def should_take_screenshots(self): return True

    def should_take_bounding_box_screenshots(self): return False

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


class DummyLogger:
    def info(self, msg, *args): print(f"INFO: {msg}" % args)

    def warning(self, msg, *args): print(f"WARNING: {msg}" % args)

    def error(self, msg, *args): print(f"ERROR: {msg}" % args)

    def debug(self, msg, *args): print(f"DEBUG: {msg}" % args)


get_global_conf = DummyConfig().get_global_conf
logger = DummyLogger()  # Using dummy logger

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

        self._take_screenshots = take_screenshots if take_screenshots is not None else get_global_conf().should_take_screenshots()
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
        os.makedirs(os.path.dirname(self.request_response_log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.console_log_file), exist_ok=True)

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

            _browser = await browser_type_launcher.connect_over_cdp(endpoint_url, timeout=120000)

            context_options = self._build_emulation_context_options()
            if self._record_video:
                context_options["record_video_dir"] = self._video_dir
                logger.info("Recording video in CDP mode.")

            self._browser_context = None
            if _browser.contexts and self.cdp_reuse_tabs:
                logger.info("Reusing existing browser context.")
                self._browser_context = _browser.contexts[0]
                # Apply new context options to existing context if possible/needed
                # Playwright does not directly support re-applying context options after creation.
                # If these need to change, a new context might be required or emulation applied per page.

            if not self._browser_context:
                logger.info("Creating new browser context.")
                self._browser_context = await _browser.new_context(**context_options)

            # Page selection logic
            pages = self._browser_context.pages
            if pages:
                # Prioritize a non-blank page, otherwise use the first available
                self._current_page = next((p for p in pages if p.url not in ["about:blank", "chrome://newtab/"]),
                                          pages[0])
                logger.info(f"Reusing existing page with URL: {self._current_page.url}")
                try:
                    await self._current_page.bring_to_front()
                    logger.info("Brought reused page to front")
                except Exception as e:
                    logger.warning(f"Failed to bring page to front: {e}")
            else:
                logger.info("No existing pages found. Creating new page.")
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

    async def get_text_from_selector(self, selector: str) -> Dict[str, Any]:
        """Retrieves the text content of an element identified by a CSS selector."""
        page = await self.get_current_page()
        try:
            text_content = await page.locator(selector).text_content(timeout=10000)
            if text_content is None:
                return {"success": False, "error": f"Element with selector '{selector}' found but has no text content."}
            return {"success": True, "text": text_content.strip()}
        except PlaywrightError as e:
            logger.error(f"Failed to get text from selector '{selector}': {e}")
            return {"success": False, "error": f"Failed to get text from selector '{selector}': {e}"}

    async def take_screenshot(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Takes a screenshot of the current page and saves it to a specified path.
        Returns the path to the screenshot."""
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

    async def wait_for_load_state_if_enabled(self, page: Page) -> None:
        """
        A placeholder for waiting for the page to load based on a configuration.
        In the original context, this might involve more complex logic.
        For this standalone version, it will simply wait for 'load' state.
        """
        try:
            await page.wait_for_load_state('load', timeout=60000)  # Wait for 'load' event
            logger.info(f"Page load state 'load' reached for {page.url}")
        except PlaywrightError as e:
            logger.warning(f"Failed to wait for page load state: {e}")


# --- Helper functions for get_page_text ---
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
    """
    Executes JavaScript on the page to extract filtered text content,
    excluding scripts, styles, and specified overlay elements.
    It also collects alt texts from images.
    """
    text_content = await page.evaluate(
        """
        () => {
        const selectorsToFilter = ['#hercules-overlay']; // Example of elements to hide/filter
        const originalStyles = [];

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

        function getTextSkippingScriptsStyles(root) {
            if (!root) return '';

            let textContent = '';
            const walker = createSkippingTreeWalker(root);

            while (walker.nextNode()) {
            const node = walker.currentNode;

            if (node.nodeType === Node.TEXT_NODE) {
                textContent += node.nodeValue;
            }
            else if (node.shadowRoot) {
                textContent += getTextSkippingScriptsStyles(node.shadowRoot);
            }
            }
            return textContent;
        }

        function getTextFromIframes(root) {
            if (!root) return '';
            let iframeText = '';

            const iframes = root.querySelectorAll('iframe');
            iframes.forEach(iframe => {
            try {
                const iframeDoc = iframe.contentDocument;
                if (iframeDoc) {
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

        function getAltTextsFromShadowDOM(root) {
            if (!root) return [];
            let altTexts = Array.from(root.querySelectorAll('img')).map(img => img.alt);

            const allNodes = root.querySelectorAll('*');
            allNodes.forEach(node => {
            if (node.shadowRoot) {
                altTexts = altTexts.concat(getAltTextsFromShadowDOM(node.shadowRoot));
            }
            });
            return altTexts.filter(alt => alt && alt.trim() !== ''); // Filter out empty alt texts
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
            return iframeAltTexts.filter(alt => alt && alt.trim() !== ''); // Filter out empty alt texts
        }

        // 1) Hide overlays
        selectorsToFilter.forEach(selector => {
            processElementsInShadowDOM(document, selector);
            processElementsInIframes(document, selector);
        });

        // 2) Collect text from the main document and its documentElement
        let mainTextContent = getTextSkippingScriptsStyles(document.body);
        mainTextContent += getTextSkippingScriptsStyles(document.documentElement);

        // 3) Collect text from iframes
        let iframeTextContent = getTextFromIframes(document);

        // 4) Collect alt texts
        let altTexts = getAltTextsFromShadowDOM(document);
        altTexts = altTexts.concat(getAltTextsFromIframes(document));

        // Combine all text
        let combinedText = mainTextContent + '\n' + iframeTextContent;
        if (altTexts.length > 0) {
            combinedText += '\nOther Alt Texts in the page: ' + altTexts.join(' ');
        }

        // 5) Restore hidden overlays
        originalStyles.forEach(entry => {
            entry.element.style.visibility = entry.originalStyle;
        });

        return combinedText;
        }
    """
    )
    return text_content


# Autogen imports
import autogen
from textwrap import dedent
from string import Template
from dotenv import load_dotenv

load_dotenv()  # Load environment variables for OpenAI API

# --- LLM Configuration ---
azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

config_list_openai = [
    {
        "model": "gpt-35-turbo",  # or "gpt-4", "gpt-4-vision-preview"
        "api_key": azure_openai_api_key,
        "base_url": azure_openai_endpoint,
        "api_type": "azure",
        "api_version": "2024-02-15",  # Use a recent API version compatible with function calling
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
- Navigate webpages and handle URL transitions.
- Interact with web elements (buttons, inputs, dropdowns, etc.) using selectors or visible text.
- Retrieve Text on the current page.
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
4. ALWAYS use precise CSS selectors or visible text for element identification.
5. If you cannot identify an element, you can attempt to get the full page content (`get_page_text` tool) to understand the page structure better, or ask for clarification.

### EXECUTION PROCESS
6. ALWAYS analyze the page (or its content) and the task BEFORE taking any action.
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

    def __init__(self, model_config_list: list, llm_config_params: dict[str, Any],
                 nav_executor: autogen.UserProxyAgent, playwright_manager: PlaywrightBrowserManager,
                 system_prompt: str | None = None, agent_name: str | None = None, agent_prompt: str | None = None):

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
            max_consecutive_auto_reply=10,
        )

        self.register_tools()

    def get_agent_instance(self) -> autogen.ConversableAgent:
        return self.agent

    def register_tools(self) -> None:
        """
        Registers Playwright-based browser tools with the Autogen agents.
        """
        logger.info(f"Registering Playwright tools for {self._agent_name}...")

        # Define the tools. Each tool will call a method on the playwright_manager instance.
        # The `tool` decorator adds metadata for Autogen's LLM.

        @tool(agent_names=[self._agent_name], name="navigate_to_url",
              description="Navigates the browser to the specified URL.")
        async def navigate_to_url(url: Annotated[str, "The URL to navigate to."]) -> Annotated[
            str, "JSON string indicating success, message, and current URL."]:
            result = await self.playwright_manager.navigate_to_url(url)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

        @tool(agent_names=[self._agent_name], name="click_element",
              description="Clicks on an element identified by a CSS selector or its exact visible text.")
        async def click_element(selector: Annotated[
            Optional[str], "CSS selector of the element to click (e.g., 'button#submit')."] = None, text: Annotated[
            Optional[str], "Exact visible text of the element to click (e.g., 'Login button')."] = None) -> Annotated[
            str, "JSON string indicating success and message."]:
            result = await self.playwright_manager.click_element(selector=selector, text=text)
            return json.dumps(result, separators=(",", ":")).replace('"', "'")

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

        # --- NEW TOOL: get_page_text ---
        @tool(agent_names=[self._agent_name],
              description="Retrieve visible and relevant text content from the current page, excluding scripts, styles, and certain overlays. Useful for comprehensive page analysis.",
              name="get_page_text")
        async def get_page_text() -> Annotated[
            str, "JSON string containing the extracted page text, indicating success and the text content. Returns 'Its Empty, try something else' if no text is found."]:
            logger.info(f"Executing get_page_text")
            start_time = time.time()

            page = await self.playwright_manager.get_current_page()

            # Using PlaywrightManager's new wait_for_load_state_if_enabled
            await self.playwright_manager.wait_for_load_state_if_enabled(page=page)

            if page is None:
                return json.dumps({"success": False, "error": "No active page found to retrieve text."},
                                  separators=(",", ":")).replace('"', "'")

            extracted_data = ""
            # Simplified wait for non-loading state - using a short sleep after load
            await asyncio.sleep(1)  # Give a moment for page to render after load state

            try:
                logger.debug("Fetching DOM for text_only")
                text_content = await get_filtered_text_content(page)
                cleaned_text = clean_text(text_content)

                # Save to file (using DummyConfig's get_source_log_folder_path)
                log_file_path = os.path.join(get_global_conf().get_source_log_folder_path(), "text_only_dom.txt")
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                logger.info(f"Page text saved to {log_file_path}")

                extracted_data = cleaned_text
            except Exception as e:
                logger.error(f"Error extracting filtered text content: {e}")
                extracted_data = f"Error extracting page text: {e}"
                return json.dumps({"success": False, "error": extracted_data}, separators=(",", ":")).replace('"', "'")

            elapsed_time = time.time() - start_time
            logger.info(f"Get DOM Command executed in {elapsed_time:.2f} seconds")

            final_response = extracted_data if extracted_data else "Its Empty, try something else"
            return json.dumps({"success": True, "text_content": final_response}, separators=(",", ":")).replace('"',
                                                                                                                "'")

        # Register the new tool
        for func in [get_page_text]:  # Add get_page_text to the list of functions to register
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
        max_consecutive_auto_reply=10,
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
    # If this is a standalone script executed directly, it's not strictly necessary but harmless.
    # nest_asyncio.apply() # Moved to top

    if not azure_openai_api_key or not azure_openai_endpoint:
        print("Please set AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in your .env file.")
        sys.exit(1)

    # Example Browser Automation Tasks
    # Note: Use get_page_text when you want to read general content on a page.
    # For verification, the LLM might call get_page_text and then perform text processing.
    browser_task_1 = "Go to 'https://www.google.com'"
    browser_task_2 = "Open 'https://www.wikipedia.org'. Then use the `get_page_text` tool to retrieve the visible text content. Once retrieved, please summarize the text and confirm if it contains the word 'encyclopedia'."
    browser_task_3 = "Go to 'https://www.demoblaze.com/'. Click on 'Laptops' category using text 'Laptops'. Then use the `get_page_text` tool to retrieve the page content and confirm if the text 'MacBook air' is present in the retrieved text."

    task_to_run = browser_task_1  # Choose which task to run

    # Run the browser automation task
    asyncio.run(run_autogen_browser_automation(task_to_run))

    print("\nScript Finished.")
