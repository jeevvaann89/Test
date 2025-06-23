import asyncio
import base64
import json
import time
import logging
import nest_asyncio
nest_asyncio.apply()
from typing import Annotated, Any, Dict, Optional, Tuple
import os , sys # This import is already present, ensuring os.getenv works.
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()
import httpx  # Make sure to install httpx: pip install httpx
import autogen  # Make sure to install autogen: pip install pyautogen[openai]
azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
# --- Setup Basic Logging (replacing testzeus_hercules loggers) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Dummy logger for file_logger equivalent for this example
def dummy_file_logger(message: str) -> None:

    print(f"[API LOG]: {message}")

def tool(agent_names: list, name: str, description: str):
    def decorator(func):
        func.__tool_metadata__ = {  # Store metadata for potential external use or inspection
            "agent_names": agent_names,
            "name": name,
            "description": description
        }
        return func
    return decorator


async def log_request(request: httpx.Request) -> None:
    """
    Log details of the outgoing HTTP request.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_data = {
        "request_data": {
            "timestamp": timestamp,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": (
                request.content.decode("utf-8", errors="ignore")
                if request.content
                else None
            ),
        }
    }
    dummy_file_logger(json.dumps(log_data))


async def log_response(response: httpx.Response) -> None:
    """
    Log details of the incoming HTTP response.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    try:
        # Use response.read() instead of response.aread() for blocking read after await response is available
        # or capture body during request if using stream=True
        # For simple cases, response.text or response.json() will consume the body.
        # httpx handles body reading typically. This is a simplified log.
        body = await response.aread()  # await response.aread() is correct for httpx response hooks
        body = body.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception(f"Error reading response body for logging: {e}")
        body = f"Failed to read response: {e}"
    log_data = {
        "response_data": {
            "timestamp": timestamp,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }
    }
    dummy_file_logger(json.dumps(log_data))


def determine_status_type(status_code: int) -> str:
    """
    Categorize the HTTP status code.
    """
    if 200 <= status_code < 300:
        return "success"
    elif 300 <= status_code < 400:
        return "redirect"
    elif 400 <= status_code < 500:
        return "client_error"
    elif 500 <= status_code < 600:
        return "server_error"
    return "unknown"


async def handle_error_response(e: httpx.HTTPStatusError) -> dict:
    """ Extract error details from an HTTPStatusError. """
    try:
        if 'application/json' in e.response.headers.get('Content-Type', ''):
            error_detail = e.response.json()
        else:
            error_detail = e.response.text or "No details"
    except Exception as ex:
        logger.exception(f"Error extracting error details: {ex}")
        error_detail = e.response.text or "No details"
    return {
        "error": str(e),
        "error_detail": error_detail,
        "status_code": e.response.status_code,
        "status_type": determine_status_type(e.response.status_code),
    }


# ------------------------------------------------------------------------------
# Core Request Helper (Adapted from your provided code)
# ------------------------------------------------------------------------------


async def _send_request(
        method: str,
        url: str,
        *,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        body_mode: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, float]:
    """
    Send an HTTP request using the given method and parameters.

    The 'body_mode' parameter specifies how to process the request body:
      - "multipart": Encodes a dict as multipart/form-data.
      - "urlencoded": Encodes a dict as application/x-www-form-urlencoded.
      - "raw": Sends the body as a raw string (caller must set Content-Type).
      - "binary": Sends the body as raw bytes (defaults to application/octet-stream).
      - "json": Encodes the body as JSON.
      - None: No body is sent.
    """
    query_params = query_params or {}
    headers = headers.copy() if headers else {}
    req_kwargs = {"params": query_params}

    if body_mode == "multipart" and body:
        form = httpx.FormData()
        for key, value in body.items():
            form.add_field(key, value)
        req_kwargs["data"] = form

    elif body_mode == "urlencoded" and body:
        headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        req_kwargs["data"] = body

    elif body_mode == "raw" and body:
        req_kwargs["content"] = body

    elif body_mode == "binary" and body:
        headers.setdefault("Content-Type", "application/octet-stream")
        req_kwargs["content"] = body

    elif body_mode == "json" and body:
        headers.setdefault("Content-Type", "application/json")
        req_kwargs["json"] = body

    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(
                event_hooks={"request": [log_request], "response": [log_response]},
                timeout=httpx.Timeout(5.0),
        ) as client:
            response = await client.request(method, url, headers=headers, **req_kwargs)
            response.raise_for_status()  # Raises an exception for 4xx/5xx responses
            duration = time.perf_counter() - start_time

            try:
                parsed_body = response.json()
            except Exception as e:
                logger.exception(f"Error parsing response body: {e}")
                parsed_body = response.text or ""
            result = {
                "status_code": response.status_code,
                "status_type": determine_status_type(response.status_code),
                "body": parsed_body,
            }
            # Minify the JSON response and replace double quotes with single quotes.
            result_str = json.dumps(result, separators=(",", ":")).replace('"', "'")
            return result_str, duration

    except httpx.HTTPStatusError as e:
        duration = time.perf_counter() - start_time
        logger.error(f"HTTP error: {e}")
        error_data = await handle_error_response(e)
        return json.dumps(error_data, separators=(",", ":")).replace('"', "'"), duration

    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(f"Unexpected error: {e}")
        error_data = {"error": str(e), "status_code": None, "status_type": "failure"}
        return json.dumps(error_data, separators=(",", ":")).replace('"', "'"), duration


# ------------------------------------------------------------------------------
# Generic HTTP API Function Tool (Adapted from your provided code)
# ------------------------------------------------------------------------------


@tool(
    agent_names=["api_nav_agent"],  # This name might be used by a custom Autogen agent setup
    name="generic_http_api",
    description=(
            "Generic HTTP API call that supports any combination of HTTP method, "
            "authentication, query parameters, and request body encoding. "
            "Parameters:\n"
            "  - method: HTTP method (GET, POST, PUT, PATCH, DELETE, etc.).\n"
            "  - url: The API endpoint URL.\n"
            "  - auth_type: Authentication type. Options: basic, jwt, form_login, bearer, api_key. (Optional)\n"
            "  - auth_value: For 'basic', pass [username, password]; for others, a string. (Optional)\n"
            "  - query_params: URL query parameters (dict).\n"
            "  - body: Request payload.\n"
            "  - body_mode: How to encode the body. Options: multipart, urlencoded, raw, binary, json. (Optional)\n"
            "  - headers: Additional HTTP headers (dict).\n"
            "This single function can generate any combination supported by _send_request."
    ),
)
async def generic_http_api(
        method: Annotated[str, "HTTP method (e.g. GET, POST, PUT, PATCH, DELETE, etc.)."],
        url: Annotated[str, "The API endpoint URL."],
        auth_type: Annotated[
            Optional[str],  # Changed to Optional[str] as per your usage
            "Authentication type. Options: basic, jwt, form_login, bearer, api_key. (Optional)",
        ] = None,
        auth_value: Annotated[
            Any,
            "Authentication value: for 'basic' provide [username, password]; for others, provide a string. (Optional)",
        ] = None,
        query_params: Annotated[Dict[str, Any], "URL query parameters."] = {},
        body: Annotated[Any, "Request payload."] = None,
        body_mode: Annotated[
            Optional[str],  # Changed to Optional[str]
            "Body mode: multipart, urlencoded, raw, binary, or json. (Optional)",
        ] = None,
        headers: Annotated[Dict[str, str], "Additional HTTP headers."] = {},
) -> Annotated[
    Tuple[str, float], "Minified JSON response and call duration (in seconds)."
]:
    # Set authentication headers based on auth_type.
    if auth_type:
        auth_type = auth_type.lower()
        if (
                auth_type == "basic"
                and isinstance(auth_value, list)
                and len(auth_value) == 2
        ):
            creds = f"{auth_value[0]}:{auth_value[1]}"
            token = base64.b64encode(creds.encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        elif auth_type == "jwt":
            headers["Authorization"] = f"JWT {auth_value}"
        elif auth_type == "form_login":
            headers["X-Form-Login"] = auth_value
        elif auth_type == "bearer":
            headers["Authorization"] = f"Bearer {auth_value}"
        elif auth_type == "api_key":
            headers["x-api-key"] = auth_value

    return await _send_request(
        method,
        url,
        query_params=query_params,
        body=body,
        body_mode=body_mode,
        headers=headers,
    )


# ------------------------------------------------------------------------------
# Autogen Example for API Automation
# ------------------------------------------------------------------------------

async def run_autogen_api_automation():

    config_list_openai = [
        {
            "model": "gpt-35-turbo",
            "api_key": azure_openai_api_key,
            "base_url": azure_openai_endpoint,
            "api_type": "azure",
            "api_version": "2024-12-01-preview",
        }
    ]



    user_proxy = autogen.UserProxyAgent(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,  # Max number of consecutive auto-replies from agents
        is_termination_msg=lambda x: x.get("content") and "TERMINATE" in x.get("content", "").upper(),
        # Enable tool use by explicitly defining a function_map
        function_map={"generic_http_api": generic_http_api}
    )

    assistant = autogen.AssistantAgent(
        name="API_Assistant",
        llm_config={
            "config_list": config_list_openai,
            # Adjust temperature for creativity/determinism
            "temperature": 0.1,
        },
        system_message="""
        ### API Navigation Agent

You are an API Navigation Agent responsible for executing API calls and handling their responses. Your primary focus is on performing function/tool calls as required by the task. Follow the guidelines below meticulously to ensure every action is executed correctly and sequentially.

---

#### 1. Core Functions

- **Execute API Calls:** Initiate the appropriate API functions.
- **Handle Responses:** Process and interpret responses carefully.
- **Retrieve Results:** Extract data from responses.
- **Build Payloads:** Construct payloads using actual test data and results.
- **Summarize Responses:** Document status codes, execution time, and any relevant details.
- focus on task in hand, use extra information cautiously, don't deviate from the task.
---

#### 2. Core Rules

1. **Task Specificity:**  
   - Process only API-related tasks as defined in the "TASK FOR HELPER."
   - If any clarification is needed, request it before proceeding.

2. **Sequential Execution:**  
   - Execute only one function at a time.
   - Wait for the response before making the next function call.

3. **Strict Data Formats:**  
   - Follow the exact data formats provided.
   - Build payloads using the actual results and test data.

4. **Accurate Parameter Passing during function calls:**  
   - Always include all required parameters in every function/tool call.
   - Do not omit parameters; if unsure, PASS THE DEFAULT values.

5. **Validation & Error Handling:**  
   - If a function call fails due to Pydantic validation, correct the issue and re-trigger the call.
   - Focus on status codes and execution time, and document these in your response summary.

6. **Result Verification:**  
   - After each function call, verify that the result is sufficient before proceeding to the next call.
   - Do not simply count function calls; ensure each result is complete and correct.

7. **Critical Actions:**  
   - For actions like login, logout, or registration, pass all required and proper values.
   - Avoid modifications to test data.
   
8. **Course Correct on Bad function calls:**
   - If a function call fails, course correct and call the function again with the correct parameters, you get only one chance to call the function again.
   - Do not repeat the function call without making any changes to the parameters.

---

#### 3. Data Usage

- **data_only:** For text extraction.
- **all_fields:** For nested data extraction.
- **Test Data:** Always use the exact provided test values without modifications.
- **USEFUL INFORMATION:** Refer to this section for additional test data and dependency details.

---

#### 4. Response Formats

Use the following standardized response formats:

- **Success:**
previous_step: [previous step assigned summary]
current_output: [DETAILED EXPANDED COMPLETE LOSS LESS output]
##FLAG::SAVE_IN_MEM##
##TERMINATE TASK##

- **Information Request:**
previous_step: [previous step assigned summary]
current_output: [DETAILED EXPANDED COMPLETE LOSS LESS output]
##TERMINATE TASK##

- **Error:**
previous_step: [previous step assigned summary]
current_output: [Issue description]
[Required information]
##TERMINATE TASK##

---

#### 5. Error Handling Rules

- **Stop After Repeated Failures:**  
- Do not continue retrying after multiple failures.
- Document each error precisely.

- **No Unnecessary Retries:**  
- Only reattempt a function call if it fails due to a known issue (e.g., Pydantic validation error).

Available Test Data: $basic_test_information
        
        You are an expert API automation assistant. 
        Your goal is to fulfill user requests by making HTTP API calls using the 'generic_http_api' tool.
        You should analyze the user's request, determine the necessary API method, URL, parameters, and body.
        Always output the final result in a clear, concise manner.
        If a task is completed or cannot be done, say 'TERMINATE'.
        """,
    )

    # --- 3. Register the Tool with the Agents ---
    # Tools need to be registered with the agent that decides to call them (AssistantAgent)
    # and the agent that executes them (UserProxyAgent).

    # For the AssistantAgent to *know* about the tool and how to call it:
    assistant.register_for_llm(name="generic_http_api", description=generic_http_api.__tool_metadata__["description"])(
        generic_http_api)

    # The UserProxyAgent is already set up to execute functions in its function_map.
    # No explicit `user_proxy.register_for_execution` needed if function_map is used directly.
    # If the tool was defined without __tool_metadata__ or if Autogen needs explicit registration beyond function_map,
    # you might use: user_proxy.register_for_execution(name="generic_http_api")(generic_http_api)

    logger.info("AutoGen agents and generic_http_api tool configured.")

    # --- 4. Define the API Automation Task ---
    api_task_1 = """
    Please make a GET request to 'https://jsonplaceholder.typicode.com/posts/1'. 
    Once you get the response, tell me the title of the post.
    """

    api_task_2 = """
    Perform a POST request to 'https://jsonplaceholder.typicode.com/posts'.
    The request body should be JSON with 'title': 'foo', 'body': 'bar', 'userId': 1.
    After the successful creation, respond with the 'id' of the newly created post.
    """

    api_task_3 = """
    Try to make a GET request to a non-existent URL: 'https://example.com/non-existent-api-path'. 
    Report the error details.
    """

    print("\n--- Running API Automation Task 1 (GET Request) ---")
    user_proxy.initiate_chat(
        assistant,
        message=api_task_1,
    )
    print("\n" + "=" * 50 + "\n")  # Separator

    print("\n--- Running API Automation Task 2 (POST Request) ---")
    user_proxy.initiate_chat(
        assistant,
        message=api_task_2,
    )
    print("\n" + "=" * 50 + "\n")  # Separator

    print("\n--- Running API Automation Task 3 (Error Handling) ---")
    user_proxy.initiate_chat(
        assistant,
        message=api_task_3,
    )
    print("\n" + "=" * 50 + "\n")  # Separator


# --- Main execution block ---
if __name__ == '__main__':
    asyncio.run(run_autogen_api_automation())
