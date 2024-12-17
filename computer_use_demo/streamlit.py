"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import os
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import PosixPath
from typing import cast

import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = load_from_storage("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = None
    if "current_patient_id" not in st.session_state:
        st.session_state.current_patient_id = None
    if "provider" not in st.session_state:
        st.session_state.provider = (
            os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        )
    if "provider_radio" not in st.session_state:
        st.session_state.provider_radio = st.session_state.provider
    if "model" not in st.session_state:
        _reset_model()
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False


def _reset_model():
    st.session_state.model = PROVIDER_TO_DEFAULT_MODEL_NAME[
        cast(APIProvider, st.session_state.provider)
    ]


async def main():
    """Render loop for streamlit"""
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("evaLabs Navigator™")

    # if not os.getenv("HIDE_WARNING", False):
    #     st.warning(WARNING_TEXT)

    with st.sidebar:

        def _reset_api_provider():
            if st.session_state.provider_radio != st.session_state.provider:
                _reset_model()
                st.session_state.provider = st.session_state.provider_radio
                st.session_state.auth_validated = False

        provider_options = [option.value for option in APIProvider]
        st.radio(
            "API Provider",
            options=provider_options,
            key="provider_radio",
            format_func=lambda x: x.title(),
            on_change=_reset_api_provider,
        )

        st.text_input("Model", key="model")

        if st.session_state.provider == APIProvider.ANTHROPIC:
            st.text_input(
                "Anthropic API Key",
                type="password",
                key="api_key",
                on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
            )

        st.number_input(
            "Only send N most recent images",
            min_value=0,
            key="only_n_most_recent_images",
            help="To decrease the total tokens sent, remove older screenshots from the conversation",
        )
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Additional instructions to append to the system prompt. see computer_use_demo/loop.py for the base system prompt.",
            on_change=lambda: save_to_storage(
                "system_prompt", st.session_state.custom_system_prompt
            ),
        )
        st.checkbox("Hide screenshots", key="hide_images")

        if st.button("Reset", type="primary"):
            with st.spinner("Resetting..."):
                st.session_state.clear()
                setup_state()

                subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                await asyncio.sleep(1)
                subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

    if not st.session_state.auth_validated:
        if auth_error := validate_auth(
            st.session_state.provider, st.session_state.api_key
        ):
            st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")
            return
        else:
            st.session_state.auth_validated = True

    kis, chat, http_logs = st.tabs(["KIS", "Chat", "HTTP Exchange Logs"])
    
    with kis:
        # Add form for Patient ID
        with st.form("patient_id_form"):
            # Patient ID input (alphanumerical)
            patient_id = st.text_input(
                "Patient ID",
                key="patient_id",
                help="Enter alphanumerical Patient ID"
            )
            
            # Submit button for Patient ID
            lookup_submitted = st.form_submit_button("Look up Patient")

            if lookup_submitted and patient_id:
                # Create the specific LibreOffice lookup query
                lookup_query = f"Open the file Patient Data 1 in LibreOffice Calc, switch to the tab 'Patients', and search for a specific Patient ID: {patient_id}. Respond with this PatientID's first and last name."
                
                # Safely create new message
                new_message = {
                    "role": Sender.USER,
                    "content": [
                        BetaTextBlockParam(type="text", text=lookup_query),
                    ],
                }
                
                # Clear and initialize messages list
                st.session_state.messages = [new_message]
                # Initialize tools dictionary if it doesn't exist
                if "tools" not in st.session_state:
                    st.session_state.tools = {}
                # Store the current patient ID
                st.session_state.current_patient_id = patient_id
                # Reset visit-related session states
                if 'visits_loaded' in st.session_state:
                    del st.session_state.visits_loaded

        # Get the most recent response if available
        response_text = "No patient information available"
        if st.session_state.messages:
            assistant_messages = [
                msg for msg in st.session_state.messages 
                if msg["role"] == Sender.BOT
            ]
            if assistant_messages:
                latest_response = assistant_messages[-1]
                if isinstance(latest_response["content"], list):
                    response_text = next((
                        block["text"] 
                        for block in latest_response["content"] 
                        if block["type"] == "text"
                    ), "No patient information found")
                else:
                    response_text = latest_response["content"]

        # Always show the patient information field
        st.text_area(
            "Patient Information",
            value=response_text,
            disabled=True,
            key="patient_info_display",
            height=300
        )

        # Show confirmation button or visit selection based on state
        if not st.session_state.get('visits_loaded', False):
            # Show confirmation button if we have patient info
            if response_text != "No patient information available":
                if st.button("Confirm Patient Identity"):
                    # Create the visits lookup query with explicit format instructions
                    visits_query = (
                        f"Look for all Visits of this specific patient with the ID: {st.session_state.current_patient_id}. "
                        "List each visit in the following format:\n"
                        "Date: MM/DD/YY - Documentation: [documentation number]\n\n"
                        "Example format:\n"
                        "Date: 12/15/24 - Documentation: documentationtest1 45641\n"
                        "Please list all visits in this exact format."
                    )
                    
                    # Add the query to messages
                    st.session_state.messages.append({
                        "role": Sender.USER,
                        "content": [
                            BetaTextBlockParam(type="text", text=visits_query),
                        ],
                    })
                    st.session_state.visits_loaded = True
                    st.rerun()
        else:
            # Parse visits from response text with the standardized format
            import re
            visits = []
            for line in response_text.split('\n'):
                if match := re.match(r'Date:\s*(\d{2}/\d{2}/\d{2})\s*-\s*Documentation:\s*(.+)', line):
                    date, doc = match.groups()
                    visits.append((date.strip(), doc.strip()))
            
            # Date picker for visits
            if visits:
                st.subheader("Visit Selection")
                selected_date = st.selectbox(
                    "Select Visit Date",
                    options=[date for date, _ in visits],
                    key="selected_visit_date"
                )
                
                # Find documentation for selected date
                selected_doc = next((doc for date, doc in visits if date == selected_date), "")
                
                # Editable documentation field
                st.subheader("Visit Documentation")
                new_documentation = st.text_area(
                    "Edit Documentation",
                    value=selected_doc,
                    height=300,
                    key="documentation_editor"
                )
                
                # Save button for documentation changes
                if st.button("Save Documentation Changes"):
                    save_query = f"Update the documentation for the visit on {selected_date} for patient {st.session_state.current_patient_id} with the following text:\n{new_documentation}"
                    st.session_state.messages.append({
                        "role": Sender.USER,
                        "content": [
                            BetaTextBlockParam(type="text", text=save_query),
                        ],
                    })
                    st.rerun()
            # else:
            #     st.warning("No visits found in the response. Please try again.")

        # If we have a user message to respond to, run the sampling loop
        if st.session_state.messages and st.session_state.messages[-1]["role"] == Sender.USER:
            try:
                with st.spinner('Processing...'):
                    with track_sampling_loop():
                        def dummy_tool_callback(tool_output, tool_id, tool_state):
                            tool_state[tool_id] = tool_output
                            
                        st.session_state.messages = await sampling_loop(
                            system_prompt_suffix=st.session_state.custom_system_prompt,
                            model=st.session_state.model,
                            provider=st.session_state.provider,
                            messages=st.session_state.messages,
                            output_callback=lambda *args, **kwargs: None,
                            tool_output_callback=partial(dummy_tool_callback, tool_state=st.session_state.tools),
                            api_response_callback=partial(
                                _api_response_callback,
                                tab=http_logs,
                                response_state=st.session_state.responses,
                            ),
                            api_key=st.session_state.api_key,
                            only_n_most_recent_images=st.session_state.only_n_most_recent_images,
                        )
                        st.rerun()
            except Exception as e:
                if "Overloaded" in str(e):
                    st.error("The service is currently overloaded. Please wait a moment and try again.")
                    await asyncio.sleep(2)
                else:
                    st.error(f"An error occurred: {str(e)}")

    with chat:
        new_message = st.chat_input(
            "Type a message to send to Claude to control the computer..."
        )
        # render past chats
        for message in st.session_state.messages:
            if isinstance(message["content"], str):
                _render_message(message["role"], message["content"])
            elif isinstance(message["content"], list):
                for block in message["content"]:
                    # the tool result we send back to the Anthropic API isn't sufficient to render all details,
                    # so we store the tool use responses
                    if isinstance(block, dict) and block["type"] == "tool_result":
                        _render_message(
                            Sender.TOOL, st.session_state.tools[block["tool_use_id"]]
                        )
                    else:
                        _render_message(
                            message["role"],
                            cast(BetaContentBlockParam | ToolResult, block),
                        )

        # render past http exchanges
        for identity, (request, response) in st.session_state.responses.items():
            _render_api_response(request, response, identity, http_logs)

        # render past chats
        if new_message:
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": [
                        *maybe_add_interruption_blocks(),
                        BetaTextBlockParam(type="text", text=new_message),
                    ],
                }
            )
            _render_message(Sender.USER, new_message)


def maybe_add_interruption_blocks():
    if not st.session_state.in_sampling_loop:
        return []
    # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
    # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
    result = []
    last_message = st.session_state.messages[-1]
    previous_tool_use_ids = [
        block["id"] for block in last_message["content"] if block["type"] == "tool_use"
    ]
    for tool_use_id in previous_tool_use_ids:
        st.session_state.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
        result.append(
            BetaToolResultBlockParam(
                tool_use_id=tool_use_id,
                type="tool_result",
                content=INTERRUPT_TOOL_ERROR,
                is_error=True,
            )
        )
    result.append(BetaTextBlockParam(type="text", text=INTERRUPT_TEXT))
    return result


@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error)
    _render_api_response(request, response, response_id, tab)


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and rendering it."""
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)


def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


def _render_message(
    sender: Sender,
    message: str | BetaContentBlockParam | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""
    # streamlit's hotreloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(message, str | dict)
    if not message or (
        is_tool_result
        and st.session_state.hide_images
        and not hasattr(message, "error")
        and not hasattr(message, "output")
    ):
        return
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not st.session_state.hide_images:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, dict):
            if message["type"] == "text":
                st.write(message["text"])
            elif message["type"] == "tool_use":
                st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
            else:
                # only expected return types are text and tool_use
                raise Exception(f'Unexpected response type {message["type"]}')
        else:
            st.markdown(message)


if __name__ == "__main__":
    asyncio.run(main())
