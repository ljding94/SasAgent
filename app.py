#!/usr/bin/env python3
"""
SAS Agent Frontend UI with Gradio
Interactive web interface for SAS data analysis and generation
"""

import gradio as gr
import os
import logging
from io import StringIO
import sys
import datetime
import traceback
import glob
import re

# Disable CrewAI telemetry to prevent connection errors
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['CREWAI_TELEMETRY_DISABLED'] = 'true'
os.environ['DO_NOT_TRACK'] = '1'

# Configure logging to suppress telemetry errors
logging.basicConfig(level=logging.WARNING)
# Disable specific telemetry loggers
logging.getLogger('crewai.telemetry').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)
logging.getLogger('requests.packages.urllib3').setLevel(logging.CRITICAL)

# Import backend directly (app.py is now in root directory)
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Create designated directories for file management
CACHE_DIR = os.path.join(os.getcwd(), "cache")
UPLOADS_DIR = os.path.join(CACHE_DIR, "uploads")
GENERATED_DIR = os.path.join(CACHE_DIR, "generated")
PLOTS_DIR = os.path.join(CACHE_DIR, "plots")

# Create directories if they don't exist
for directory in [CACHE_DIR, UPLOADS_DIR, GENERATED_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

try:
    from crewai_sas_agents import UnifiedSASAnalysisSystem
    BACKEND_AVAILABLE = True
    print("✅ Backend available")
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📁 Uploads directory: {UPLOADS_DIR}")
    print(f"📁 Generated directory: {GENERATED_DIR}")
    print(f"📁 Plots directory: {PLOTS_DIR}")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"⚠️  Backend not available: {e}")

# Set up logging for the right panel
log_stream = StringIO()


def is_complete_request(message):
    """
    Determine if a user message is complete enough to bypass interactive guidance.
    Complete requests should go directly to the coordinator agent.
    """
    message_lower = message.lower()

    # Complete generation requests
    generation_keywords = ['generate', 'create', 'synthetic', 'simulate']
    generation_patterns = [
        r'generate.*data.*for',
        r'create.*synthetic.*data',
        r'simulate.*scattering',
        r'generate.*using.*model'
    ]

    # Complete SLD calculation requests
    sld_keywords = ['sld', 'scattering length density', 'calculate']
    sld_patterns = [
        r'calculate.*sld.*for',
        r'sld.*for.*\w+',
        r'scattering length density'
    ]

    # Complete fitting requests
    fitting_keywords = ['fit', 'analyze', 'model']
    fitting_indicators = ['csv', 'data', 'file', 'experimental']

    # Check for generation requests
    if any(keyword in message_lower for keyword in generation_keywords):
        if any(re.search(pattern, message_lower) for pattern in generation_patterns):
            return True
        # Also check if it mentions specific models or shapes
        if any(shape in message_lower for shape in ['sphere', 'cylinder', 'ellipsoid', 'lamellar']):
            return True

    # Check for SLD calculation requests
    if any(keyword in message_lower for keyword in sld_keywords):
        if any(re.search(pattern, message_lower) for pattern in sld_patterns):
            return True

    # Check for fitting requests (usually have data files mentioned)
    if any(keyword in message_lower for keyword in fitting_keywords):
        if any(indicator in message_lower for indicator in fitting_indicators):
            return True

    # If none of the above, it's likely incomplete/ambiguous
    return False


def clean_ansi_escape_sequences(text):
    """Remove ANSI escape sequences and clean up terminal formatting"""
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', text)

    # Remove carriage returns that cause overwriting
    cleaned = cleaned.replace('\r', '')

    # Handle CrewAI's verbose box drawing characters and formatting
    # Replace box drawing characters with simpler alternatives
    cleaned = cleaned.replace('╭', '+').replace('╮', '+')
    cleaned = cleaned.replace('╰', '+').replace('╯', '+')
    cleaned = cleaned.replace('│', '|').replace('─', '-')
    cleaned = cleaned.replace('├', '+').replace('┤', '+')
    cleaned = cleaned.replace('┬', '+').replace('┴', '+')
    cleaned = cleaned.replace('┼', '+')

    # Clean up excessive newlines but preserve structure
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Preserve indentation for CrewAI's structured output
    lines = cleaned.split('\n')
    processed_lines = []
    for line in lines:
        # Don't strip leading whitespace for indented content
        if line.strip():
            processed_lines.append(line)
        else:
            processed_lines.append('')

    return '\n'.join(processed_lines)


# Create a custom handler that writes to both our stream and console
class DualHandler(logging.Handler):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def emit(self, record):
        msg = self.format(record)
        self.stream.write(msg + '\n')
        # Also print to console for debugging
        print(msg)


# Custom stdout/stderr capture for CrewAI print statements
class StreamCapture:
    def __init__(self, original_stream, log_stream):
        self.original_stream = original_stream
        self.log_stream = log_stream

    def write(self, text):
        # Write to original stream (terminal)
        self.original_stream.write(text)
        self.original_stream.flush()

        # Clean and write to our log stream
        if text.strip():  # Only log non-empty lines
            cleaned_text = clean_ansi_escape_sequences(text)
            if cleaned_text.strip():  # Only write if there's still content after cleaning
                self.log_stream.write(cleaned_text)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        # Delegate other attributes to original stream
        return getattr(self.original_stream, name)


# Configure logging to capture CrewAI output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Add our custom handler
dual_handler = DualHandler(log_stream)
dual_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(dual_handler)

# Also capture CrewAI specific loggers
crewai_logger = logging.getLogger('crewai')
crewai_logger.setLevel(logging.INFO)
crewai_logger.addHandler(dual_handler)

# Capture LiteLLM logs too
litellm_logger = logging.getLogger('LiteLLM')
litellm_logger.setLevel(logging.INFO)
litellm_logger.addHandler(dual_handler)


def log_message(msg):
    """Helper to log with timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info(f"[{timestamp}] {msg}")


# Global state for session management
session_state = {}


def save_config(api_key, model):
    """Save API key and model to session state."""
    global session_state
    session_state['api_key'] = api_key
    session_state['model'] = model

    # Set environment variable for immediate use
    if api_key:
        os.environ['OPENROUTER_API_KEY'] = api_key

    log_message(f"Configuration saved - Model: {model}")
    return "✅ Configuration saved successfully!"


def get_logs():
    """Get current logs from the stream."""
    # Get all content from the stream
    content = log_stream.getvalue()

    # Return all content without truncation
    return content


def get_cache_files():
    """Get list of files in cache directories."""
    files_info = {
        "uploads": [],
        "generated": [],
        "plots": []
    }

    try:
        # Get uploaded files
        for file in os.listdir(UPLOADS_DIR):
            if os.path.isfile(os.path.join(UPLOADS_DIR, file)):
                file_path = os.path.join(UPLOADS_DIR, file)
                file_size = os.path.getsize(file_path)
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                files_info["uploads"].append(f"{file} ({file_size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")

        # Get generated files
        for file in os.listdir(GENERATED_DIR):
            if os.path.isfile(os.path.join(GENERATED_DIR, file)):
                file_path = os.path.join(GENERATED_DIR, file)
                file_size = os.path.getsize(file_path)
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                files_info["generated"].append(f"{file} ({file_size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")

        # Get plot files
        for file in os.listdir(PLOTS_DIR):
            if os.path.isfile(os.path.join(PLOTS_DIR, file)):
                file_path = os.path.join(PLOTS_DIR, file)
                file_size = os.path.getsize(file_path)
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                files_info["plots"].append(f"{file} ({file_size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M')})")

    except Exception as e:
        log_message(f"Error reading cache directories: {e}")

    return files_info


def clear_cache():
    """Clear all cache directories."""
    try:
        import shutil
        for directory in [UPLOADS_DIR, GENERATED_DIR, PLOTS_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
        log_message("Cache directories cleared successfully")
        return "✅ Cache cleared successfully!"
    except Exception as e:
        error_msg = f"❌ Error clearing cache: {str(e)}"
        log_message(error_msg)
        return error_msg


def clear_logs():
    """Clear the log stream."""
    log_stream.truncate(0)
    log_stream.seek(0)
    return ""


def format_results_for_chat(result):
    """Format analysis results for chat display."""
    if not result:
        return "❌ No results returned from the analysis."

    # Extract text content
    text_parts = []

    # First, let's handle the main result - CrewAI returns a CrewOutput object
    if hasattr(result, 'raw'):
        # CrewAI result with .raw attribute
        #text_parts.append("**🤖 Analysis Complete!**")
        #text_parts.append("✅ **Task completed successfully!**\n")
        text_parts.append(str(result.raw))
    elif isinstance(result, str):
        # String result
        #text_parts.append("**🤖 Analysis Results:**")
        #text_parts.append("✅ **Task completed successfully!**\n")
        text_parts.append(result)
    elif isinstance(result, dict):
        # Dictionary result - handle various possible keys

        # Handle other dictionary results
        #text_parts.append("**🤖 Analysis Complete!**")
        #text_parts.append("✅ **Task completed successfully!**\n")

        # Handle success/error status
        if 'success' in result:
            if not result['success']:
                text_parts[-1] = "❌ **Task failed!**\n"
                if 'error' in result:
                    text_parts.append(f"**💥 Error Details:** {result['error']}")
                if 'prompt' in result:
                    text_parts.append(f"**📝 Original Prompt:** {result['prompt']}")
                return "\n".join(text_parts)

        # Look for the main results from CrewAI
        if 'results' in result and result['results']:
            #text_parts.append("**📋 Agent Output:**")
            result_text = str(result['results'])
            # Clean up the result text a bit
            if result_text.startswith("Task output:"):
                result_text = result_text.replace("Task output:", "").strip()
            text_parts.append(result_text)

        # Handle task type with appropriate emoji
        if 'task_type' in result:
            task_emoji = "🧪" if result['task_type'] == 'generation' else "📊"
            text_parts.append(f"\n**{task_emoji} Task Type:** {result['task_type'].title()}")

        # Handle sample description
        if 'sample_description' in result:
            text_parts.append(f"**🔬 Sample:** {result['sample_description']}")

        # Handle fitted parameters
        if 'fitted_parameters' in result:
            text_parts.append("\n**📊 Fitted Parameters:**")
            params = result['fitted_parameters']
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, float):
                        text_parts.append(f"- {key}: {value:.4f}")
                    else:
                        text_parts.append(f"- {key}: {value}")
            else:
                text_parts.append(str(params))

        # Handle R-squared
        if 'r_squared' in result:
            text_parts.append(f"\n**📈 R-squared:** {result['r_squared']:.4f}")

        # Handle model information
        if 'model_used' in result:
            text_parts.append(f"**🔧 Model Used:** {result['model_used']}")

        # Handle RAG enhancement indicator
        if 'rag_enhanced' in result and result['rag_enhanced']:
            text_parts.append("**🧠 RAG Enhanced:** Yes")

        # Handle file paths
        files_info = []
        file_keys = ['csv_path', 'plot_file', 'plot_path', 'figure_path', 'data_file']
        for key in file_keys:
            if key in result and result[key]:
                file_path = result[key]
                # Show clean filename only, not full path
                filename = os.path.basename(file_path)
                file_type = key.replace('_', ' ').title()
                files_info.append(f"{file_type}: {filename}")

        if files_info:
            text_parts.append("\n**📁 Generated Files:**")
            text_parts.extend([f"- {info}" for info in files_info])

        # Look for other common result keys if we haven't found main content yet
        if len([p for p in text_parts if not p.startswith("**🤖") and not p.startswith("✅")]) == 0:
            other_keys = ['output', 'collaborative_analysis', 'analysis', 'summary']
            for key in other_keys:
                if key in result and result[key]:
                    text_parts.append(f"\n**{key.title()}:**")
                    text_parts.append(str(result[key]))
                    break

            # If still no meaningful content, show the full result
            if len([p for p in text_parts if not p.startswith("**🤖") and not p.startswith("✅")]) == 0:
                text_parts.append("\n**Raw Output:**")
                text_parts.append(str(result))
    else:
        # Unknown result type
        text_parts.append("**🤖 Analysis Results:**")
        text_parts.append("✅ **Task completed successfully!**\n")
        text_parts.append(str(result))

    return "\n".join(text_parts)


def process_message(message, history, uploaded_file=None):
    """Process chat messages and integrate with agents."""
    global session_state

    # Check if backend is available
    if not BACKEND_AVAILABLE:
        error_msg = "❌ Backend not available. Please check your installation."
        log_message("Backend not available")
        return error_msg, None

    # Get API key from session state or environment
    api_key = session_state.get('api_key', os.getenv('OPENROUTER_API_KEY'))
    if not api_key:
        warning_msg = "⚠️ No API key configured. Please add your OpenRouter API key in the Settings panel."
        log_message("No API key configured")
        return warning_msg, None

    # Set API key in environment
    os.environ['OPENROUTER_API_KEY'] = api_key
    model = session_state.get('model', 'openai/gpt-4o-mini')

    log_message(f"Processing user message: {message}")
    log_message(f"🤖 Using model: {model}")
    log_message(f"🔑 API key configured: {'✅' if api_key else '❌'}")

    # Clear the log stream before starting
    log_stream.seek(0)
    log_stream.truncate(0)

    try:
        # Handle uploaded file - copy to uploads directory
        data_path = None
        if uploaded_file is not None:
            original_path = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
            if original_path and os.path.exists(original_path):
                # Create a unique filename to avoid conflicts
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"upload_{timestamp}_{os.path.basename(original_path)}"
                data_path = os.path.join(UPLOADS_DIR, filename)

                # Copy the uploaded file to our uploads directory
                import shutil
                shutil.copy2(original_path, data_path)
                log_message(f"File uploaded and saved to: {data_path}")
            else:
                log_message(f"Warning: Uploaded file not found at {original_path}")

        # Set environment variables for the backend to use our directories
        os.environ['SAS_CACHE_DIR'] = CACHE_DIR
        os.environ['SAS_UPLOADS_DIR'] = UPLOADS_DIR
        os.environ['SAS_GENERATED_DIR'] = GENERATED_DIR
        os.environ['SAS_PLOTS_DIR'] = PLOTS_DIR

        # Call the backend analysis function
        log_message("Starting collaborative SAS analysis...")

        # Capture stdout/stderr to include CrewAI print statements in logs
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Replace stdout/stderr with our capturing streams
            sys.stdout = StreamCapture(original_stdout, log_stream)
            sys.stderr = StreamCapture(original_stderr, log_stream)

            # The logging should now be captured automatically by our DualHandler

            # Always use unified coordinator system
            log_message("🤖 Using Unified Coordinator System")

            # Initialize unified system with current API key and model
            # Check if we need to recreate the system (model changed or doesn't exist)
            current_system_model = session_state.get('unified_system_model')
            current_system_api_key = session_state.get('unified_system_api_key')

            if ('unified_system' not in session_state or
                current_system_model != model or
                current_system_api_key != api_key):

                log_message(f"Creating new UnifiedSASAnalysisSystem with model: {model}")
                session_state['unified_system'] = UnifiedSASAnalysisSystem(api_key=api_key, model=model)
                session_state['unified_system_model'] = model
                session_state['unified_system_api_key'] = api_key
                log_message("Created new UnifiedSASAnalysisSystem")
            else:
                log_message(f"Using existing UnifiedSASAnalysisSystem with model: {model}")

            unified_system = session_state['unified_system']

            # Use the unified system
            result = unified_system.analyze_data(
                prompt=message,
                data_path=data_path,
                output_folder=GENERATED_DIR,
                chat_history=history if history else []
            )
        finally:
            # Always restore original streams
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        log_message("Analysis complete!")

        # Debug: Log the result structure
        log_message(f"Result type: {type(result)}")
        if hasattr(result, '__dict__'):
            log_message(f"Result attributes: {list(result.__dict__.keys())}")
        elif isinstance(result, dict):
            log_message(f"Result keys: {list(result.keys())}")

        # Format response for chat
        response_text = format_results_for_chat(result)

        # Find plot file - prioritize our managed directories
        plot_path = None
        if result:
            # Check multiple possible plot locations
            possible_plots = [
                result.get('plot_file'),
                result.get('plot_path'),
                result.get('figure_path')
            ]

            for path in possible_plots:
                if path and os.path.exists(path):
                    plot_path = path
                    log_message(f"Plot found: {plot_path}")
                    break

            # If no plot found in result, check our managed directories
            if not plot_path:
                # Determine task type from message content to prioritize the right directory
                is_generation_task = any(keyword in message.lower() for keyword in [
                    'generate', 'create', 'synthetic', 'simulate', 'random', 'sample'
                ])

                if is_generation_task:
                    # For generation tasks, prioritize generated folder
                    plot_patterns = [
                        os.path.join(GENERATED_DIR, '*.png'),
                        os.path.join(GENERATED_DIR, '*.jpg'),
                        os.path.join(PLOTS_DIR, '*.png'),
                        os.path.join(PLOTS_DIR, '*.jpg'),
                        'data/*.png', 'data/*.jpg',  # Legacy data directory
                        '*.png', '*.jpg',
                        'plots/*.png', 'plots/*.jpg'
                    ]
                else:
                    # For fitting tasks, prioritize plots folder
                    plot_patterns = [
                        os.path.join(PLOTS_DIR, '*.png'),
                        os.path.join(PLOTS_DIR, '*.jpg'),
                        os.path.join(GENERATED_DIR, '*.png'),
                        os.path.join(GENERATED_DIR, '*.jpg'),
                        'data/*.png', 'data/*.jpg',  # Legacy data directory
                        '*.png', '*.jpg',
                        'plots/*.png', 'plots/*.jpg'
                    ]

                for pattern in plot_patterns:
                    files = glob.glob(pattern)
                    if files:
                        # Get the most recent file
                        plot_path = max(files, key=os.path.getctime)
                        log_message(f"Found recent plot ({'generation' if is_generation_task else 'fitting'} task): {plot_path}")
                        break

        return response_text, plot_path

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        log_message(f"Error occurred: {str(e)}")
        log_message(f"Traceback: {traceback.format_exc()}")
        return error_msg, None


def create_ui():
    """Create the Gradio UI."""

    with gr.Blocks(
        title="SAS Agent Interactive UI",
        theme=gr.themes.Soft(),
        css="""
        .log-box {
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            font-size: 12px;
            line-height: 1.3;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
            overflow-x: auto;
        }
        .log-box textarea {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        .settings-panel {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
        }
        """
    ) as demo:

        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <h1 style="margin-bottom: 10px;"> 👨‍🔬 SAS Agent 🤖</h1>
            <p style="font-style: italic; color: #666; margin: 0;">Intelligent Small-Angle Scattering Analysis & Data Generation</p>
        </div>
        """)

        with gr.Row(equal_height=False):
            # Left Column: Chat Interface + System Logs
            with gr.Column(scale=2):
                gr.Markdown("## 💬 Chat with SAS Agent")

                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=330,
                    show_label=False,
                    avatar_images=None,
                    type="messages"
                )

                # State to store message for processing
                message_state = gr.State("")

                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Enter your prompt, e.g., 'Generate data for spheres' or 'Fit my data to a cylinder model'",
                    lines=2
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=3)
                    clear_btn = gr.Button("🗑️ Clear Chat", scale=1)

                # Example prompts
                gr.Markdown("###  Example Prompts:")
                with gr.Row():
                    example_btns = [
                        gr.Button("What can you do for me?", size="sm"),
                        gr.Button("Generate synthetic data for spherical colloidal particles of diameter 100 A", size="sm"),
                    ]
                with gr.Row():
                    example_btns.extend([
                        gr.Button("What is the SLD of Tetrahydrofuran", size="sm"),
                        gr.Button("Fit uploaded data to a flexible cylinder model, solven is Tetrahydrofuran", size="sm")
                    ])

                # System Logs Panel (moved from right column)
                gr.Markdown("## 📋 System Logs")
                log_output = gr.Textbox(
                    # label="Real-time Logs",
                    lines=10,
                    max_lines=10,
                    autoscroll=True,
                    show_copy_button=True,
                    elem_classes=["log-box"],
                    interactive=False
                )

                with gr.Row():
                    refresh_logs_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                    clear_logs_btn = gr.Button("🧹 Clear", size="sm", scale=1)

            # Right Column: Settings + Plot Display
            with gr.Column(scale=1):
                # Settings Panel
                gr.Markdown("## ⚙️ Settings")
                with gr.Group(elem_classes=["settings-panel"]):
                    api_key_input = gr.Textbox(
                        label="🔑 OpenRouter API Key",
                        type="password",
                        placeholder="Enter your API key...",
                        info="Required for LLM access"
                    )

                    model_choice = gr.Dropdown(
                        choices=[
                            "openai/gpt-4o-mini",
                            "openai/gpt-4o",
                            "openai/gpt-5",
                            "anthropic/claude-sonnet-4",
                            "x-ai/grok-3",
                            "x-ai/grok-4",
                            "google/gemini-2.5-pro",
                            "google/gemini-2.5-flash"
                        ],
                        label="🤖 LLM Model",
                        value="openai/gpt-4o-mini",
                        info="Select the language model"
                    )

                # Image display for plots (moved below settings)
                gr.Markdown("## 📈 Plot Generation")
                plot_display = gr.Image(
                    # label="📈 Generated Plot",
                    show_label=True,
                    height=350,
                    visible=True
                )

                # Data Management Panel
                gr.Markdown("## 📁 Data Management")
                # File upload section
                file_upload = gr.File(
                    label="📁 Upload CSV Data (optional)",
                    file_types=[".csv", ".txt", ".dat"],
                    type="filepath"
                )

                gr.Markdown("""
                **Supported formats:**
                • CSV files with Q, I(Q), and optional error columns
                • Text files with space/tab separated values
                • DAT files from instrument software
                """)

                # Auto-refresh logs during processing
                log_timer = gr.Timer(2.0)  # Refresh every 2 seconds

        # Event handlers
        def submit_message_immediate(message, history, uploaded_file):
            """Immediately show user message and clear input"""
            if not message.strip():
                return history, message, get_logs(), gr.Image(visible=True), message

            # Add user message to history immediately
            updated_history = history + [{"role": "user", "content": message}]

            # Clear input field immediately and keep plot visible but clear
            return updated_history, "", get_logs(), gr.Image(value=None, visible=True), message

        def process_and_respond(stored_message, history, uploaded_file):
            """Process the message and add response"""
            if not stored_message.strip():
                return history, get_logs(), gr.Image(visible=False)

            # Process the message
            response_text, plot_path = process_message(stored_message, history, uploaded_file)

            # Add response to history (messages format)
            updated_history = history + [{"role": "assistant", "content": response_text}]

            # Handle plot display
            if plot_path and os.path.exists(plot_path):
                plot_update = gr.Image(value=plot_path, visible=True)
                log_message(f"Displaying plot: {plot_path}")
            else:
                plot_update = gr.Image(value=None, visible=True)

            return updated_history, get_logs(), plot_update

        def set_example_prompt(example_text):
            return example_text

        # Wire up event handlers
        def auto_save_api_key(api_key):
            """Auto-save API key when changed"""
            session_state['api_key'] = api_key
            if api_key:
                os.environ['OPENROUTER_API_KEY'] = api_key
            log_message("API key updated")
            return api_key

        def auto_save_model(model):
            """Auto-save model when changed"""
            session_state['model'] = model
            log_message(f"Model updated: {model}")
            return model

        # Auto-save settings when changed
        api_key_input.change(
            auto_save_api_key,
            inputs=api_key_input,
            outputs=api_key_input
        )

        model_choice.change(
            auto_save_model,
            inputs=model_choice,
            outputs=model_choice
        )

        submit_btn.click(
            submit_message_immediate,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input, log_output, plot_display, message_state]
        ).then(
            process_and_respond,
            inputs=[message_state, chatbot, file_upload],
            outputs=[chatbot, log_output, plot_display]
        )

        msg_input.submit(
            submit_message_immediate,
            inputs=[msg_input, chatbot, file_upload],
            outputs=[chatbot, msg_input, log_output, plot_display, message_state]
        ).then(
            process_and_respond,
            inputs=[message_state, chatbot, file_upload],
            outputs=[chatbot, log_output, plot_display]
        )

        clear_btn.click(
            lambda: ([], get_logs(), gr.Image(visible=False)),
            outputs=[chatbot, log_output, plot_display]
        )

        clear_logs_btn.click(
            clear_logs,
            outputs=log_output
        )

        refresh_logs_btn.click(
            get_logs,
            outputs=log_output
        )

        # Auto-refresh logs during processing
        log_timer.tick(
            get_logs,
            outputs=log_output
        )

        # Example button handlers
        for i, btn in enumerate(example_btns):
            btn.click(
                set_example_prompt,
                inputs=gr.State(btn.value),
                outputs=msg_input
            )

        # Initial log message
        demo.load(
            lambda: log_message("SAS Agent UI initialized successfully!"),
            outputs=None
        )

        demo.load(
            get_logs,
            outputs=log_output
        )

    return demo


if __name__ == "__main__":
    # Initialize logging
    log_message("Starting SAS Agent Frontend...")

    # Create and launch the UI
    demo = create_ui()

    print("🚀 Launching SAS Agent Frontend...")
    print("📝 Access the interface at: http://localhost:7862")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        debug=False
    )
