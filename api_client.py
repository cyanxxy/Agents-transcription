"""
API Client module for ExactTranscriber.
This module provides minimal functions for API initialization.
"""
import os
import logging
from typing import Tuple, Dict, Any, Optional

import streamlit as st
import google.generativeai as genai
from jinja2 import Template

from config import GEMINI_MODELS, DEFAULT_MODEL
from utils.error_handler import (
    with_retry, with_error_handling, ErrorCategory, ErrorSeverity,
    StructuredError, RetryConfig
)

@with_retry(max_attempts=3, category=ErrorCategory.API_ERROR)
@with_error_handling(category=ErrorCategory.API_ERROR, severity=ErrorSeverity.HIGH)
def initialize_gemini(model_name: Optional[str] = None) -> Tuple[Any, Optional[str], str]:
    """
    Initialize the Gemini client.
    Uses API key from Streamlit secrets or environment variables.
    
    Args:
        model_name: The name of the Gemini model to use. If None, uses the default.
        
    Returns:
        Tuple (genai.Client, error_message, model_name):
            - genai.Client: The client object, or None if initialization fails
            - error_message: Error message string if initialization fails, otherwise None
            - model_name: The model name that will be used
    """
    # If no model specified, use default from model mapping
    if not model_name:
        model_id = GEMINI_MODELS.get(DEFAULT_MODEL)
    else:
        model_id = model_name
    
    api_key = None
    error_message = None

    # Try getting API key from Streamlit secrets
    try:
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            logging.info("Using API key from st.secrets['GOOGLE_API_KEY']")
        elif hasattr(st, 'secrets') and "secrets" in st.secrets and "GEMINI_API_KEY" in st.secrets["secrets"]:
            api_key = st.secrets["secrets"]["GEMINI_API_KEY"]
            logging.info("Using API key from st.secrets['secrets']['GEMINI_API_KEY']")
    except AttributeError as e:
        logging.warning(f"Could not access Streamlit secrets: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error accessing Streamlit secrets: {e}")

    # If not found in secrets, try environment variables
    if not api_key:
        # Try GOOGLE_API_KEY first (new standard)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            logging.info("Using API key from GOOGLE_API_KEY environment variable")
        else:
            # Fall back to GEMINI_API_KEY
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                logging.info("Using API key from GEMINI_API_KEY environment variable")

    # If API key is still not found
    if not api_key:
        raise StructuredError(
            message="API key not found",
            category=ErrorCategory.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.CRITICAL,
            user_message="API key not found. Please set GOOGLE_API_KEY in Streamlit secrets or as an environment variable.",
            recoverable=False
        )

    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Create a dummy client object for compatibility
        # In the new API, we don't need a client object
        class GeminiClient:
            def __init__(self):
                self.models = self
                self.files = self
            
            def generate_content(self, model, contents):
                model_obj = genai.GenerativeModel(model)
                return model_obj.generate_content(contents)
            
            def upload(self, file, config):
                return genai.upload_file(file, mime_type=config.get("mimeType"))
        
        client = GeminiClient()
        
        # Validate model name against common models
        valid_models = list(GEMINI_MODELS.values())
        if model_id not in valid_models:
            warning_msg = f"Model name '{model_id}' may not be valid. Using default model."
            logging.warning(warning_msg)
            st.warning(warning_msg)
            model_id = GEMINI_MODELS.get(DEFAULT_MODEL)
        
        logging.info(f"Successfully initialized Gemini client with model: {model_id}")
        # Return client and model name (no error)
        return client, None, model_id
        
    except Exception as e:
        # Categorize different API client initialization errors
        if "invalid api key" in str(e).lower() or "unauthorized" in str(e).lower():
            raise StructuredError(
                message=str(e),
                category=ErrorCategory.AUTHENTICATION_ERROR,
                severity=ErrorSeverity.CRITICAL,
                user_message="Invalid API key. Please check your API key and try again.",
                recoverable=False
            )
        elif "quota" in str(e).lower() or "rate limit" in str(e).lower():
            raise StructuredError(
                message=str(e),
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                user_message="API quota exceeded or rate limited. Please try again later.",
                details={'retry_after': 60},
                recoverable=True
            )
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            raise StructuredError(
                message=str(e),
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.HIGH,
                user_message="Network error connecting to Gemini API. Please check your internet connection.",
                recoverable=True
            )
        else:
            # Clean up potentially sensitive info from error
            error_msg = str(e)
            if api_key and api_key in error_msg:
                error_msg = error_msg.replace(api_key, "[REDACTED]")
            
            # Additional sanitization for common API key patterns
            import re
            error_msg = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[REDACTED]', error_msg)
            
            raise StructuredError(
                message=error_msg,
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                user_message=f"Failed to initialize Gemini client",
                recoverable=True
            )

def get_transcription_prompt(metadata: Dict[str, Any] = None) -> Template:
    """
    Return the Jinja2 template for transcription prompt.
    
    Args:
        metadata: Dictionary of metadata to include in the prompt
        
    Returns:
        Jinja2 Template for the transcription prompt
    """
    # Enhanced prompt for better speaker diarization and consistency
    return Template("""TASK: Perform accurate transcription and speaker diarization for the provided {{ metadata.content_type|default('audio file', true) }}.

CONTEXT:
{% if metadata and metadata.description %}- Description: {{ metadata.description }}
{% endif %}{% if metadata and metadata.topic %}- Topic: {{ metadata.topic }}
{% endif %}{% if metadata and metadata.language %}- Language: {{ metadata.language }}
{% endif %}- Number of distinct speakers: {{ num_speakers }}

INSTRUCTIONS:
1. Transcribe the audio accurately.
2. Perform speaker diarization: Identify the {{ num_speakers }} distinct speakers present in the audio.
3. Consistently label each speaker throughout the entire transcript using the format "Speaker 1:", "Speaker 2:", ..., "Speaker {{ num_speakers }}:". Ensure that each label (e.g., "Speaker 1") always refers to the same individual.
4. Include precise timestamps in [HH:MM:SS] format at the beginning of each speaker's utterance or segment.

OUTPUT FORMAT:
The output MUST strictly follow this format for each line:
[HH:MM:SS] Speaker X: Dialogue text...

EXAMPLE:
[00:00:05] Speaker 1: Hello, welcome to the meeting.
[00:00:08] Speaker 2: Thanks for having me.
[00:00:10] Speaker 1: Let's get started.

CRITICAL: Adhere strictly to the requested speaker labeling based on the {{ num_speakers }} distinct speakers identified. Maintain consistency in labeling throughout the transcript.

If there is music or a short jingle playing, signify like so:
[01:02] [MUSIC] or [01:02] [JINGLE]

If you can identify the name of the music or jingle playing then use that instead, eg:
[01:02] [Firework by Katy Perry] or [01:02] [The Sofa Shop jingle]

If there is some other sound playing try to identify the sound, eg:
[01:02] [Bell ringing]

Each individual caption should be quite short, a few short sentences at most.

Signify the end of the episode with [END].

Don't use any markdown formatting, like bolding or italics.

Only use characters from the English alphabet, unless you genuinely believe foreign characters are correct.

It is important that you use the correct words and spell everything correctly. Use the context to help.""")