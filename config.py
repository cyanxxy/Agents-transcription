"""
Configuration for the ExactTranscriber application.
This module centralizes constants and settings used throughout the application.
"""
import os
import logging

# --------- Logging Configuration ---------
LOG_FILE = "transcriber_app.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --------- API Configuration ---------
# Supported Gemini model IDs
GEMINI_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite-preview-06-17"
}

# Default model to use
DEFAULT_MODEL = "Gemini 2.5 Flash"

# --------- File Processing Configuration ---------
# MIME type mapping for audio formats
MIME_TYPE_MAPPING = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "flac": "audio/flac",
    "ogg": "audio/ogg"
}

# Allowed audio file extensions
ALLOWED_AUDIO_TYPES = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/x-wav']

# Maximum file size in bytes (200 MB - Streamlit limit)
MAX_FILE_SIZE = 200 * 1024 * 1024

# File size threshold for chunking (20 MB)
CHUNK_THRESHOLD_MB = 20

# Chunk duration in milliseconds (2 minutes)
CHUNK_DURATION_MS = 120000

# Maximum number of worker threads for parallel processing
MAX_WORKERS = 5

# Minimum chunk success percentage required before fallback
MIN_CHUNK_SUCCESS_PERCENTAGE = 0.8

# --------- Export Configuration ---------
# Export format options and their configurations
EXPORT_FORMATS = {
    "TXT": {
        "extension": "txt",
        "mime_type": "text/plain",
        "description": "Simple text format with timestamps and speakers"
    },
    "SRT": {
        "extension": "srt",
        "mime_type": "application/x-subrip",
        "description": "Subtitle format with precise timestamps"
    },
    "JSON": {
        "extension": "json",
        "mime_type": "application/json",
        "description": "Structured data format for programmatic use"
    }
}

# --------- Content Context Options ---------
CONTENT_TYPES = ["Podcast", "Interview", "Meeting", "Presentation", "Other"]
LANGUAGES = ["English", "Spanish", "French", "German", "Other"]

# --------- SRT Export Configuration ---------
# Default subtitle duration in seconds for SRT export
DEFAULT_SUBTITLE_DURATION_SECONDS = 3

# --------- Validation Configuration ---------
# Valid export formats
VALID_EXPORT_FORMATS = ['txt', 'srt', 'json']
