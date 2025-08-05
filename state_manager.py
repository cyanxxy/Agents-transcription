"""
Centralized state management for the ExactTranscriber application.
This module provides consistent methods for initializing and accessing session state.
"""
import streamlit as st
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from utils.state_optimizer import (
    get_state_optimizer, batch_state_updates, 
    StateOptimizer, StatePersistence
)

def initialize_state() -> None:
    """Initialize all required session state variables with default values."""
    optimizer = get_state_optimizer()
    
    # Load persistent state if available
    persistent_data = StatePersistence.load_state('app_state', storage='local')
    
    if persistent_data:
        # Restore persistent state
        for key, value in persistent_data.items():
            if key not in ['password_correct', 'processing_status']:  # Don't restore sensitive/temporary state
                _init_state_var(key, value)
    
    # Authentication state
    _init_state_var("password_correct", False)
    
    # Processing state
    _init_state_var("processing_status", "idle")  # idle, processing, complete, error
    _init_state_var("error_message", None)
    
    # File data
    _init_state_var("current_file_name", None)
    
    # Transcript data
    _init_state_var("transcript_text", None)
    _init_state_var("edited_transcript", None)
    _init_state_var("transcript_editor_content", "")
    
    # Model selection
    _init_state_var("selected_model_id", None)
    _init_state_var("model_display_radio", None)
    
    # Context information
    _init_state_var("content_type_select", "Podcast")
    _init_state_var("language_select", "English")
    _init_state_var("topic_input", "")
    _init_state_var("description_input", "")
    _init_state_var("num_speakers_input", 1)
    
    # Export options
    _init_state_var("export_format_select", "TXT")
    
    logging.debug("Session state initialized")

def _init_state_var(key: str, default_value: Any) -> None:
    """Initialize a session state variable if it doesn't exist."""
    if key not in st.session_state:
        st.session_state[key] = default_value

def get_state(key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state with caching.
    
    Args:
        key: The key to get from session state
        default: Value to return if key doesn't exist
        
    Returns:
        The value from session state or the default
    """
    optimizer = get_state_optimizer()
    return optimizer.get(key, default)

def set_state(key: str, value: Any, immediate: bool = True) -> None:
    """
    Set a value in session state with optimization.
    
    Args:
        key: The key to set in session state
        value: The value to set
        immediate: Whether to update immediately or batch
    """
    optimizer = get_state_optimizer()
    optimizer.set(key, value, immediate=immediate)
    
def update_states(state_dict: Dict[str, Any], commit: bool = True) -> None:
    """
    Update multiple session state variables at once with batching.
    
    Args:
        state_dict: Dictionary of {key: value} pairs to update
        commit: Whether to commit changes immediately
    """
    optimizer = get_state_optimizer()
    optimizer.batch_update(state_dict)
    
    if commit:
        optimizer.commit_batch(rerun=False)

def reset_transcript_states() -> None:
    """Reset all transcript-related state variables to defaults."""
    with batch_state_updates() as optimizer:
        optimizer.set("transcript_text", None)
        optimizer.set("edited_transcript", None)
        optimizer.set("transcript_editor_content", "")
        optimizer.set("processing_status", "idle")
        optimizer.set("error_message", None)
        optimizer.commit_batch(rerun=False)
    
@dataclass
class SessionStateValidator:
    """Validates session state values."""
    
    @staticmethod
    def validate_processing_status(status: str) -> bool:
        """Validate processing status value."""
        valid_statuses = ["idle", "processing", "complete", "error"]
        return status in valid_statuses
    
    @staticmethod
    def validate_file_name(filename: Optional[str]) -> bool:
        """Validate file name."""
        if filename is None:
            return True
        return isinstance(filename, str) and len(filename) > 0
    
    @staticmethod
    def validate_transcript(transcript: Optional[str]) -> bool:
        """Validate transcript text."""
        if transcript is None:
            return True
        return isinstance(transcript, str)


def get_state_with_validation(key: str, default: Any = None, validator=None) -> Any:
    """
    Get a value from session state with optional validation.
    
    Args:
        key: The key to get from session state
        default: Value to return if key doesn't exist
        validator: Optional validation function
        
    Returns:
        The value from session state or the default
    """
    value = st.session_state.get(key, default)
    
    if validator and value is not None:
        if not validator(value):
            logging.warning(f"Invalid value for session state key '{key}': {value}")
            return default
    
    return value


def set_state_with_validation(key: str, value: Any, validator=None) -> bool:
    """
    Set a value in session state with optional validation.
    
    Args:
        key: The key to set in session state
        value: The value to set
        validator: Optional validation function
        
    Returns:
        True if value was set, False if validation failed
    """
    if validator and value is not None:
        if not validator(value):
            logging.error(f"Validation failed for key '{key}' with value: {value}")
            return False
    
    st.session_state[key] = value
    return True


def get_metadata() -> Dict[str, str]:
    """
    Get the metadata dictionary from the current session state.
    Filters out None values and handles "Other" selections.
    
    Returns:
        Dictionary of metadata for the transcription prompt
    """
    metadata = {
        "content_type": get_state("content_type_select").lower() 
                        if get_state("content_type_select") != "Other" else None,
        "topic": get_state("topic_input") if get_state("topic_input") else None,
        "description": get_state("description_input") if get_state("description_input") else None,
        "language": get_state("language_select") 
                    if get_state("language_select") != "Other" else None
    }
    
    # Filter out None values
    return {k: v for k, v in metadata.items() if v is not None}


def update_processing_state(status: str, error_message: Optional[str] = None) -> None:
    """
    Update the processing state with validation.
    
    Args:
        status: New processing status
        error_message: Optional error message
    """
    validator = SessionStateValidator()
    
    if set_state_with_validation("processing_status", status, validator.validate_processing_status):
        if status == "error" and error_message:
            set_state("error_message", error_message)
        elif status != "error":
            set_state("error_message", None)
    else:
        logging.error(f"Failed to update processing state to: {status}")


def is_file_being_processed(filename: str) -> bool:
    """
    Check if a specific file is currently being processed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if the file is being processed
    """
    return (get_state("current_file_name") == filename and 
            get_state("processing_status") == "processing")


def is_file_complete(filename: str) -> bool:
    """
    Check if a specific file has completed processing.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if the file processing is complete
    """
    return (get_state("current_file_name") == filename and 
            get_state("processing_status") == "complete" and
            get_state("transcript_text") is not None)


def clear_transcript_data() -> None:
    """Clear all transcript-related data from session state."""
    with batch_state_updates() as optimizer:
        optimizer.set("transcript_text", None)
        optimizer.set("edited_transcript", None)
        optimizer.set("transcript_editor_content", "")
        optimizer.set("current_file_name", None)
        optimizer.commit_batch(rerun=False)

def save_state_to_persistent() -> None:
    """Save current state to persistent storage."""
    # Select which state to persist
    persist_keys = [
        "content_type_select", "language_select", 
        "topic_input", "description_input", "num_speakers_input",
        "export_format_select", "selected_model_id"
    ]
    
    state_to_save = {k: get_state(k) for k in persist_keys if get_state(k) is not None}
    StatePersistence.save_state('app_state', state_to_save, storage='local')

def cached_computation(key: str, func, *args, ttl=None, **kwargs) -> Any:
    """Perform a cached computation."""
    optimizer = get_state_optimizer()
    return optimizer.cached_computation(key, func, *args, ttl=ttl, **kwargs)
