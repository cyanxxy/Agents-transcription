# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ExactTranscriber is a Streamlit application that transcribes audio files using Google's Gemini AI API. It allows users to upload audio files, transcribe them, edit the generated transcript, and download the transcript in various formats (TXT, SRT, JSON).

## Environment Setup

1. **Required Dependencies:**
   - FFmpeg must be installed on the system
   - Python dependencies are managed through pyproject.toml

2. **Run the Application:**
   ```bash
   # Install dependencies
   pip install -e .
   
   # Run the application
   streamlit run main.py
   ```

3. **API Key Configuration:**
   - Set the Gemini API key as an environment variable: `export GOOGLE_API_KEY='your_api_key'`
   - Alternatively, configure it in Streamlit secrets

## Project Structure

- **main.py**: Entry point with agent-based architecture and enhanced UI
- **agents/**: Agent system for modular processing
  - `base_agent.py`: Base class and supervisor
  - `file_processing_agent.py`: Audio file handling
  - `transcription_agent.py`: Gemini API orchestration
  - `quality_assurance_agent.py`: Quality analysis
  - `editing_assistant_agent.py`: Smart editing features
  - `export_agent.py`: Multi-format export
  - `workflow_coordinator.py`: Agent orchestration
- **ui_components.py**: Reusable UI components
- **state_manager.py**: Session state management
- **api_client.py**: Gemini API initialization
- **styles.py**: CSS styling for the Streamlit UI
- **config.py**: Configuration constants
- **app_setup.py**: Application initialization

## Key Components

1. **Authentication Flow**: Simple password-based authentication using Streamlit secrets
2. **Audio Processing**:
   - Validates uploaded audio files
   - For large files (>20MB), splits audio into chunks for processing
   - Uses pydub for audio manipulation
3. **Transcription**:
   - Uses Google's Gemini API via Files API
   - Supports multiple models (Gemini 2.0 Flash, Gemini 2.5 Flash)
   - Processes chunks in parallel with ThreadPoolExecutor
4. **Export Formats**:
   - Plain text (TXT)
   - Subtitle format (SRT) with timestamps
   - Structured data (JSON)

## Common Development Tasks

1. **Adding a New Export Format**:
   - Add new format handler in `ExportAgent` class (`agents/export_agent.py`)
   - Update the format options in `EXPORT_FORMATS` in config.py

2. **Updating Transcription Models**:
   - Models are defined in the `GEMINI_MODELS` dictionary in config.py
   - Currently configured: Gemini 2.5 Flash and Gemini 2.5 Flash Lite

3. **UI Customization**:
   - Most styling is centralized in styles.py
   - Custom components use the 'styled-container' CSS class