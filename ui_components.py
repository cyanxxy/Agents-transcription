"""
UI components module for ExactTranscriber.
This module contains reusable UI components for the Streamlit interface.
"""
import streamlit as st
import os
from typing import Optional, Tuple, Dict, Any

from streamlit_ace import st_ace

from config import GEMINI_MODELS, DEFAULT_MODEL, EXPORT_FORMATS
from styles import format_transcript_line


def render_model_selection() -> str:
    """Render model selection UI and return selected model ID."""
    with st.container():
        st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 10px;'>Select Transcription Model</h4>", unsafe_allow_html=True)
        
        # Get available model options
        model_options = list(GEMINI_MODELS.keys())
        
        # Find default index
        default_model_name = DEFAULT_MODEL
        default_index = 0
        
        if default_model_name in model_options:
            default_index = model_options.index(default_model_name)
        
        # Model selection
        model_display = st.radio(
            "Select transcription model", 
            options=model_options,
            index=default_index,
            horizontal=True,
            help="Choose between faster (Flash) or more accurate (Pro) transcription",
            label_visibility="collapsed",
            key="model_display_radio"
        )
        
        # Store the actual model ID
        selected_model_id = GEMINI_MODELS.get(model_display, GEMINI_MODELS[DEFAULT_MODEL])
        st.session_state.selected_model_id = selected_model_id
        
        # Display appropriate caption
        if model_display and "Flash" in model_display:
            st.caption("⚡ Optimized for speed, good for most transcriptions")
        else:
            st.caption("✨ Higher quality, better for complex audio or multiple speakers")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    return selected_model_id


def render_context_inputs() -> Dict[str, Any]:
    """Render optional context inputs and return metadata."""
    with st.expander("Optional Context", expanded=False):
        col1_ctx, col2_ctx = st.columns(2)
        
        with col1_ctx:
            content_type = st.selectbox(
                "Type", 
                options=["Podcast", "Interview", "Meeting", "Presentation", "Other"], 
                index=0, 
                key="ctx_type"
            )
            language = st.selectbox(
                "Language", 
                options=["English", "Spanish", "French", "German", "Other"], 
                index=0, 
                key="ctx_lang"
            )
        
        with col2_ctx:
            topic = st.text_input("Topic", key="ctx_topic")
            description = st.text_input("Description", key="ctx_desc")
            num_speakers = st.number_input(
                "Number of Speakers", 
                min_value=1, 
                value=1, 
                step=1, 
                help="Specify the total number of distinct speakers in the audio.", 
                key="ctx_speakers"
            )
    
    # Build metadata dictionary
    metadata = {
        "content_type": content_type.lower() if content_type != "Other" else None,
        "topic": topic if topic else None,
        "description": description if description else None,
        "language": language if language != "Other" else None
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return metadata, num_speakers


def render_file_upload() -> Tuple[Optional[Any], bool]:
    """Render file upload section and return uploaded file and process button state."""
    with st.container():
        st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 10px;'>Upload Your Audio File</h4>", unsafe_allow_html=True)
        st.caption("Supported formats: MP3, WAV, OGG (max 200MB)")
        
        uploaded_file = st.file_uploader(
            "Upload audio file", 
            type=['mp3', 'wav', 'ogg'], 
            key="file_uploader_widget"
        )
        
        process_button = False
        
        if uploaded_file:
            # Simple validation inline
            valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.webm']
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_ext not in valid_extensions:
                st.error(f"Unsupported file type: {file_ext}. Please upload a valid audio file.")
            elif uploaded_file.size > 200 * 1024 * 1024:  # 200MB limit
                st.error("File too large. Please upload a file smaller than 200MB.")
            else:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.caption(f"File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
                
                # Display status/error related to this specific file
                if st.session_state.current_file_name == uploaded_file.name:
                    if st.session_state.processing_status == "processing":
                        st.info("Transcription is already in progress...")
                    elif st.session_state.processing_status == "error":
                        st.error(f"Previous attempt failed: {st.session_state.error_message}")
                
                # Show transcribe button only if idle or error state for this file
                if not (st.session_state.current_file_name == uploaded_file.name and 
                        st.session_state.processing_status in ["processing", "complete"]):
                    process_button = st.button("🎯 Transcribe", type="primary", key="transcribe_button")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    return uploaded_file, process_button


def render_transcript_tabs(transcript_text: str, uploaded_file_name: str):
    """Render transcript display, edit, and export tabs."""
    tabs = st.tabs(["Transcript", "Edit", "Export"])
    
    with tabs[0]:
        render_transcript_display(transcript_text)
    
    with tabs[1]:
        render_transcript_editor()
    
    with tabs[2]:
        render_export_options(uploaded_file_name)


def render_transcript_display(transcript_text: str):
    """Render the transcript display tab."""
    st.markdown("### Transcript")
    with st.container():
        st.markdown("<div class='styled-container transcript-container'>", unsafe_allow_html=True)
        formatted_lines = []
        
        for line in transcript_text.split('\n'):
            if line.strip():
                formatted_lines.append(format_transcript_line(line))
        
        formatted_transcript = '<p>' + '</p><p>'.join(formatted_lines) + '</p>'
        st.markdown(formatted_transcript, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_transcript_editor():
    """Render the transcript editor tab."""
    st.markdown("### Edit Transcript")
    
    # Initialize editor content if empty
    if not st.session_state.transcript_editor_content:
        st.session_state.transcript_editor_content = st.session_state.get(
            "edited_transcript", 
            st.session_state.get("transcript_text", "")
        )
    
    edited_text = st_ace(
        value=st.session_state.transcript_editor_content, 
        language='text',
        theme='tomorrow_night',
        keybinding='vscode',
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=False,
        readonly=False,
        height=400,
        key="transcript_editor_widget"
    )
    
    # Save button
    if st.button("Save Edits", key="save_edits_button"):
        st.session_state.edited_transcript = edited_text
        st.session_state.transcript_editor_content = edited_text
        st.success("Edits saved!")


def render_export_options(uploaded_file_name: str):
    """Render the export options tab."""
    st.markdown("### Export Transcript")
    
    with st.container():
        st.markdown("<div class='styled-container'>", unsafe_allow_html=True)
        st.markdown("Choose a format and download your transcript:")
        
        col1_exp, col2_exp = st.columns([3, 2])
        
        with col1_exp:
            export_format = st.selectbox(
                "Export Format", 
                options=list(EXPORT_FORMATS.keys()), 
                index=0,
                help="TXT: Plain text | SRT: Subtitles | JSON: Data format", 
                key="export_format_select"
            )
            
            format_info = EXPORT_FORMATS[export_format]
            st.caption(format_info["description"])
        
        # Get export content
        export_content = st.session_state.get(
            "transcript_editor_content", 
            st.session_state.get("edited_transcript", st.session_state.get("transcript_text", ""))
        )
        
        # Format content for export based on type
        if format_info["extension"] == ".txt":
            formatted_content = export_content
        elif format_info["extension"] == ".srt":
            # Simple SRT formatting
            lines = export_content.split('\n')
            formatted_content = ""
            counter = 1
            for line in lines:
                if line.strip():
                    formatted_content += f"{counter}\n00:00:{counter:02d},000 --> 00:00:{counter+5:02d},000\n{line}\n\n"
                    counter += 1
        elif format_info["extension"] == ".json":
            import json
            formatted_content = json.dumps({
                "transcript": export_content,
                "filename": st.session_state.get("current_file_name", "unknown"),
                "timestamp": str(st.session_state.get("transcription_timestamp", ""))
            }, indent=2)
        else:
            formatted_content = export_content
        
        # Generate filename
        base_filename = os.path.splitext(uploaded_file_name)[0]
        export_filename = f"{base_filename}_transcript.{format_info['extension']}"
        
        # Check for unsaved edits
        unsaved_edits = (
            st.session_state.get("transcript_editor_content", "") != 
            st.session_state.get("edited_transcript", "")
        )
        if unsaved_edits:
            st.warning("You have unsaved edits. Please click 'Save Edits' before exporting.")
        
        # Preview
        with st.expander("Preview Export"):
            st.text(formatted_content[:500] + ("..." if len(formatted_content) > 500 else ""))
        
        with col2_exp:
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label=f"📥 Download {export_format.upper()}", 
                data=formatted_content,
                file_name=export_filename, 
                mime=format_info["mime_type"], 
                type="primary",
                use_container_width=True, 
                key=f"download_button_{export_format}"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render the application footer."""
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8em; margin-top: 50px;'>"
        "<a href='https://www.linkedin.com/in/mansour-damanpak/' target='_blank' "
        "style='color: #1E88E5; text-decoration: none;'>Developed by Mansour Damanpak</a></div>", 
        unsafe_allow_html=True
    )