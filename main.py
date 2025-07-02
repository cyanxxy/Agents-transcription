import streamlit as st
import logging
import os
import asyncio
from typing import Optional
from datetime import datetime

# Import from new module structure
from api_client import initialize_gemini
from ui_components import (
    render_model_selection,
    render_context_inputs,
    render_file_upload,
    render_transcript_tabs,
    render_footer
)
from styles import apply_custom_styles
from app_setup import setup_logging, setup_streamlit_page
from state_manager import initialize_state

# Import agent system
from agents.workflow_coordinator import WorkflowCoordinator, StreamlitAgentInterface

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)


def check_password() -> bool:
    """Returns `True` if the user had the correct password."""
    # Initialize password_correct in session state if it doesn't exist
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # Only show input if password is not correct
    if not st.session_state["password_correct"]:
        # Password input
        password = st.text_input("Password", type="password")
        
        # Check password
        if password:
            # Get password from secrets
            correct_password = st.secrets.get("app_password", "")
            
            if not correct_password:
                st.error("No password configured in secrets. Please set a password.")
                return False
            
            if password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect")
                return False
        else:
            st.warning("Please enter a password.")
            return False
    
    return True


async def handle_transcription_with_agents(client, model_name: str, uploaded_file, metadata: dict, num_speakers: int) -> None:
    """Handle transcription using the agent-based workflow."""
    st.session_state.processing_status = "processing"
    st.session_state.current_file_name = uploaded_file.name
    
    # Initialize agent interface
    agent_interface = StreamlitAgentInterface()
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        quality_container = st.empty()
        
        # Progress callback
        def update_progress(message, progress):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Read file data
            uploaded_file.seek(0)  # Reset file pointer
            file_data = uploaded_file.read()
            
            # Get coordinator
            coordinator = await agent_interface.initialize()
            
            # Process with agents
            result = await coordinator.process_audio_file(
                file_data=file_data,
                filename=uploaded_file.name,
                model_name=model_name,
                custom_prompt=None,  # Could be added from metadata
                progress_callback=update_progress
            )
            
            if result["success"]:
                # Update session state with results
                st.session_state.transcript_text = result["transcript"]
                st.session_state.edited_transcript = result["transcript"]
                st.session_state.transcript_editor_content = result["transcript"]
                st.session_state.processing_status = "complete"
                
                # Store quality metrics
                st.session_state.quality_score = result.get("quality_score", 0)
                st.session_state.quality_assessment = result.get("quality_assessment", "Unknown")
                st.session_state.quality_metrics = result.get("quality_metrics", {})
                st.session_state.processing_time = result.get("processing_time", 0)
                
                # Display quality assessment
                with quality_container:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Quality Score", f"{result['quality_score']:.1f}/100")
                    with col2:
                        st.metric("Assessment", result['quality_assessment'])
                    with col3:
                        st.metric("Processing Time", f"{result['processing_time']:.1f}s")
                
                logger.info(f"Transcription successful for file: {uploaded_file.name}")
                
                # Wait a moment before rerun to show metrics
                await asyncio.sleep(2)
                st.rerun()
            else:
                # Handle error
                handle_transcription_error(", ".join(result["errors"]), uploaded_file.name)
                
        except Exception as e:
            # Handle unexpected error
            handle_transcription_error(str(e), uploaded_file.name, unexpected=True)
        finally:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()


def handle_transcription_error(error_message: str, filename: str, unexpected: bool = False) -> None:
    """Handle transcription errors consistently."""
    if unexpected:
        user_message = "An unexpected error occurred during transcription."
        logger.error(f"Unexpected transcription error for {filename}: {error_message}", exc_info=True)
    else:
        user_message = error_message
        logger.error(f"Transcription failed for {filename}: {error_message}")
    
    st.error(f"Transcription failed: {user_message}")
    st.session_state.processing_status = "error"
    st.session_state.error_message = user_message
    st.rerun()


def render_quality_insights():
    """Render quality insights panel"""
    if hasattr(st.session_state, 'quality_metrics') and st.session_state.quality_metrics:
        with st.expander("📊 Transcript Quality Insights", expanded=True):
            metrics = st.session_state.quality_metrics
            
            # Quality score visualization
            score = st.session_state.get('quality_score', 0)
            assessment = st.session_state.get('quality_assessment', 'Unknown')
            
            # Create a colored progress bar based on score
            if score >= 80:
                color = "green"
            elif score >= 60:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
                <div style='margin-bottom: 20px;'>
                    <h4>Overall Quality: {assessment} ({score:.0f}/100)</h4>
                    <div style='background-color: #f0f0f0; border-radius: 10px; height: 20px;'>
                        <div style='background-color: {color}; width: {score}%; height: 100%; 
                                    border-radius: 10px; transition: width 0.5s;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Readability", f"{metrics.get('readability_score', 0):.0f}/100")
                st.metric("Sentence Variety", f"{metrics.get('sentence_variety', 0):.0f}/100")
                st.metric("Avg Sentence Length", f"{metrics.get('average_sentence_length', 0):.1f} words")
            
            with col2:
                st.metric("Vocabulary Richness", f"{metrics.get('vocabulary_richness', 0):.0f}/100")
                st.metric("Punctuation Density", f"{metrics.get('punctuation_density', 0):.2%}")
                st.metric("Timestamp Coverage", f"{metrics.get('timestamp_coverage', 0):.0f}%")


async def handle_smart_editing(action: str, **kwargs):
    """Handle smart editing actions using agents"""
    agent_interface = StreamlitAgentInterface()
    
    with st.spinner(f"Performing {action}..."):
        result = await agent_interface.edit_with_agent(
            transcript=st.session_state.edited_transcript,
            action=action,
            **kwargs
        )
        
        if result.get("success"):
            # Update transcript
            if "new_transcript" in result:
                st.session_state.edited_transcript = result["new_transcript"]
                st.session_state.transcript_editor_content = result["new_transcript"]
            elif "formatted_transcript" in result:
                st.session_state.edited_transcript = result["formatted_transcript"]
                st.session_state.transcript_editor_content = result["formatted_transcript"]
            
            # Show success message
            if "changes_applied" in result:
                st.success(f"Applied: {', '.join(result['changes_applied'])}")
            elif "replacement_count" in result:
                st.success(f"Replaced {result['replacement_count']} occurrences")
            
            st.rerun()
        else:
            st.error(result.get("error", "Operation failed"))


def render_editing_tools():
    """Render advanced editing tools powered by agents"""
    with st.expander("🛠️ Smart Editing Tools", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs(["Find & Replace", "Auto Format", "Quality Check", "History"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                find_text = st.text_input("Find text:")
                replace_text = st.text_input("Replace with:")
            with col2:
                case_sensitive = st.checkbox("Case sensitive")
                whole_word = st.checkbox("Whole words only")
            
            if st.button("Replace All", type="primary"):
                asyncio.run(handle_smart_editing(
                    "replace_text",
                    find=find_text,
                    replace=replace_text,
                    case_sensitive=case_sensitive,
                    whole_word=whole_word
                ))
        
        with tab2:
            st.markdown("**Auto-formatting Options:**")
            fix_caps = st.checkbox("Fix sentence capitalization", value=True)
            fix_punct = st.checkbox("Fix punctuation spacing", value=True)
            remove_filler = st.checkbox("Remove filler words (um, uh, like)")
            fix_numbers = st.checkbox("Standardize numbers")
            
            if st.button("Apply Auto-Format", type="primary"):
                asyncio.run(handle_smart_editing(
                    "auto_format",
                    options={
                        "fix_capitalization": fix_caps,
                        "fix_punctuation_spacing": fix_punct,
                        "remove_filler_words": remove_filler,
                        "standardize_numbers": fix_numbers
                    }
                ))
        
        with tab3:
            if st.button("Run Quality Check"):
                agent_interface = StreamlitAgentInterface()
                coordinator = asyncio.run(agent_interface.initialize())
                
                with st.spinner("Analyzing transcript quality..."):
                    from agents.base_agent import Message, MessageType
                    errors_result = asyncio.run(coordinator.supervisor.route_message(
                        Message(
                            sender="ui",
                            recipient="QualityAssurance",
                            type=MessageType.REQUEST,
                            content={
                                "action": "detect_errors",
                                "transcript": st.session_state.edited_transcript
                            }
                        )
                    ))
                    
                    if errors_result and errors_result.content.get("error_count", 0) > 0:
                        st.warning(f"Found {errors_result.content['error_count']} potential issues")
                        
                        for error in errors_result.content.get("errors", [])[:10]:
                            with st.container():
                                st.markdown(f"**{error['message']}**")
                                st.text(f"Position: {error.get('position', 'Unknown')}")
                                if error.get("suggestion"):
                                    st.info(f"Suggestion: {error['suggestion']}")
                    else:
                        st.success("No issues found!")
        
        with tab4:
            if st.button("Show Edit History"):
                agent_interface = StreamlitAgentInterface()
                history_result = asyncio.run(agent_interface.edit_with_agent(
                    transcript="",
                    action="get_history",
                    limit=10
                ))
                
                if history_result.get("history"):
                    for edit in history_result["history"]:
                        st.text(f"{edit['timestamp']}: {edit['action']}")


def main():
    """Main application entry point."""
    logger.info("Application started/restarted.")
    
    # Setup page configuration
    setup_streamlit_page()
    apply_custom_styles()
    
    # Initialize session state
    initialize_state()
    
    logger.info(f"Initial state: processing_status={st.session_state.processing_status}, "
               f"current_file_name={st.session_state.current_file_name}")
    
    # Check password first
    if not check_password():
        st.stop()
    
    # Main app layout
    st.title("🎙️ ExactTranscriber")
    st.markdown("<p class='subtitle'>Precision audio transcription powered by Gemini AI and Smart Agents</p>", 
                unsafe_allow_html=True)
    
    # Check API initialization
    client, error, _ = initialize_gemini()
    if error:
        st.error(f"⚠️ {error}")
        st.info("Please check your API key configuration and try again.")
        st.stop()
    
    # Render components based on state
    if st.session_state.processing_status == "complete" and st.session_state.transcript_text:
        # Show quality insights
        render_quality_insights()
        
        # Show editing tools
        render_editing_tools()
        
        # Show transcript editor
        render_transcript_tabs()
        
        # Reset button
        if st.button("🔄 Transcribe New File", type="secondary"):
            # Reset all state variables
            for key in ['transcript_text', 'edited_transcript', 'current_file_name', 
                       'quality_score', 'quality_metrics']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.processing_status = "idle"
            st.rerun()
    
    elif st.session_state.processing_status == "processing":
        # Show processing status
        st.info(f"Processing: {st.session_state.current_file_name}")
        st.markdown("Please wait while your audio is being transcribed...")
    
    elif st.session_state.processing_status == "error":
        # Show error state
        st.error(f"Error: {st.session_state.get('error_message', 'Unknown error')}")
        if st.button("Try Again"):
            st.session_state.processing_status = "idle"
            st.rerun()
    
    else:
        # Show upload interface
        model_name = render_model_selection()
        metadata = render_context_inputs()
        uploaded_file = render_file_upload()
        
        # Adjust number of speakers
        num_speakers = st.number_input(
            "Number of Speakers",
            min_value=1,
            max_value=10,
            value=st.session_state.num_speakers_input,
            help="Specify the number of distinct speakers in the audio"
        )
        st.session_state.num_speakers_input = num_speakers
        
        # Process uploaded file
        if uploaded_file and st.button("🚀 Transcribe Audio", type="primary"):
            asyncio.run(handle_transcription_with_agents(
                client, model_name, uploaded_file, metadata, num_speakers
            ))
    
    # Footer
    render_footer()


if __name__ == "__main__":
    main()