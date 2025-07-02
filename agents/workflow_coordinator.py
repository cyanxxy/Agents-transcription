"""Workflow Coordinator for orchestrating agent-based transcription"""

import asyncio
from typing import Dict, Any, Optional, List
import streamlit as st
from datetime import datetime

from .base_agent import AgentSupervisor, Message, MessageType
from .file_processing_agent import FileProcessingAgent
from .transcription_agent import TranscriptionAgent
from .quality_assurance_agent import QualityAssuranceAgent
from .editing_assistant_agent import EditingAssistantAgent
from .export_agent import ExportAgent


class WorkflowCoordinator:
    """Coordinates the entire transcription workflow using agents"""
    
    def __init__(self):
        self.supervisor = AgentSupervisor()
        self._setup_agents()
        self._running = False
        
    def _setup_agents(self):
        """Initialize and register all agents"""
        # Create agents
        self.file_agent = FileProcessingAgent()
        self.transcription_agent = TranscriptionAgent()
        self.qa_agent = QualityAssuranceAgent()
        self.editing_agent = EditingAssistantAgent()
        self.export_agent = ExportAgent()
        
        # Register with supervisor
        self.supervisor.register_agent(self.file_agent)
        self.supervisor.register_agent(self.transcription_agent)
        self.supervisor.register_agent(self.qa_agent)
        self.supervisor.register_agent(self.editing_agent)
        self.supervisor.register_agent(self.export_agent)
    
    async def start(self):
        """Start all agents"""
        if not self._running:
            self._running = True
            # Start agents in background
            asyncio.create_task(self.supervisor.start_all())
            await asyncio.sleep(0.1)  # Give agents time to start
    
    async def stop(self):
        """Stop all agents"""
        if self._running:
            await self.supervisor.stop_all()
            self._running = False
    
    async def process_audio_file(self, 
                               file_data: bytes, 
                               filename: str,
                               model_name: str,
                               custom_prompt: Optional[str] = None,
                               progress_callback=None) -> Dict[str, Any]:
        """Process audio file through the complete workflow"""
        
        result = {
            "success": False,
            "transcript": "",
            "errors": [],
            "processing_time": 0,
            "quality_score": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Validate and process file
            if progress_callback:
                progress_callback("Validating file...", 0.1)
            
            validation_response = await self.supervisor.route_message(Message(
                sender="coordinator",
                recipient="FileProcessor",
                type=MessageType.REQUEST,
                content={
                    "action": "validate_file",
                    "file": file_data,
                    "filename": filename
                }
            ))
            
            if not validation_response or not validation_response.content.get("valid"):
                result["errors"].append(validation_response.content.get("error", "File validation failed"))
                return result
            
            # Step 2: Process audio file
            if progress_callback:
                progress_callback("Processing audio file...", 0.2)
            
            process_response = await self.supervisor.route_message(Message(
                sender="coordinator",
                recipient="FileProcessor",
                type=MessageType.REQUEST,
                content={
                    "action": "process_audio",
                    "file": file_data,
                    "filename": filename
                }
            ))
            
            if not process_response or not process_response.content.get("success"):
                result["errors"].append("Failed to process audio file")
                return result
            
            file_info = process_response.content
            
            # Step 3: Determine if chunking is needed
            if file_info.get("needs_chunking"):
                if progress_callback:
                    progress_callback("Splitting audio into chunks...", 0.3)
                
                # Chunk the audio
                chunk_response = await self.supervisor.route_message(Message(
                    sender="coordinator",
                    recipient="FileProcessor",
                    type=MessageType.REQUEST,
                    content={
                        "action": "chunk_audio",
                        "file_path": file_info["temp_path"]
                    }
                ))
                
                if not chunk_response or not chunk_response.content.get("success"):
                    result["errors"].append("Failed to chunk audio")
                    return result
                
                # Transcribe chunks
                if progress_callback:
                    progress_callback("Transcribing audio chunks...", 0.4)
                
                transcribe_response = await self.supervisor.route_message(Message(
                    sender="coordinator",
                    recipient="Transcriber",
                    type=MessageType.REQUEST,
                    content={
                        "action": "transcribe_chunks",
                        "chunks": chunk_response.content["chunks"],
                        "model": model_name,
                        "custom_prompt": custom_prompt
                    }
                ))
                
                if not transcribe_response or not transcribe_response.content.get("success"):
                    result["errors"].append("Failed to transcribe chunks")
                    
                    # Try to retry failed chunks
                    if transcribe_response and transcribe_response.content.get("failed_chunks"):
                        if progress_callback:
                            progress_callback("Retrying failed chunks...", 0.6)
                        
                        retry_response = await self.supervisor.route_message(Message(
                            sender="coordinator",
                            recipient="Transcriber",
                            type=MessageType.REQUEST,
                            content={
                                "action": "retry_failed",
                                "failed_chunks": transcribe_response.content["failed_chunks"],
                                "model": model_name,
                                "custom_prompt": custom_prompt
                            }
                        ))
                
                # Merge transcripts
                if progress_callback:
                    progress_callback("Merging transcripts...", 0.7)
                
                merge_response = await self.supervisor.route_message(Message(
                    sender="coordinator",
                    recipient="Transcriber",
                    type=MessageType.REQUEST,
                    content={
                        "action": "merge_transcripts",
                        "transcripts": transcribe_response.content.get("transcripts", [])
                    }
                ))
                
                if merge_response and merge_response.content.get("success"):
                    result["transcript"] = merge_response.content["merged_transcript"]
                
            else:
                # Single file transcription
                if progress_callback:
                    progress_callback("Transcribing audio...", 0.4)
                
                transcribe_response = await self.supervisor.route_message(Message(
                    sender="coordinator",
                    recipient="Transcriber",
                    type=MessageType.REQUEST,
                    content={
                        "action": "transcribe_single",
                        "file_path": file_info["temp_path"],
                        "model": model_name,
                        "file_hash": file_info.get("file_hash"),
                        "custom_prompt": custom_prompt
                    }
                ))
                
                if transcribe_response and transcribe_response.content.get("success"):
                    result["transcript"] = transcribe_response.content["transcript"]
                else:
                    result["errors"].append("Transcription failed")
                    return result
            
            # Step 4: Quality check
            if progress_callback:
                progress_callback("Checking transcript quality...", 0.8)
            
            qa_response = await self.supervisor.route_message(Message(
                sender="coordinator",
                recipient="QualityAssurance",
                type=MessageType.REQUEST,
                content={
                    "action": "analyze_quality",
                    "transcript": result["transcript"]
                }
            ))
            
            if qa_response:
                result["quality_score"] = qa_response.content.get("quality_score", 0)
                result["quality_assessment"] = qa_response.content.get("assessment", "Unknown")
                result["quality_metrics"] = qa_response.content.get("metrics", {})
            
            # Step 5: Auto-fix common errors if quality is low
            if result["quality_score"] < 70:
                if progress_callback:
                    progress_callback("Fixing common errors...", 0.9)
                
                fix_response = await self.supervisor.route_message(Message(
                    sender="coordinator",
                    recipient="QualityAssurance",
                    type=MessageType.REQUEST,
                    content={
                        "action": "fix_common_errors",
                        "transcript": result["transcript"]
                    }
                ))
                
                if fix_response and fix_response.content.get("success"):
                    result["transcript"] = fix_response.content["fixed_transcript"]
                    result["fixes_applied"] = fix_response.content.get("fixes_applied", [])
            
            # Step 6: Cleanup
            await self.supervisor.route_message(Message(
                sender="coordinator",
                recipient="FileProcessor",
                type=MessageType.REQUEST,
                content={"action": "cleanup"}
            ))
            
            result["success"] = True
            result["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            if progress_callback:
                progress_callback("Processing complete!", 1.0)
            
        except Exception as e:
            result["errors"].append(f"Workflow error: {str(e)}")
        
        return result
    
    async def edit_transcript(self,
                            transcript: str,
                            edit_action: str,
                            parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transcript editing operations"""
        
        response = await self.supervisor.route_message(Message(
            sender="coordinator",
            recipient="EditingAssistant",
            type=MessageType.REQUEST,
            content={
                "action": edit_action,
                "transcript": transcript,
                **parameters
            }
        ))
        
        if response:
            return response.content
        else:
            return {"error": "Edit operation failed"}
    
    async def export_transcript(self,
                              transcript: str,
                              format_type: str,
                              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export transcript to specified format"""
        
        response = await self.supervisor.route_message(Message(
            sender="coordinator",
            recipient="Exporter",
            type=MessageType.REQUEST,
            content={
                "action": "export",
                "transcript": transcript,
                "format": format_type,
                "options": options or {}
            }
        ))
        
        if response:
            return response.content
        else:
            return {"error": "Export operation failed"}
    
    async def get_quality_suggestions(self, transcript: str) -> Dict[str, Any]:
        """Get quality improvement suggestions"""
        
        response = await self.supervisor.route_message(Message(
            sender="coordinator",
            recipient="QualityAssurance",
            type=MessageType.REQUEST,
            content={
                "action": "suggest_corrections",
                "transcript": transcript,
                "errors": []  # Would be populated by detect_errors first
            }
        ))
        
        if response:
            return response.content
        else:
            return {"error": "Failed to get suggestions"}


# Streamlit integration helper
class StreamlitAgentInterface:
    """Helper class for integrating agents with Streamlit UI"""
    
    def __init__(self):
        if 'workflow_coordinator' not in st.session_state:
            st.session_state.workflow_coordinator = None
        
    async def initialize(self):
        """Initialize the workflow coordinator"""
        if st.session_state.workflow_coordinator is None:
            coordinator = WorkflowCoordinator()
            await coordinator.start()
            st.session_state.workflow_coordinator = coordinator
        
        return st.session_state.workflow_coordinator
    
    def progress_callback(self, message: str, progress: float):
        """Update progress in Streamlit"""
        if hasattr(st, 'progress_bar'):
            st.progress_bar.progress(progress)
        if hasattr(st, 'status_text'):
            st.status_text.text(message)
    
    async def process_file_with_agents(self,
                                     uploaded_file,
                                     model_name: str,
                                     custom_prompt: Optional[str] = None):
        """Process uploaded file using agent workflow"""
        
        coordinator = await self.initialize()
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Set up progress callback
        def update_progress(message, progress):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Process file
        result = await coordinator.process_audio_file(
            file_data=uploaded_file.read(),
            filename=uploaded_file.name,
            model_name=model_name,
            custom_prompt=custom_prompt,
            progress_callback=update_progress
        )
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return result
    
    async def edit_with_agent(self,
                            transcript: str,
                            action: str,
                            **kwargs):
        """Edit transcript using agent"""
        coordinator = await self.initialize()
        
        return await coordinator.edit_transcript(
            transcript=transcript,
            edit_action=action,
            parameters=kwargs
        )
    
    async def export_with_agent(self,
                              transcript: str,
                              format_type: str,
                              **options):
        """Export transcript using agent"""
        coordinator = await self.initialize()
        
        return await coordinator.export_transcript(
            transcript=transcript,
            format_type=format_type,
            options=options
        )


# Example usage in main.py
def integrate_agents_example():
    """Example of how to integrate agents in main.py"""
    
    # In your Streamlit app
    agent_interface = StreamlitAgentInterface()
    
    # When processing a file
    if st.button("Transcribe with Agents"):
        result = asyncio.run(
            agent_interface.process_file_with_agents(
                uploaded_file=st.file_uploader("Upload audio"),
                model_name="gemini-2.0-flash-exp",
                custom_prompt="Transcribe with speaker labels"
            )
        )
        
        if result["success"]:
            st.success(f"Quality Score: {result['quality_score']}")
            st.text_area("Transcript", result["transcript"])
        else:
            st.error(f"Errors: {result['errors']}")
    
    # For editing
    if st.button("Smart Replace"):
        result = asyncio.run(
            agent_interface.edit_with_agent(
                transcript=st.session_state.transcript,
                action="smart_replace",
                replacements=[{"old": "um", "new": "", "position": 100}]
            )
        )
    
    # For export
    format_type = st.selectbox("Export Format", ["txt", "srt", "vtt", "json"])
    if st.button("Export"):
        result = asyncio.run(
            agent_interface.export_with_agent(
                transcript=st.session_state.transcript,
                format_type=format_type,
                include_timestamps=True
            )
        )