"""File Processing Agent for handling audio file validation and chunking"""

import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import hashlib
from pathlib import Path
from pydub import AudioSegment
import logging

from .base_agent import BaseAgent, Message, MessageType
from ..config import MAX_FILE_SIZE_MB, CHUNK_DURATION_MS


class FileProcessingAgent(BaseAgent):
    """Agent responsible for file validation, processing, and chunking"""
    
    def __init__(self, name: str = "FileProcessor"):
        super().__init__(name)
        self._capabilities = [
            "validate_file",
            "process_audio",
            "chunk_audio",
            "calculate_hash",
            "get_file_metadata"
        ]
        self._temp_files: List[str] = []
        
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities"""
        return self._capabilities
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process file-related messages"""
        try:
            action = message.content.get("action")
            
            if action == "validate_file":
                return await self._handle_validate_file(message)
            elif action == "process_audio":
                return await self._handle_process_audio(message)
            elif action == "chunk_audio":
                return await self._handle_chunk_audio(message)
            elif action == "calculate_hash":
                return await self._handle_calculate_hash(message)
            elif action == "get_file_metadata":
                return await self._handle_get_metadata(message)
            elif action == "cleanup":
                return await self._handle_cleanup(message)
            else:
                return message.reply(
                    {"error": f"Unknown action: {action}"},
                    MessageType.ERROR
                )
                
        except Exception as e:
            self.logger.error(f"Error in FileProcessingAgent: {e}")
            return message.reply(
                {"error": str(e), "details": "File processing failed"},
                MessageType.ERROR
            )
    
    async def _handle_validate_file(self, message: Message) -> Message:
        """Validate uploaded file"""
        file_data = message.content.get("file")
        filename = message.content.get("filename", "")
        
        if not file_data:
            return message.reply(
                {"error": "No file data provided"},
                MessageType.ERROR
            )
        
        # Check file extension
        valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.webm']
        file_ext = Path(filename).suffix.lower()
        
        if file_ext not in valid_extensions:
            return message.reply(
                {
                    "valid": False,
                    "error": f"Unsupported file type: {file_ext}",
                    "supported_types": valid_extensions
                }
            )
        
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return message.reply(
                {
                    "valid": False,
                    "error": f"File too large: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)",
                    "size_mb": file_size_mb
                }
            )
        
        return message.reply({
            "valid": True,
            "filename": filename,
            "size_mb": file_size_mb,
            "extension": file_ext
        })
    
    async def _handle_process_audio(self, message: Message) -> Message:
        """Process audio file and prepare for transcription"""
        file_data = message.content.get("file")
        filename = message.content.get("filename", "audio_file")
        
        if not file_data:
            return message.reply(
                {"error": "No file data provided"},
                MessageType.ERROR
            )
        
        try:
            # Save file temporarily
            temp_path = self._save_uploaded_file(file_data, filename)
            self._temp_files.append(temp_path)
            
            # Get audio metadata
            duration = self._get_audio_duration(temp_path)
            file_hash = self._calculate_file_hash(file_data)
            
            # Determine if chunking is needed
            needs_chunking = len(file_data) > 20 * 1024 * 1024  # 20MB threshold
            
            result = {
                "success": True,
                "temp_path": temp_path,
                "duration_seconds": duration,
                "file_hash": file_hash,
                "needs_chunking": needs_chunking,
                "original_filename": filename
            }
            
            return message.reply(result)
            
        except Exception as e:
            return message.reply(
                {"error": f"Failed to process audio: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_chunk_audio(self, message: Message) -> Message:
        """Split audio file into chunks"""
        file_path = message.content.get("file_path")
        chunk_duration = message.content.get("chunk_duration_ms", CHUNK_DURATION_MS)
        
        if not file_path or not os.path.exists(file_path):
            return message.reply(
                {"error": "Invalid file path"},
                MessageType.ERROR
            )
        
        try:
            # Create chunks
            chunk_paths = self._chunk_audio(file_path, chunk_duration)
            self._temp_files.extend(chunk_paths)
            
            # Get chunk metadata
            chunks_info = []
            for i, chunk_path in enumerate(chunk_paths):
                chunk_size = os.path.getsize(chunk_path)
                chunk_duration_sec = self._get_audio_duration(chunk_path)
                
                chunks_info.append({
                    "index": i,
                    "path": chunk_path,
                    "size_bytes": chunk_size,
                    "duration_seconds": chunk_duration_sec,
                    "start_time": i * (chunk_duration_ms / 1000)
                })
            
            return message.reply({
                "success": True,
                "chunk_count": len(chunks_info),
                "chunks": chunks_info,
                "total_duration": sum(c["duration_seconds"] for c in chunks_info)
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Failed to chunk audio: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_calculate_hash(self, message: Message) -> Message:
        """Calculate file hash for caching"""
        file_data = message.content.get("file")
        
        if not file_data:
            return message.reply(
                {"error": "No file data provided"},
                MessageType.ERROR
            )
        
        file_hash = self._calculate_file_hash(file_data)
        
        return message.reply({
            "hash": file_hash,
            "algorithm": "sha256"
        })
    
    async def _handle_get_metadata(self, message: Message) -> Message:
        """Get detailed file metadata"""
        file_path = message.content.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            return message.reply(
                {"error": "Invalid file path"},
                MessageType.ERROR
            )
        
        try:
            from pydub import AudioSegment
            
            audio = AudioSegment.from_file(file_path)
            
            metadata = {
                "duration_seconds": len(audio) / 1000.0,
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "bitrate": audio.frame_rate * audio.frame_width * 8 * audio.channels,
                "format": Path(file_path).suffix[1:],
                "size_bytes": os.path.getsize(file_path)
            }
            
            return message.reply(metadata)
            
        except Exception as e:
            return message.reply(
                {"error": f"Failed to get metadata: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_cleanup(self, message: Message) -> Message:
        """Clean up temporary files"""
        try:
            cleaned_count = 0
            failed_count = 0
            
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {temp_file}: {e}")
                    failed_count += 1
            
            self._temp_files.clear()
            
            return message.reply({
                "success": True,
                "cleaned_files": cleaned_count,
                "failed_files": failed_count
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Cleanup failed: {str(e)}"},
                MessageType.ERROR
            )
    
    def _calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate SHA256 hash of file data"""
        return hashlib.sha256(file_data).hexdigest()
    
    def _save_uploaded_file(self, file_data: bytes, filename: str) -> str:
        """Save uploaded file to temporary location"""
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        
        # Generate unique filename
        unique_filename = f"{hashlib.sha256(filename.encode()).hexdigest()[:16]}_{filename}"
        temp_path = os.path.join(temp_dir, unique_filename)
        
        # Write file data
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        
        return temp_path
    
    def _get_audio_duration(self, file_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            self.logger.warning(f"Failed to get audio duration: {e}")
            return 0.0
    
    def _chunk_audio(self, file_path: str, chunk_duration_ms: int) -> List[str]:
        """Split audio file into chunks"""
        try:
            audio = AudioSegment.from_file(file_path)
            chunks = []
            chunk_paths = []
            
            # Split audio into chunks
            for i in range(0, len(audio), chunk_duration_ms):
                chunk = audio[i:i + chunk_duration_ms]
                
                # Save chunk to temp file
                chunk_filename = f"chunk_{i // chunk_duration_ms}.mp3"
                chunk_path = os.path.join(tempfile.gettempdir(), chunk_filename)
                
                # Export with consistent parameters
                chunk.export(
                    chunk_path,
                    format="mp3",
                    bitrate="128k",
                    parameters=["-ac", "1"]  # Convert to mono
                )
                
                chunk_paths.append(chunk_path)
            
            return chunk_paths
            
        except Exception as e:
            self.logger.error(f"Failed to chunk audio: {e}")
            return []
    
    async def stop(self):
        """Clean up before stopping"""
        # Clean up any remaining temp files
        await self._handle_cleanup(Message(content={"action": "cleanup"}))
        await super().stop()