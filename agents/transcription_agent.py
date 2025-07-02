"""Transcription Agent for orchestrating Gemini API calls"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

from .base_agent import BaseAgent, Message, MessageType
from ..api_client import initialize_gemini, get_transcription_prompt
from ..config import MAX_RETRIES, RETRY_DELAY, GEMINI_MODELS
import google.generativeai as genai


class TranscriptionAgent(BaseAgent):
    """Agent responsible for managing transcription via Gemini API"""
    
    def __init__(self, name: str = "Transcriber"):
        super().__init__(name)
        self._capabilities = [
            "transcribe_single",
            "transcribe_chunks",
            "merge_transcripts",
            "retry_failed",
            "check_api_status"
        ]
        self.api_client = None  # Will be initialized on first use
        self._cache: Dict[str, Dict[str, Any]] = {}  # Simple in-memory cache
        
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities"""
        return self._capabilities
    
    def _ensure_api_client(self, model_name: str) -> bool:
        """Ensure API client is initialized"""
        if self.api_client is None:
            client, error, model_id = initialize_gemini(model_name)
            if error:
                self.logger.error(f"Failed to initialize Gemini API: {error}")
                return False
            self.api_client = client
            self.model_id = model_id
        return True
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process transcription-related messages"""
        try:
            action = message.content.get("action")
            
            if action == "transcribe_single":
                return await self._handle_transcribe_single(message)
            elif action == "transcribe_chunks":
                return await self._handle_transcribe_chunks(message)
            elif action == "merge_transcripts":
                return await self._handle_merge_transcripts(message)
            elif action == "retry_failed":
                return await self._handle_retry_failed(message)
            elif action == "check_api_status":
                return await self._handle_check_api_status(message)
            elif action == "get_cached":
                return await self._handle_get_cached(message)
            else:
                return message.reply(
                    {"error": f"Unknown action: {action}"},
                    MessageType.ERROR
                )
                
        except Exception as e:
            self.logger.error(f"Error in TranscriptionAgent: {e}")
            return message.reply(
                {"error": str(e), "details": "Transcription failed"},
                MessageType.ERROR
            )
    
    async def _handle_transcribe_single(self, message: Message) -> Message:
        """Handle single file transcription"""
        file_path = message.content.get("file_path")
        model_name = message.content.get("model", "gemini-2.0-flash-exp")
        file_hash = message.content.get("file_hash")
        custom_prompt = message.content.get("custom_prompt")
        
        if not file_path:
            return message.reply(
                {"error": "No file path provided"},
                MessageType.ERROR
            )
        
        # Check cache first
        if file_hash and file_hash in self._cache:
            cached_result = self._cache[file_hash]
            if cached_result.get("model") == model_name:
                self.logger.info(f"Returning cached transcription for {file_hash}")
                return message.reply({
                    "success": True,
                    "transcript": cached_result["transcript"],
                    "model": model_name,
                    "cached": True,
                    "processing_time": 0
                })
        
        # Ensure API client is initialized
        if not self._ensure_api_client(model_name):
            return message.reply(
                {"error": "Failed to initialize Gemini API"},
                MessageType.ERROR
            )
        
        try:
            start_time = datetime.now()
            
            # Upload file and transcribe
            transcript = await self._transcribe_with_retry(
                file_path, 
                model_name,
                custom_prompt
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Cache the result
            if file_hash:
                self._cache[file_hash] = {
                    "transcript": transcript,
                    "model": model_name,
                    "timestamp": datetime.now()
                }
            
            # Send progress update
            if self.supervisor:
                await self.supervisor.route_message(Message(
                    sender=self.name,
                    recipient="*",
                    type=MessageType.STATUS,
                    content={
                        "status": "transcription_complete",
                        "file": file_path,
                        "processing_time": processing_time
                    }
                ))
            
            return message.reply({
                "success": True,
                "transcript": transcript,
                "model": model_name,
                "cached": False,
                "processing_time": processing_time
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Transcription failed: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_transcribe_chunks(self, message: Message) -> Message:
        """Handle multi-chunk transcription"""
        chunks = message.content.get("chunks", [])
        model_name = message.content.get("model", "gemini-2.0-flash-exp")
        custom_prompt = message.content.get("custom_prompt")
        max_workers = message.content.get("max_workers", 3)
        
        if not chunks:
            return message.reply(
                {"error": "No chunks provided"},
                MessageType.ERROR
            )
        
        try:
            start_time = datetime.now()
            transcripts = []
            failed_chunks = []
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(
                        self._transcribe_chunk_sync,
                        chunk["path"],
                        model_name,
                        chunk["index"],
                        custom_prompt
                    ): chunk
                    for chunk in chunks
                }
                
                # Process completed transcriptions
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        result = future.result()
                        transcripts.append(result)
                        
                        # Send progress update
                        if self.supervisor:
                            await self.supervisor.route_message(Message(
                                sender=self.name,
                                recipient="*",
                                type=MessageType.STATUS,
                                content={
                                    "status": "chunk_complete",
                                    "chunk_index": chunk["index"],
                                    "total_chunks": len(chunks),
                                    "progress": len(transcripts) / len(chunks)
                                }
                            ))
                            
                    except Exception as e:
                        self.logger.error(f"Chunk {chunk['index']} failed: {e}")
                        failed_chunks.append({
                            "chunk": chunk,
                            "error": str(e)
                        })
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Sort transcripts by chunk index
            transcripts.sort(key=lambda x: x["chunk_index"])
            
            return message.reply({
                "success": len(failed_chunks) == 0,
                "transcripts": transcripts,
                "failed_chunks": failed_chunks,
                "total_chunks": len(chunks),
                "successful_chunks": len(transcripts),
                "processing_time": processing_time,
                "model": model_name
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Chunk transcription failed: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_merge_transcripts(self, message: Message) -> Message:
        """Merge multiple transcript chunks into one"""
        transcripts = message.content.get("transcripts", [])
        
        if not transcripts:
            return message.reply(
                {"error": "No transcripts to merge"},
                MessageType.ERROR
            )
        
        try:
            # Sort by chunk index to ensure correct order
            sorted_transcripts = sorted(transcripts, key=lambda x: x.get("chunk_index", 0))
            
            # Merge transcripts
            merged_text = ""
            current_timestamp = 0.0
            
            for transcript in sorted_transcripts:
                text = transcript.get("text", "")
                chunk_start_time = transcript.get("start_time", 0)
                
                # Add chunk text with adjusted timestamps if needed
                if text:
                    # If transcript has timestamps, adjust them
                    if "[" in text and "]" in text:
                        adjusted_text = self._adjust_timestamps(text, chunk_start_time)
                        merged_text += adjusted_text + "\n\n"
                    else:
                        merged_text += text + "\n\n"
            
            # Clean up the merged transcript
            merged_text = merged_text.strip()
            
            return message.reply({
                "success": True,
                "merged_transcript": merged_text,
                "chunk_count": len(transcripts)
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Failed to merge transcripts: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_retry_failed(self, message: Message) -> Message:
        """Retry failed chunk transcriptions"""
        failed_chunks = message.content.get("failed_chunks", [])
        model_name = message.content.get("model", "gemini-2.0-flash-exp")
        custom_prompt = message.content.get("custom_prompt")
        
        if not failed_chunks:
            return message.reply({
                "success": True,
                "message": "No failed chunks to retry"
            })
        
        try:
            retried_transcripts = []
            still_failed = []
            
            for failed_item in failed_chunks:
                chunk = failed_item["chunk"]
                try:
                    # Retry with exponential backoff
                    result = await self._transcribe_with_retry(
                        chunk["path"],
                        model_name,
                        custom_prompt,
                        max_retries=2  # Fewer retries for already failed chunks
                    )
                    
                    retried_transcripts.append({
                        "chunk_index": chunk["index"],
                        "text": result,
                        "start_time": chunk.get("start_time", 0)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Retry failed for chunk {chunk['index']}: {e}")
                    still_failed.append({
                        "chunk": chunk,
                        "error": str(e)
                    })
            
            return message.reply({
                "success": len(still_failed) == 0,
                "retried_transcripts": retried_transcripts,
                "still_failed": still_failed,
                "retry_success_count": len(retried_transcripts)
            })
            
        except Exception as e:
            return message.reply(
                {"error": f"Retry operation failed: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_check_api_status(self, message: Message) -> Message:
        """Check Gemini API status"""
        try:
            # Simple health check - try to list models
            models = GEMINI_MODELS
            
            return message.reply({
                "status": "healthy",
                "available_models": list(models.keys()),
                "api_version": "v1beta"
            })
            
        except Exception as e:
            return message.reply({
                "status": "unhealthy",
                "error": str(e)
            })
    
    async def _handle_get_cached(self, message: Message) -> Message:
        """Get cached transcription if available"""
        file_hash = message.content.get("file_hash")
        
        if not file_hash:
            return message.reply(
                {"error": "No file hash provided"},
                MessageType.ERROR
            )
        
        if file_hash in self._cache:
            cached_data = self._cache[file_hash]
            return message.reply({
                "found": True,
                "transcript": cached_data["transcript"],
                "model": cached_data["model"],
                "cached_at": cached_data["timestamp"].isoformat()
            })
        
        return message.reply({
            "found": False
        })
    
    async def _transcribe_with_retry(self, 
                                   file_path: str, 
                                   model_name: str,
                                   custom_prompt: Optional[str] = None,
                                   max_retries: int = MAX_RETRIES) -> str:
        """Transcribe with retry logic"""
        # Ensure API client is initialized
        if not self._ensure_api_client(model_name):
            raise Exception("Failed to initialize API client")
            
        for attempt in range(max_retries):
            try:
                result = await asyncio.to_thread(
                    self._transcribe_audio,
                    file_path,
                    model_name,
                    custom_prompt
                )
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Transcription attempt {attempt + 1} failed, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    def _transcribe_chunk_sync(self, 
                              file_path: str, 
                              model_name: str,
                              chunk_index: int,
                              custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous transcription for use with ThreadPoolExecutor"""
        # Ensure API client is initialized
        if not self._ensure_api_client(model_name):
            raise Exception("Failed to initialize API client")
            
        transcript = self._transcribe_audio(
            file_path,
            model_name,
            custom_prompt
        )
        
        return {
            "chunk_index": chunk_index,
            "text": transcript,
            "file_path": file_path
        }
    
    def _transcribe_audio(self, file_path: str, model_name: str, custom_prompt: Optional[str] = None) -> str:
        """Transcribe audio file using Gemini API"""
        try:
            # Upload file
            file_obj = self.api_client.upload(file_path, {"mimeType": "audio/mpeg"})
            
            # Get transcription prompt
            prompt_template = get_transcription_prompt()
            prompt = prompt_template.render(num_speakers=2, metadata={})
            
            if custom_prompt:
                prompt = custom_prompt + "\n\n" + prompt
            
            # Generate content
            response = self.api_client.generate_content(model_name, [prompt, file_obj])
            
            # Extract text
            if hasattr(response, 'text'):
                return response.text
            else:
                return response.candidates[0].content.parts[0].text
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _adjust_timestamps(self, text: str, offset: float) -> str:
        """Adjust timestamps in transcript by adding offset"""
        # This is a simplified version - in production, use proper timestamp parsing
        lines = text.split('\n')
        adjusted_lines = []
        
        for line in lines:
            if line.startswith('[') and ']' in line:
                # Extract timestamp and adjust it
                try:
                    timestamp_end = line.index(']')
                    timestamp_str = line[1:timestamp_end]
                    
                    # Parse timestamp (assuming format like "00:01:23.45")
                    parts = timestamp_str.split(':')
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                        adjusted_seconds = total_seconds + offset
                        
                        # Format back to timestamp
                        adj_hours = int(adjusted_seconds // 3600)
                        adj_minutes = int((adjusted_seconds % 3600) // 60)
                        adj_seconds = adjusted_seconds % 60
                        
                        new_timestamp = f"[{adj_hours:02d}:{adj_minutes:02d}:{adj_seconds:05.2f}]"
                        adjusted_line = new_timestamp + line[timestamp_end + 1:]
                        adjusted_lines.append(adjusted_line)
                    else:
                        adjusted_lines.append(line)
                except:
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)
        
        return '\n'.join(adjusted_lines)