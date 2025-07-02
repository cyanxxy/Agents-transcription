"""Editing Assistant Agent for smart transcript editing features"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import difflib
from collections import deque

from .base_agent import BaseAgent, Message, MessageType


class EditingAssistantAgent(BaseAgent):
    """Agent responsible for intelligent editing assistance"""
    
    def __init__(self, name: str = "EditingAssistant"):
        super().__init__(name)
        self._capabilities = [
            "search_transcript",
            "replace_text",
            "smart_replace",
            "undo_redo",
            "auto_format",
            "suggest_edits",
            "track_changes"
        ]
        # Edit history for undo/redo
        self._edit_history: deque = deque(maxlen=50)
        self._redo_stack: deque = deque(maxlen=50)
        self._current_version = 0
        
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities"""
        return self._capabilities
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process editing assistant messages"""
        try:
            action = message.content.get("action")
            
            if action == "search_transcript":
                return await self._handle_search(message)
            elif action == "replace_text":
                return await self._handle_replace(message)
            elif action == "smart_replace":
                return await self._handle_smart_replace(message)
            elif action == "undo":
                return await self._handle_undo(message)
            elif action == "redo":
                return await self._handle_redo(message)
            elif action == "auto_format":
                return await self._handle_auto_format(message)
            elif action == "suggest_edits":
                return await self._handle_suggest_edits(message)
            elif action == "track_changes":
                return await self._handle_track_changes(message)
            elif action == "get_history":
                return await self._handle_get_history(message)
            else:
                return message.reply(
                    {"error": f"Unknown action: {action}"},
                    MessageType.ERROR
                )
                
        except Exception as e:
            self.logger.error(f"Error in EditingAssistantAgent: {e}")
            return message.reply(
                {"error": str(e), "details": "Editing operation failed"},
                MessageType.ERROR
            )
    
    async def _handle_search(self, message: Message) -> Message:
        """Search for text in transcript"""
        transcript = message.content.get("transcript", "")
        query = message.content.get("query", "")
        case_sensitive = message.content.get("case_sensitive", False)
        use_regex = message.content.get("use_regex", False)
        
        if not transcript or not query:
            return message.reply(
                {"error": "Transcript and query are required"},
                MessageType.ERROR
            )
        
        matches = []
        
        try:
            if use_regex:
                # Regex search
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(query, flags)
                
                for match in pattern.finditer(transcript):
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(transcript), match.end() + 50)
                    context = transcript[start:end]
                    
                    matches.append({
                        "start": match.start(),
                        "end": match.end(),
                        "text": match.group(),
                        "context": context,
                        "line_number": transcript[:match.start()].count('\n') + 1
                    })
            else:
                # Simple text search
                search_text = transcript if case_sensitive else transcript.lower()
                search_query = query if case_sensitive else query.lower()
                
                start = 0
                while True:
                    pos = search_text.find(search_query, start)
                    if pos == -1:
                        break
                    
                    # Get context
                    context_start = max(0, pos - 50)
                    context_end = min(len(transcript), pos + len(query) + 50)
                    context = transcript[context_start:context_end]
                    
                    matches.append({
                        "start": pos,
                        "end": pos + len(query),
                        "text": transcript[pos:pos + len(query)],
                        "context": context,
                        "line_number": transcript[:pos].count('\n') + 1
                    })
                    
                    start = pos + 1
            
            return message.reply({
                "success": True,
                "match_count": len(matches),
                "matches": matches[:100],  # Limit to first 100 matches
                "truncated": len(matches) > 100
            })
            
        except re.error as e:
            return message.reply(
                {"error": f"Invalid regex pattern: {str(e)}"},
                MessageType.ERROR
            )
    
    async def _handle_replace(self, message: Message) -> Message:
        """Simple find and replace"""
        transcript = message.content.get("transcript", "")
        find_text = message.content.get("find", "")
        replace_text = message.content.get("replace", "")
        case_sensitive = message.content.get("case_sensitive", False)
        whole_word = message.content.get("whole_word", False)
        
        if not transcript or not find_text:
            return message.reply(
                {"error": "Transcript and find text are required"},
                MessageType.ERROR
            )
        
        # Save current state for undo
        self._save_edit_state(transcript, "replace", {
            "find": find_text,
            "replace": replace_text
        })
        
        # Perform replacement
        if whole_word:
            # Use word boundaries
            pattern = r'\b' + re.escape(find_text) + r'\b'
            flags = 0 if case_sensitive else re.IGNORECASE
            new_transcript = re.sub(pattern, replace_text, transcript, flags=flags)
            replacement_count = len(re.findall(pattern, transcript, flags=flags))
        else:
            if case_sensitive:
                new_transcript = transcript.replace(find_text, replace_text)
                replacement_count = transcript.count(find_text)
            else:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(find_text), re.IGNORECASE)
                new_transcript = pattern.sub(replace_text, transcript)
                replacement_count = len(pattern.findall(transcript))
        
        return message.reply({
            "success": True,
            "new_transcript": new_transcript,
            "replacement_count": replacement_count,
            "version": self._current_version
        })
    
    async def _handle_smart_replace(self, message: Message) -> Message:
        """Smart replacement with context awareness"""
        transcript = message.content.get("transcript", "")
        replacements = message.content.get("replacements", [])
        
        if not transcript or not replacements:
            return message.reply(
                {"error": "Transcript and replacements are required"},
                MessageType.ERROR
            )
        
        # Save current state
        self._save_edit_state(transcript, "smart_replace", {
            "replacements": replacements
        })
        
        new_transcript = transcript
        applied_replacements = []
        
        # Sort replacements by position (reverse order to maintain positions)
        sorted_replacements = sorted(replacements, 
                                   key=lambda x: x.get("position", 0), 
                                   reverse=True)
        
        for replacement in sorted_replacements:
            old_text = replacement.get("old")
            new_text = replacement.get("new")
            position = replacement.get("position")
            context = replacement.get("context")
            
            if position is not None:
                # Position-based replacement
                if (position >= 0 and position < len(new_transcript) and
                    new_transcript[position:position + len(old_text)] == old_text):
                    
                    new_transcript = (new_transcript[:position] + 
                                    new_text + 
                                    new_transcript[position + len(old_text):])
                    applied_replacements.append(replacement)
                    
            elif context:
                # Context-based replacement
                context_pos = new_transcript.find(context)
                if context_pos != -1:
                    # Find the old text within the context
                    relative_pos = context.find(old_text)
                    if relative_pos != -1:
                        actual_pos = context_pos + relative_pos
                        new_transcript = (new_transcript[:actual_pos] + 
                                        new_text + 
                                        new_transcript[actual_pos + len(old_text):])
                        applied_replacements.append(replacement)
        
        return message.reply({
            "success": True,
            "new_transcript": new_transcript,
            "applied_count": len(applied_replacements),
            "applied_replacements": applied_replacements,
            "version": self._current_version
        })
    
    async def _handle_undo(self, message: Message) -> Message:
        """Undo last edit"""
        if not self._edit_history:
            return message.reply({
                "success": False,
                "error": "No edits to undo"
            })
        
        # Get last edit
        last_edit = self._edit_history.pop()
        
        # Save to redo stack
        self._redo_stack.append(last_edit)
        
        # Get previous state
        if self._edit_history:
            previous_state = self._edit_history[-1]
            transcript = previous_state["transcript"]
        else:
            # No more history, return to original
            transcript = last_edit.get("original_transcript", "")
        
        self._current_version += 1
        
        return message.reply({
            "success": True,
            "transcript": transcript,
            "undone_action": last_edit["action"],
            "version": self._current_version,
            "can_undo": len(self._edit_history) > 0,
            "can_redo": True
        })
    
    async def _handle_redo(self, message: Message) -> Message:
        """Redo last undone edit"""
        if not self._redo_stack:
            return message.reply({
                "success": False,
                "error": "No edits to redo"
            })
        
        # Get edit to redo
        redo_edit = self._redo_stack.pop()
        
        # Apply it back
        self._edit_history.append(redo_edit)
        self._current_version += 1
        
        return message.reply({
            "success": True,
            "transcript": redo_edit["transcript"],
            "redone_action": redo_edit["action"],
            "version": self._current_version,
            "can_undo": True,
            "can_redo": len(self._redo_stack) > 0
        })
    
    async def _handle_auto_format(self, message: Message) -> Message:
        """Auto-format transcript"""
        transcript = message.content.get("transcript", "")
        options = message.content.get("options", {})
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        # Save current state
        self._save_edit_state(transcript, "auto_format", options)
        
        formatted = transcript
        changes = []
        
        # Apply formatting based on options
        if options.get("fix_capitalization", True):
            # Fix sentence capitalization
            formatted = self._fix_sentence_capitalization(formatted)
            changes.append("Fixed sentence capitalization")
        
        if options.get("fix_punctuation_spacing", True):
            # Fix spacing around punctuation
            formatted = re.sub(r'\s+([,.!?;:])', r'\1', formatted)
            formatted = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', formatted)
            changes.append("Fixed punctuation spacing")
        
        if options.get("remove_filler_words", False):
            # Remove common filler words
            filler_words = [" um ", " uh ", " like ", " you know ", " I mean "]
            for filler in filler_words:
                formatted = formatted.replace(filler, " ")
            changes.append("Removed filler words")
        
        if options.get("standardize_numbers", False):
            # Convert number words to digits
            number_map = {
                "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
            }
            for word, digit in number_map.items():
                formatted = re.sub(r'\b' + word + r'\b', digit, formatted, flags=re.IGNORECASE)
            changes.append("Standardized numbers")
        
        if options.get("fix_line_breaks", True):
            # Normalize line breaks
            formatted = re.sub(r'\n{3,}', '\n\n', formatted)
            changes.append("Fixed line breaks")
        
        if options.get("add_paragraph_breaks", False):
            # Add paragraph breaks at natural pauses
            formatted = self._add_paragraph_breaks(formatted)
            changes.append("Added paragraph breaks")
        
        return message.reply({
            "success": True,
            "formatted_transcript": formatted,
            "changes_applied": changes,
            "version": self._current_version
        })
    
    async def _handle_suggest_edits(self, message: Message) -> Message:
        """Suggest intelligent edits"""
        transcript = message.content.get("transcript", "")
        context = message.content.get("context", {})
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        suggestions = []
        
        # Check for common issues and suggest fixes
        
        # 1. Repeated phrases
        repeated_phrases = self._find_repeated_phrases(transcript)
        for phrase, positions in repeated_phrases.items():
            if len(positions) > 2:  # Phrase appears more than twice
                suggestions.append({
                    "type": "repeated_phrase",
                    "severity": "medium",
                    "description": f"The phrase '{phrase}' appears {len(positions)} times",
                    "positions": positions[:5],  # Limit to first 5
                    "suggestion": "Consider removing repetitions or rephrasing"
                })
        
        # 2. Long sentences
        sentences = re.split(r'[.!?]+', transcript)
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            if word_count > 40:
                suggestions.append({
                    "type": "long_sentence",
                    "severity": "low",
                    "description": f"Sentence {i+1} has {word_count} words",
                    "text": sentence[:100] + "...",
                    "suggestion": "Consider breaking into shorter sentences"
                })
        
        # 3. Missing punctuation indicators
        lines = transcript.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100 and not any(p in line for p in '.!?,;:'):
                suggestions.append({
                    "type": "missing_punctuation",
                    "severity": "medium",
                    "description": f"Line {i+1} appears to be missing punctuation",
                    "text": line[:100] + "...",
                    "suggestion": "Add appropriate punctuation"
                })
        
        # 4. Inconsistent speaker labels
        speaker_pattern = r'^(\w+):\s*'
        speakers = set()
        for line in lines:
            match = re.match(speaker_pattern, line)
            if match:
                speakers.add(match.group(1))
        
        if len(speakers) > 1:
            # Check for similar speaker names
            for s1 in speakers:
                for s2 in speakers:
                    if s1 != s2 and self._string_similarity(s1, s2) > 0.8:
                        suggestions.append({
                            "type": "inconsistent_speaker",
                            "severity": "high",
                            "description": f"Similar speaker names: '{s1}' and '{s2}'",
                            "suggestion": "Use consistent speaker labels"
                        })
        
        # 5. Technical terminology consistency
        if context.get("domain"):
            # Domain-specific checks could be added here
            pass
        
        # Sort suggestions by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return message.reply({
            "success": True,
            "suggestion_count": len(suggestions),
            "suggestions": suggestions[:20],  # Limit to top 20
            "has_critical_issues": any(s["severity"] == "high" for s in suggestions)
        })
    
    async def _handle_track_changes(self, message: Message) -> Message:
        """Track changes between versions"""
        original = message.content.get("original", "")
        modified = message.content.get("modified", "")
        
        if not original or not modified:
            return message.reply(
                {"error": "Both original and modified transcripts required"},
                MessageType.ERROR
            )
        
        # Generate diff
        differ = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="Original",
            tofile="Modified",
            n=3
        )
        
        diff_text = ''.join(differ)
        
        # Calculate statistics
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        
        stats = {
            "lines_added": 0,
            "lines_removed": 0,
            "lines_modified": 0,
            "similarity_ratio": matcher.ratio()
        }
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'insert':
                stats["lines_added"] += j2 - j1
            elif op == 'delete':
                stats["lines_removed"] += i2 - i1
            elif op == 'replace':
                stats["lines_modified"] += max(i2 - i1, j2 - j1)
        
        # Find specific changes
        changes = []
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != 'equal':
                change = {
                    "type": op,
                    "original_start": i1,
                    "original_end": i2,
                    "modified_start": j1,
                    "modified_end": j2
                }
                
                if op == 'replace':
                    change["original_text"] = '\n'.join(original_lines[i1:i2])
                    change["modified_text"] = '\n'.join(modified_lines[j1:j2])
                elif op == 'delete':
                    change["deleted_text"] = '\n'.join(original_lines[i1:i2])
                elif op == 'insert':
                    change["inserted_text"] = '\n'.join(modified_lines[j1:j2])
                
                changes.append(change)
        
        return message.reply({
            "success": True,
            "diff": diff_text,
            "statistics": stats,
            "changes": changes[:50],  # Limit to 50 changes
            "total_changes": len(changes)
        })
    
    async def _handle_get_history(self, message: Message) -> Message:
        """Get edit history"""
        limit = message.content.get("limit", 10)
        
        history = list(self._edit_history)[-limit:]
        
        return message.reply({
            "history": [
                {
                    "action": edit["action"],
                    "timestamp": edit["timestamp"],
                    "details": edit.get("details", {}),
                    "version": edit.get("version", 0)
                }
                for edit in history
            ],
            "total_edits": len(self._edit_history),
            "can_undo": len(self._edit_history) > 0,
            "can_redo": len(self._redo_stack) > 0
        })
    
    def _save_edit_state(self, transcript: str, action: str, details: Dict[str, Any]):
        """Save edit state for undo/redo"""
        self._current_version += 1
        
        edit_state = {
            "transcript": transcript,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "version": self._current_version
        }
        
        # Save original if this is the first edit
        if not self._edit_history:
            edit_state["original_transcript"] = transcript
        
        self._edit_history.append(edit_state)
        
        # Clear redo stack when new edit is made
        self._redo_stack.clear()
    
    def _fix_sentence_capitalization(self, text: str) -> str:
        """Fix sentence capitalization"""
        # Split into sentences
        sentences = re.split(r'([.!?]+\s*)', text)
        
        fixed_sentences = []
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Text part (not punctuation)
                # Capitalize first letter
                part = part.lstrip()
                if part:
                    part = part[0].upper() + part[1:]
            fixed_sentences.append(part)
        
        return ''.join(fixed_sentences)
    
    def _add_paragraph_breaks(self, text: str) -> str:
        """Add paragraph breaks at natural pauses"""
        # Simple heuristic: add breaks after sentences ending with certain patterns
        patterns = [
            r'(\. )(And |But |However |Therefore |Thus |So |Then )',
            r'(\. )(Now |Next |Finally |First |Second |Third )',
            r'(\? )',
            r'(! )'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, r'\1\n\n\2', text)
        
        return text
    
    def _find_repeated_phrases(self, text: str, min_length: int = 3) -> Dict[str, List[int]]:
        """Find repeated phrases in text"""
        words = text.split()
        repeated = {}
        
        # Check phrases of different lengths
        for phrase_len in range(min_length, min(len(words), 10)):
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                
                # Skip if too short or common
                if len(phrase) < 15:
                    continue
                
                # Find all occurrences
                positions = []
                start = 0
                while True:
                    pos = text.find(phrase, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                if len(positions) > 1:
                    repeated[phrase] = positions
        
        return repeated
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity ratio"""
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()