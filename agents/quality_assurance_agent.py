"""Quality Assurance Agent for transcript validation and improvement"""

import re
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher
import nltk
from collections import Counter

from .base_agent import BaseAgent, Message, MessageType


class QualityAssuranceAgent(BaseAgent):
    """Agent responsible for transcript quality checking and improvement"""
    
    def __init__(self, name: str = "QualityAssurance"):
        super().__init__(name)
        self._capabilities = [
            "validate_transcript",
            "detect_errors",
            "suggest_corrections",
            "check_consistency",
            "analyze_quality",
            "fix_common_errors"
        ]
        self._init_nlp_resources()
        
    def _init_nlp_resources(self):
        """Initialize NLP resources"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            self.logger.warning("Failed to download NLTK resources")
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities"""
        return self._capabilities
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process quality assurance messages"""
        try:
            action = message.content.get("action")
            
            if action == "validate_transcript":
                return await self._handle_validate_transcript(message)
            elif action == "detect_errors":
                return await self._handle_detect_errors(message)
            elif action == "suggest_corrections":
                return await self._handle_suggest_corrections(message)
            elif action == "check_consistency":
                return await self._handle_check_consistency(message)
            elif action == "analyze_quality":
                return await self._handle_analyze_quality(message)
            elif action == "fix_common_errors":
                return await self._handle_fix_common_errors(message)
            else:
                return message.reply(
                    {"error": f"Unknown action: {action}"},
                    MessageType.ERROR
                )
                
        except Exception as e:
            self.logger.error(f"Error in QualityAssuranceAgent: {e}")
            return message.reply(
                {"error": str(e), "details": "Quality check failed"},
                MessageType.ERROR
            )
    
    async def _handle_validate_transcript(self, message: Message) -> Message:
        """Validate transcript structure and content"""
        transcript = message.content.get("transcript", "")
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        issues = []
        warnings = []
        
        # Check for empty transcript
        if not transcript.strip():
            issues.append("Transcript is empty")
        
        # Check for timestamp format issues
        timestamp_pattern = r'\[\d{2}:\d{2}:\d{2}\.\d{2}\]'
        has_timestamps = bool(re.search(timestamp_pattern, transcript))
        
        if has_timestamps:
            # Validate timestamp consistency
            timestamps = re.findall(timestamp_pattern, transcript)
            if timestamps:
                prev_time = None
                for ts in timestamps:
                    time_str = ts[1:-1]  # Remove brackets
                    try:
                        parts = time_str.split(':')
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        
                        current_time = hours * 3600 + minutes * 60 + seconds
                        
                        if prev_time and current_time < prev_time:
                            issues.append(f"Timestamp order issue: {ts} comes after later timestamp")
                        
                        prev_time = current_time
                        
                    except:
                        issues.append(f"Invalid timestamp format: {ts}")
        
        # Check for common formatting issues
        lines = transcript.split('\n')
        
        # Check for excessive blank lines
        consecutive_blanks = 0
        for line in lines:
            if not line.strip():
                consecutive_blanks += 1
                if consecutive_blanks > 2:
                    warnings.append("Excessive blank lines detected")
                    break
            else:
                consecutive_blanks = 0
        
        # Check for very long lines
        for i, line in enumerate(lines):
            if len(line) > 500:
                warnings.append(f"Very long line detected at line {i+1}")
        
        # Check for encoding issues
        if '\ufffd' in transcript:
            issues.append("Encoding issues detected (replacement characters found)")
        
        return message.reply({
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "has_timestamps": has_timestamps,
            "line_count": len(lines),
            "word_count": len(transcript.split()),
            "character_count": len(transcript)
        })
    
    async def _handle_detect_errors(self, message: Message) -> Message:
        """Detect potential errors in transcript"""
        transcript = message.content.get("transcript", "")
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        errors = []
        
        # Common transcription errors patterns
        error_patterns = [
            # Repeated words
            (r'\b(\w+)\s+\1\b', "Repeated word: {0}"),
            # Missing spaces after punctuation
            (r'[.!?][a-zA-Z]', "Missing space after punctuation"),
            # Multiple punctuation
            (r'[.!?]{2,}', "Multiple punctuation marks"),
            # Incomplete sentences (very short segments between periods)
            (r'(?<=[.!?])\s*\w{1,3}\s*(?=[.!?])', "Possibly incomplete sentence"),
        ]
        
        for pattern, error_msg in error_patterns:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                errors.append({
                    "type": "pattern_match",
                    "pattern": pattern,
                    "text": match.group(),
                    "position": match.start(),
                    "message": error_msg.format(match.group())
                })
        
        # Check for common mistranscriptions
        common_errors = {
            " gonna ": " going to ",
            " wanna ": " want to ",
            " gotta ": " got to ",
            " kinda ": " kind of ",
            " sorta ": " sort of ",
            " 'cause ": " because ",
            " prolly ": " probably ",
            " dunno ": " don't know ",
        }
        
        for error, correction in common_errors.items():
            if error in transcript.lower():
                positions = [m.start() for m in re.finditer(re.escape(error), transcript, re.IGNORECASE)]
                for pos in positions:
                    errors.append({
                        "type": "informal_speech",
                        "text": error.strip(),
                        "suggestion": correction.strip(),
                        "position": pos,
                        "message": f"Informal: '{error.strip()}' -> '{correction.strip()}'"
                    })
        
        # Check for number/word inconsistencies
        number_word_issues = self._check_number_consistency(transcript)
        errors.extend(number_word_issues)
        
        return message.reply({
            "error_count": len(errors),
            "errors": errors[:50],  # Limit to first 50 errors
            "has_informal_speech": any(e["type"] == "informal_speech" for e in errors),
            "has_pattern_errors": any(e["type"] == "pattern_match" for e in errors)
        })
    
    async def _handle_suggest_corrections(self, message: Message) -> Message:
        """Suggest corrections for detected errors"""
        transcript = message.content.get("transcript", "")
        errors = message.content.get("errors", [])
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        corrections = []
        
        for error in errors:
            if error["type"] == "informal_speech":
                corrections.append({
                    "original": error["text"],
                    "suggestion": error["suggestion"],
                    "position": error["position"],
                    "confidence": 0.9,
                    "type": "informal_to_formal"
                })
            
            elif error["type"] == "pattern_match":
                if "Repeated word" in error["message"]:
                    # Remove the repeated word
                    words = error["text"].split()
                    if len(words) == 2 and words[0] == words[1]:
                        corrections.append({
                            "original": error["text"],
                            "suggestion": words[0],
                            "position": error["position"],
                            "confidence": 0.8,
                            "type": "remove_duplicate"
                        })
                
                elif "Missing space after punctuation" in error["message"]:
                    # Add space after punctuation
                    corrected = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', error["text"])
                    corrections.append({
                        "original": error["text"],
                        "suggestion": corrected,
                        "position": error["position"],
                        "confidence": 0.95,
                        "type": "add_space"
                    })
        
        # Sort corrections by position (descending) to apply from end to start
        corrections.sort(key=lambda x: x["position"], reverse=True)
        
        return message.reply({
            "corrections": corrections,
            "total_suggestions": len(corrections)
        })
    
    async def _handle_check_consistency(self, message: Message) -> Message:
        """Check consistency across transcript chunks"""
        chunks = message.content.get("chunks", [])
        
        if not chunks or len(chunks) < 2:
            return message.reply({
                "consistent": True,
                "message": "Not enough chunks to check consistency"
            })
        
        inconsistencies = []
        
        # Check for overlap/gap issues between chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Get last few words of current chunk
            current_words = current_chunk.strip().split()[-10:]
            # Get first few words of next chunk
            next_words = next_chunk.strip().split()[:10]
            
            # Check for repeated content (overlap)
            overlap = self._find_overlap(current_words, next_words)
            if overlap > 3:  # More than 3 words repeated
                inconsistencies.append({
                    "type": "overlap",
                    "chunks": [i, i + 1],
                    "description": f"Chunks {i} and {i+1} have {overlap} words overlap"
                })
            
            # Check for potential gaps (abrupt transitions)
            if current_chunk and next_chunk:
                if current_chunk[-1] not in '.!?' and next_chunk[0].isupper():
                    inconsistencies.append({
                        "type": "potential_gap",
                        "chunks": [i, i + 1],
                        "description": f"Potential gap between chunks {i} and {i+1}"
                    })
        
        # Check for style consistency
        style_stats = []
        for i, chunk in enumerate(chunks):
            stats = {
                "chunk_index": i,
                "has_timestamps": bool(re.search(r'\[\d{2}:\d{2}:\d{2}\.\d{2}\]', chunk)),
                "avg_sentence_length": self._avg_sentence_length(chunk),
                "capitalization_ratio": self._capitalization_ratio(chunk)
            }
            style_stats.append(stats)
        
        # Check for significant style differences
        for i in range(len(style_stats) - 1):
            current = style_stats[i]
            next_stat = style_stats[i + 1]
            
            if current["has_timestamps"] != next_stat["has_timestamps"]:
                inconsistencies.append({
                    "type": "timestamp_inconsistency",
                    "chunks": [i, i + 1],
                    "description": "Timestamp format inconsistency between chunks"
                })
        
        return message.reply({
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "chunk_count": len(chunks)
        })
    
    async def _handle_analyze_quality(self, message: Message) -> Message:
        """Analyze overall transcript quality"""
        transcript = message.content.get("transcript", "")
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        # Calculate quality metrics
        metrics = {
            "readability_score": self._calculate_readability(transcript),
            "sentence_variety": self._sentence_variety_score(transcript),
            "punctuation_density": self._punctuation_density(transcript),
            "average_sentence_length": self._avg_sentence_length(transcript),
            "vocabulary_richness": self._vocabulary_richness(transcript),
            "timestamp_coverage": self._timestamp_coverage(transcript)
        }
        
        # Overall quality score (0-100)
        quality_score = (
            metrics["readability_score"] * 0.3 +
            metrics["sentence_variety"] * 0.2 +
            (1 - abs(metrics["punctuation_density"] - 0.15)) * 100 * 0.2 +  # Optimal ~15%
            min(100, max(0, 100 - abs(metrics["average_sentence_length"] - 15) * 5)) * 0.2 +  # Optimal ~15 words
            metrics["vocabulary_richness"] * 0.1
        )
        
        # Quality assessment
        if quality_score >= 80:
            assessment = "Excellent"
        elif quality_score >= 70:
            assessment = "Good"
        elif quality_score >= 60:
            assessment = "Fair"
        else:
            assessment = "Needs Improvement"
        
        return message.reply({
            "quality_score": round(quality_score, 2),
            "assessment": assessment,
            "metrics": metrics,
            "recommendations": self._get_quality_recommendations(metrics)
        })
    
    async def _handle_fix_common_errors(self, message: Message) -> Message:
        """Automatically fix common errors"""
        transcript = message.content.get("transcript", "")
        
        if not transcript:
            return message.reply(
                {"error": "No transcript provided"},
                MessageType.ERROR
            )
        
        fixed_transcript = transcript
        fixes_applied = []
        
        # Fix double spaces
        if "  " in fixed_transcript:
            fixed_transcript = re.sub(r'\s+', ' ', fixed_transcript)
            fixes_applied.append("Removed extra spaces")
        
        # Fix missing spaces after punctuation
        before = fixed_transcript
        fixed_transcript = re.sub(r'([.!?])([A-Z])', r'\1 \2', fixed_transcript)
        if before != fixed_transcript:
            fixes_applied.append("Added missing spaces after punctuation")
        
        # Fix repeated words
        before = fixed_transcript
        fixed_transcript = re.sub(r'\b(\w+)\s+\1\b', r'\1', fixed_transcript, flags=re.IGNORECASE)
        if before != fixed_transcript:
            fixes_applied.append("Removed repeated words")
        
        # Fix common informal speech (optional)
        informal_replacements = {
            " gonna ": " going to ",
            " wanna ": " want to ",
            " gotta ": " got to ",
            "'cause ": "because ",
        }
        
        for informal, formal in informal_replacements.items():
            if informal in fixed_transcript.lower():
                fixed_transcript = re.sub(
                    re.escape(informal), 
                    formal, 
                    fixed_transcript, 
                    flags=re.IGNORECASE
                )
                fixes_applied.append(f"Replaced '{informal.strip()}' with '{formal.strip()}'")
        
        # Fix multiple punctuation
        before = fixed_transcript
        fixed_transcript = re.sub(r'([.!?]){2,}', r'\1', fixed_transcript)
        if before != fixed_transcript:
            fixes_applied.append("Fixed multiple punctuation marks")
        
        return message.reply({
            "success": True,
            "fixed_transcript": fixed_transcript,
            "fixes_applied": fixes_applied,
            "changes_made": len(fixes_applied) > 0
        })
    
    def _check_number_consistency(self, text: str) -> List[Dict[str, Any]]:
        """Check for inconsistencies in number representation"""
        errors = []
        
        # Find numbers written as words
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        # Check if both numeric and word forms are used
        has_numeric = bool(re.search(r'\b\d+\b', text))
        has_word_numbers = any(word in text.lower() for word in number_words)
        
        if has_numeric and has_word_numbers:
            errors.append({
                "type": "number_consistency",
                "message": "Mix of numeric and word number formats detected",
                "position": 0
            })
        
        return errors
    
    def _find_overlap(self, words1: List[str], words2: List[str]) -> int:
        """Find the longest overlap between end of words1 and start of words2"""
        max_overlap = 0
        
        for i in range(1, min(len(words1), len(words2)) + 1):
            if words1[-i:] == words2[:i]:
                max_overlap = i
        
        return max_overlap
    
    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _capitalization_ratio(self, text: str) -> float:
        """Calculate ratio of capitalized words"""
        words = text.split()
        if not words:
            return 0
        
        capitalized = sum(1 for w in words if w and w[0].isupper())
        return capitalized / len(words)
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (0-100)"""
        avg_sentence_len = self._avg_sentence_length(text)
        
        # Optimal sentence length is around 15-20 words
        if 15 <= avg_sentence_len <= 20:
            score = 100
        elif avg_sentence_len < 15:
            score = max(0, 100 - (15 - avg_sentence_len) * 5)
        else:
            score = max(0, 100 - (avg_sentence_len - 20) * 3)
        
        return score
    
    def _sentence_variety_score(self, text: str) -> float:
        """Score based on sentence length variety"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 50
        
        lengths = [len(s.split()) for s in sentences]
        
        # Calculate standard deviation
        avg = sum(lengths) / len(lengths)
        variance = sum((x - avg) ** 2 for x in lengths) / len(lengths)
        std_dev = variance ** 0.5
        
        # Good variety has std dev of 5-10
        if 5 <= std_dev <= 10:
            return 100
        elif std_dev < 5:
            return std_dev * 20
        else:
            return max(0, 100 - (std_dev - 10) * 5)
    
    def _punctuation_density(self, text: str) -> float:
        """Calculate punctuation density"""
        if not text:
            return 0
        
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        return punctuation_count / len(text)
    
    def _vocabulary_richness(self, text: str) -> float:
        """Calculate vocabulary richness (0-100)"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return 0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Type-token ratio
        ttr = unique_words / total_words
        
        # Scale to 0-100 (TTR of 0.5 is considered good)
        return min(100, ttr * 200)
    
    def _timestamp_coverage(self, text: str) -> float:
        """Calculate percentage of text with timestamps"""
        lines = text.split('\n')
        lines_with_timestamps = sum(1 for line in lines 
                                   if re.search(r'\[\d{2}:\d{2}:\d{2}\.\d{2}\]', line))
        
        if not lines:
            return 0
        
        return (lines_with_timestamps / len(lines)) * 100
    
    def _get_quality_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Get recommendations based on quality metrics"""
        recommendations = []
        
        if metrics["readability_score"] < 70:
            if metrics["average_sentence_length"] > 25:
                recommendations.append("Consider breaking up long sentences for better readability")
            elif metrics["average_sentence_length"] < 10:
                recommendations.append("Some sentences are very short. Consider combining related ideas")
        
        if metrics["sentence_variety"] < 50:
            recommendations.append("Try to vary sentence lengths for better flow")
        
        if metrics["punctuation_density"] < 0.05:
            recommendations.append("The transcript may be missing punctuation")
        elif metrics["punctuation_density"] > 0.25:
            recommendations.append("The transcript may have excessive punctuation")
        
        if metrics["vocabulary_richness"] < 30:
            recommendations.append("The vocabulary appears repetitive")
        
        if metrics["timestamp_coverage"] > 0 and metrics["timestamp_coverage"] < 50:
            recommendations.append("Timestamp coverage is incomplete")
        
        return recommendations