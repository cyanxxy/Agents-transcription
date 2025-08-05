"""Structured error handling with retry mechanisms"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Type, Union, TypeVar
from functools import wraps
from datetime import datetime, timedelta
import traceback
from enum import Enum
import streamlit as st

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for structured handling"""
    API_ERROR = "api_error"
    FILE_ERROR = "file_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"
    AUTHENTICATION_ERROR = "authentication_error"
    SYSTEM_ERROR = "system_error"


class StructuredError(Exception):
    """Structured error with metadata"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 details: Optional[Dict[str, Any]] = None,
                 user_message: Optional[str] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or message
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'details': self.details,
            'user_message': self.user_message,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback
        }


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(self.initial_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay


class ErrorHandler:
    """Centralized error handling with recovery strategies"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies"""
        self.recovery_strategies[ErrorCategory.API_ERROR] = self._recover_api_error
        self.recovery_strategies[ErrorCategory.NETWORK_ERROR] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.FILE_ERROR] = self._recover_file_error
    
    async def _recover_api_error(self, error: StructuredError) -> bool:
        """Recovery strategy for API errors"""
        # Check if it's a rate limit error
        if 'rate_limit' in error.details:
            wait_time = error.details.get('retry_after', 60)
            logger.info(f"Rate limited, waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)
            return True
        
        # Check if it's an authentication error
        if 'authentication' in error.details:
            # Clear cached credentials
            if 'api_client' in st.session_state:
                del st.session_state.api_client
            return False  # Cannot auto-recover
        
        return True  # Can retry
    
    async def _recover_network_error(self, error: StructuredError) -> bool:
        """Recovery strategy for network errors"""
        # Simple wait and retry
        await asyncio.sleep(5)
        return True
    
    async def _recover_file_error(self, error: StructuredError) -> bool:
        """Recovery strategy for file errors"""
        # Check if it's a permission error
        if 'permission' in str(error.message).lower():
            return False  # Cannot auto-recover
        
        # Check if it's a space issue
        if 'space' in str(error.message).lower():
            # Try to clean up temp files
            import tempfile
            import shutil
            temp_dir = tempfile.gettempdir()
            try:
                # Clean up old temp files
                import os
                for file in os.listdir(temp_dir):
                    if file.startswith('transcriber_'):
                        try:
                            os.remove(os.path.join(temp_dir, file))
                        except:
                            pass
                return True
            except:
                return False
        
        return False
    
    def log_error(self, error: Union[StructuredError, Exception]):
        """Log error with context"""
        if isinstance(error, StructuredError):
            self.error_history.append(error.to_dict())
            
            # Log based on severity
            if error.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"{error.category.value}: {error.message}", extra=error.details)
            elif error.severity == ErrorSeverity.HIGH:
                logger.error(f"{error.category.value}: {error.message}", extra=error.details)
            elif error.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"{error.category.value}: {error.message}", extra=error.details)
            else:
                logger.info(f"{error.category.value}: {error.message}", extra=error.details)
        else:
            logger.error(f"Unstructured error: {str(error)}", exc_info=True)
    
    async def handle_with_retry(self,
                               func: Callable[..., T],
                               *args,
                               retry_config: Optional[RetryConfig] = None,
                               error_category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                               **kwargs) -> T:
        """Execute function with retry logic"""
        config = retry_config or RetryConfig()
        last_error = None
        
        for attempt in range(config.max_attempts):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_error = e
                
                # Convert to structured error if needed
                if not isinstance(e, StructuredError):
                    e = StructuredError(
                        message=str(e),
                        category=error_category,
                        details={'attempt': attempt + 1, 'max_attempts': config.max_attempts}
                    )
                
                self.log_error(e)
                
                # Check if we should retry
                if attempt < config.max_attempts - 1:
                    # Try recovery strategy
                    if e.category in self.recovery_strategies:
                        can_retry = await self.recovery_strategies[e.category](e)
                        if not can_retry:
                            raise e
                    
                    # Calculate delay
                    delay = config.get_delay(attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds (attempt {attempt + 2}/{config.max_attempts})")
                    await asyncio.sleep(delay)
                else:
                    # Max attempts reached
                    raise StructuredError(
                        message=f"Max retry attempts ({config.max_attempts}) reached",
                        category=error_category,
                        severity=ErrorSeverity.HIGH,
                        details={
                            'original_error': str(last_error),
                            'attempts': config.max_attempts
                        },
                        user_message="Operation failed after multiple attempts. Please try again later."
                    )
        
        # Should not reach here
        raise last_error


def with_error_handling(
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    user_message: Optional[str] = None
):
    """Decorator for structured error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except StructuredError:
                raise  # Already structured
            except Exception as e:
                raise StructuredError(
                    message=str(e),
                    category=category,
                    severity=severity,
                    user_message=user_message or f"An error occurred in {func.__name__}",
                    details={'function': func.__name__, 'args': str(args)[:100]}
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except StructuredError:
                raise  # Already structured
            except Exception as e:
                raise StructuredError(
                    message=str(e),
                    category=category,
                    severity=severity,
                    user_message=user_message or f"An error occurred in {func.__name__}",
                    details={'function': func.__name__, 'args': str(args)[:100]}
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR
):
    """Decorator to add retry logic to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            handler = ErrorHandler()
            config = RetryConfig(max_attempts=max_attempts, initial_delay=initial_delay)
            return await handler.handle_with_retry(
                func, *args, 
                retry_config=config, 
                error_category=category,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            handler = ErrorHandler()
            config = RetryConfig(max_attempts=max_attempts, initial_delay=initial_delay)
            
            # Convert to async for unified handling
            async def async_func():
                return func(*args, **kwargs)
            
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    handler.handle_with_retry(
                        async_func,
                        retry_config=config,
                        error_category=category
                    )
                )
            finally:
                loop.close()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator