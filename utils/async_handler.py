"""Async handler for proper async/await management in Streamlit"""

import asyncio
import streamlit as st
from typing import Any, Callable, Dict, Optional, TypeVar, Coroutine
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncHandler:
    """Manages async operations in Streamlit without blocking"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._loop = None
        self._thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._running_tasks = {}
        self._initialized = True
    
    def get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations"""
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, create one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop
    
    async def run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run async coroutine properly"""
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We're already in async context, just await
            return await coro
        except RuntimeError:
            # No running loop, need to run in thread
            loop = self.get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
    
    def run_in_background(self, 
                         coro: Coroutine[Any, Any, T], 
                         task_id: str,
                         callback: Optional[Callable[[T], None]] = None) -> None:
        """Run async task in background with optional callback"""
        def run_task():
            try:
                loop = self.get_or_create_loop()
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                result = future.result()
                
                if callback:
                    callback(result)
                    
                # Store result in session state
                if 'async_results' not in st.session_state:
                    st.session_state.async_results = {}
                st.session_state.async_results[task_id] = {
                    'status': 'completed',
                    'result': result
                }
            except Exception as e:
                logger.error(f"Background task {task_id} failed: {e}")
                if 'async_results' not in st.session_state:
                    st.session_state.async_results = {}
                st.session_state.async_results[task_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Submit to executor
        self._executor.submit(run_task)
        
        # Mark as running
        if 'async_results' not in st.session_state:
            st.session_state.async_results = {}
        st.session_state.async_results[task_id] = {
            'status': 'running'
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of background task"""
        if 'async_results' not in st.session_state:
            return {'status': 'not_found'}
        return st.session_state.async_results.get(task_id, {'status': 'not_found'})
    
    def cleanup(self):
        """Clean up resources"""
        if self._executor:
            self._executor.shutdown(wait=False)
        if self._loop and not self._loop.is_closed():
            self._loop.close()


def async_streamlit(func: Callable) -> Callable:
    """Decorator to handle async functions in Streamlit"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = AsyncHandler()
        coro = func(*args, **kwargs)
        
        try:
            # Try to run in existing loop
            loop = asyncio.get_running_loop()
            return asyncio.create_task(coro)
        except RuntimeError:
            # No loop, create one
            loop = handler.get_or_create_loop()
            
            # Run in thread to avoid blocking
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
    
    return wrapper


def run_async_task(coro: Coroutine[Any, Any, T]) -> T:
    """Helper function to run async tasks in Streamlit"""
    handler = AsyncHandler()
    
    try:
        # Check if we're already in async context
        loop = asyncio.get_running_loop()
        # Create task in current loop
        task = asyncio.create_task(coro)
        return asyncio.run_coroutine_threadsafe(task, loop).result()
    except RuntimeError:
        # No running loop, need to handle differently
        loop = handler.get_or_create_loop()
        
        # Start loop in thread if not running
        if not loop.is_running():
            def run_loop():
                asyncio.set_event_loop(loop)
                loop.run_forever()
            
            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            
            # Give loop time to start
            import time
            time.sleep(0.1)
        
        # Now run coroutine in the loop
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()