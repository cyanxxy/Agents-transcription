"""Optimized state management to reduce unnecessary reruns"""

import streamlit as st
from typing import Any, Dict, Optional, Set, Callable, List
from functools import wraps
import hashlib
import json
import pickle
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StateOptimizer:
    """Optimizes Streamlit state management to minimize reruns"""
    
    def __init__(self):
        self._state_cache = {}
        self._dirty_keys = set()
        self._rerun_needed = False
        self._batch_updates = {}
        self._cached_computations = {}
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state tracking"""
        if 'state_metadata' not in st.session_state:
            st.session_state.state_metadata = {
                'last_update': datetime.now(),
                'update_count': 0,
                'rerun_count': 0,
                'cached_keys': set()
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state with caching"""
        if key in self._state_cache:
            return self._state_cache[key]
        
        value = st.session_state.get(key, default)
        self._state_cache[key] = value
        return value
    
    def set(self, key: str, value: Any, immediate: bool = False) -> None:
        """Set value in session state with optional batching"""
        # Check if value actually changed
        current = st.session_state.get(key)
        if self._values_equal(current, value):
            return  # No change, no need to update
        
        if immediate:
            # Immediate update
            st.session_state[key] = value
            self._state_cache[key] = value
            self._dirty_keys.add(key)
        else:
            # Batch update
            self._batch_updates[key] = value
            self._state_cache[key] = value
    
    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal"""
        try:
            if type(val1) != type(val2):
                return False
            
            # Handle special cases
            if isinstance(val1, (list, dict, set)):
                return json.dumps(val1, sort_keys=True) == json.dumps(val2, sort_keys=True)
            elif hasattr(val1, '__dict__'):
                return val1.__dict__ == val2.__dict__
            else:
                return val1 == val2
        except:
            return val1 == val2
    
    def batch_update(self, updates: Dict[str, Any]) -> None:
        """Batch multiple state updates"""
        for key, value in updates.items():
            self.set(key, value, immediate=False)
    
    def commit_batch(self, rerun: bool = True) -> None:
        """Commit all batched updates at once"""
        if not self._batch_updates:
            return
        
        # Apply all updates
        for key, value in self._batch_updates.items():
            st.session_state[key] = value
            self._dirty_keys.add(key)
        
        self._batch_updates.clear()
        
        # Update metadata
        st.session_state.state_metadata['last_update'] = datetime.now()
        st.session_state.state_metadata['update_count'] += 1
        
        if rerun:
            st.session_state.state_metadata['rerun_count'] += 1
            st.rerun()
    
    def needs_rerun(self, keys: Optional[Set[str]] = None) -> bool:
        """Check if rerun is needed based on dirty keys"""
        if keys:
            return bool(self._dirty_keys.intersection(keys))
        return bool(self._dirty_keys)
    
    def clear_dirty(self) -> None:
        """Clear dirty key tracking"""
        self._dirty_keys.clear()
    
    @staticmethod
    def cached_computation(
        key: str,
        func: Callable,
        *args,
        ttl: Optional[timedelta] = None,
        **kwargs
    ) -> Any:
        """Cache computation results in session state"""
        cache_key = f"_cache_{key}"
        cache_metadata_key = f"_cache_metadata_{key}"
        
        # Generate hash of arguments
        arg_hash = hashlib.md5(
            pickle.dumps((args, kwargs))
        ).hexdigest()
        
        # Check if cached result exists and is valid
        if cache_key in st.session_state:
            metadata = st.session_state.get(cache_metadata_key, {})
            
            # Check hash
            if metadata.get('arg_hash') == arg_hash:
                # Check TTL if specified
                if ttl:
                    cached_time = metadata.get('timestamp')
                    if cached_time and datetime.now() - cached_time < ttl:
                        logger.debug(f"Using cached result for {key}")
                        return st.session_state[cache_key]
                else:
                    logger.debug(f"Using cached result for {key}")
                    return st.session_state[cache_key]
        
        # Compute and cache result
        logger.debug(f"Computing result for {key}")
        result = func(*args, **kwargs)
        
        st.session_state[cache_key] = result
        st.session_state[cache_metadata_key] = {
            'arg_hash': arg_hash,
            'timestamp': datetime.now()
        }
        
        return result
    
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """Invalidate cached computations"""
        if key:
            cache_key = f"_cache_{key}"
            cache_metadata_key = f"_cache_metadata_{key}"
            
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            if cache_metadata_key in st.session_state:
                del st.session_state[cache_metadata_key]
        else:
            # Invalidate all caches
            keys_to_delete = [
                k for k in st.session_state.keys() 
                if k.startswith('_cache_')
            ]
            for key in keys_to_delete:
                del st.session_state[key]


class StatePersistence:
    """Handles state persistence across sessions"""
    
    @staticmethod
    def save_state(key: str, data: Any, storage: str = 'local') -> None:
        """Save state to persistent storage"""
        if storage == 'local':
            # Use browser local storage via custom component
            # For now, store in session state with special prefix
            st.session_state[f"_persistent_{key}"] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
        elif storage == 'file':
            # Save to file (requires file system access)
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, f"streamlit_state_{key}.pkl")
            
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now()
                }, f)
    
    @staticmethod
    def load_state(key: str, storage: str = 'local', max_age: Optional[timedelta] = None) -> Optional[Any]:
        """Load state from persistent storage"""
        if storage == 'local':
            stored = st.session_state.get(f"_persistent_{key}")
            if stored:
                if max_age:
                    timestamp = datetime.fromisoformat(stored['timestamp'])
                    if datetime.now() - timestamp > max_age:
                        return None
                return stored['data']
        elif storage == 'file':
            import tempfile
            import os
            
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, f"streamlit_state_{key}.pkl")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        stored = pickle.load(f)
                    
                    if max_age:
                        if datetime.now() - stored['timestamp'] > max_age:
                            return None
                    
                    return stored['data']
                except:
                    return None
        
        return None


def optimized_state(func: Callable) -> Callable:
    """Decorator to optimize state operations in a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = StateOptimizer()
        
        # Store current state
        initial_state = dict(st.session_state)
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Check what changed
        changed_keys = set()
        for key, value in st.session_state.items():
            if key not in initial_state or initial_state[key] != value:
                changed_keys.add(key)
        
        # Only rerun if critical keys changed
        critical_keys = {'processing_status', 'transcript_text', 'error_message'}
        if changed_keys.intersection(critical_keys):
            optimizer.commit_batch(rerun=True)
        
        return result
    
    return wrapper


def batch_state_updates():
    """Context manager for batching state updates"""
    class BatchContext:
        def __init__(self):
            self.optimizer = StateOptimizer()
        
        def __enter__(self):
            return self.optimizer
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.optimizer.commit_batch(rerun=False)
    
    return BatchContext()


def smart_rerun(condition: Callable[[], bool] = None, delay: float = 0.0):
    """Smart rerun that only triggers when necessary"""
    if condition is None or condition():
        if delay > 0:
            import time
            time.sleep(delay)
        
        # Check if we've rerun too many times recently
        metadata = st.session_state.get('state_metadata', {})
        rerun_count = metadata.get('rerun_count', 0)
        
        if rerun_count < 10:  # Prevent rerun loops
            st.rerun()
        else:
            logger.warning("Rerun limit reached, skipping rerun")


# Singleton instance
_optimizer = StateOptimizer()

def get_state_optimizer() -> StateOptimizer:
    """Get the singleton state optimizer instance"""
    return _optimizer