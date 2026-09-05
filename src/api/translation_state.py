"""
Thread-safe translation state management
"""
import atexit
import threading
import time
import copy
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
from src.persistence.checkpoint_manager import CheckpointManager

if TYPE_CHECKING:
    from src.core.glossary import GlossaryStore


def generate_server_session_id() -> str:
    """Generate a unique session ID for this server instance using timestamp."""
    import time
    return str(int(time.time()))


class TranslationStateManager:
    """Thread-safe manager for translation state"""

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, server_session_id: Optional[str] = None):
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Use RLock to allow nested locking
        # Generate a unique session ID for this server instance
        self.server_session_id = server_session_id or generate_server_session_id()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            server_session_id=self.server_session_id
        )
        self.glossary_store: Optional["GlossaryStore"] = None
        self._glossary_lock = threading.Lock()
    
    def create_translation(self, translation_id: str, config: Dict[str, Any]) -> None:
        """Create a new translation entry"""
        with self._lock:
            self._translations[translation_id] = {
                'status': 'queued',
                'progress': 0,
                'stats': {
                    'start_time': time.time(),
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                    # OpenRouter cost tracking
                    'openrouter_cost': 0.0,
                    'openrouter_prompt_tokens': 0,
                    'openrouter_completion_tokens': 0
                },
                'logs': [f"[{datetime.now().strftime('%H:%M:%S')}] Translation {translation_id} queued."],
                'result': None,
                'config': config,
                'interrupted': False,
                'output_filepath': None
            }
    
    def update_translation(self, translation_id: str, updates: Dict[str, Any]) -> bool:
        """Update translation state safely"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            
            translation = self._translations[translation_id]
            
            # Handle nested updates for stats
            if 'stats' in updates and isinstance(updates['stats'], dict):
                if 'stats' not in translation:
                    translation['stats'] = {}
                translation['stats'].update(updates['stats'])
                updates = {k: v for k, v in updates.items() if k != 'stats'}
            
            # Handle logs append
            if 'log' in updates:
                if 'logs' not in translation:
                    translation['logs'] = []
                translation['logs'].append(updates['log'])
                updates = {k: v for k, v in updates.items() if k != 'log'}
            
            # Update remaining fields
            translation.update(updates)
            return True
    
    def get_translation(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """Get translation state safely"""
        with self._lock:
            if translation_id not in self._translations:
                return None
            # Return a deep copy to prevent external modification of nested objects
            return copy.deepcopy(self._translations[translation_id])
    
    def get_translation_field(self, translation_id: str, field: str, default=None):
        """Get a specific field from translation state"""
        with self._lock:
            if translation_id not in self._translations:
                return default
            return self._translations[translation_id].get(field, default)
    
    def set_translation_field(self, translation_id: str, field: str, value: Any) -> bool:
        """Set a specific field in translation state"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            self._translations[translation_id][field] = value
            return True
    
    def append_log(self, translation_id: str, log_entry: str) -> bool:
        """Append a log entry to translation"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            if 'logs' not in self._translations[translation_id]:
                self._translations[translation_id]['logs'] = []
            self._translations[translation_id]['logs'].append(log_entry)
            return True
    
    def update_stats(self, translation_id: str, stats_update: Dict[str, Any]) -> bool:
        """Update translation statistics"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            if 'stats' not in self._translations[translation_id]:
                self._translations[translation_id]['stats'] = {}
            self._translations[translation_id]['stats'].update(stats_update)
            return True
    
    def exists(self, translation_id: str) -> bool:
        """Check if translation exists"""
        with self._lock:
            return translation_id in self._translations
    
    def get_all_translations(self) -> Dict[str, Dict[str, Any]]:
        """Get all translations (returns a deep copy)"""
        with self._lock:
            return copy.deepcopy(self._translations)
    
    def get_translation_summaries(self) -> list:
        """Get summaries of all translations for listing"""
        with self._lock:
            summaries = []
            for tid, data in self._translations.items():
                config = data.get('config', {})
                stats = data.get('stats', {})
                summaries.append({
                    "translation_id": tid,
                    "status": data.get('status'),
                    "progress": data.get('progress'),
                    "start_time": stats.get('start_time'),
                    "output_filename": config.get('output_filename'),
                    "input_filename": config.get('input_filename'),
                    "file_type": config.get('file_type', 'txt'),
                    # Include stats for UI restoration
                    "total_chunks": stats.get('total_chunks', 0),
                    "completed_chunks": stats.get('completed_chunks', 0),
                    "progress_percent": stats.get('progress_percent'),
                    "current_phase": stats.get('current_phase'),
                    "enable_refinement": stats.get('enable_refinement', False),
                    # Canonical progress contract (Step 2 seam), alongside the
                    # legacy fields above for back-compat.
                    "percent": stats.get('percent'),
                    "phase": stats.get('phase'),
                    "last_translation": data.get('last_translation')
                })
            return sorted(summaries, key=lambda x: x.get('start_time', 0), reverse=True)
    
    def is_interrupted(self, translation_id: str) -> bool:
        """Check if translation is interrupted"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            return self._translations[translation_id].get('interrupted', False)
    
    def set_interrupted(self, translation_id: str, interrupted: bool = True) -> bool:
        """Set interrupted flag for translation"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            self._translations[translation_id]['interrupted'] = interrupted
            return True

    def get_resumable_jobs(self):
        """Get all jobs that can be resumed from database"""
        return self.checkpoint_manager.get_resumable_jobs()

    def restore_job_from_checkpoint(self, translation_id: str) -> bool:
        """
        Restore a job from checkpoint into in-memory state.

        Args:
            translation_id: Job identifier

        Returns:
            True if restored successfully
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return False

        job = checkpoint_data['job']
        with self._lock:
            # Restore job into in-memory state
            # Use deepcopy for config to prevent mutation of stored config
            self._translations[translation_id] = {
                'status': 'paused',  # Will be set to 'running' when resumed
                'progress': 0,
                'stats': copy.deepcopy(job['progress']),
                'logs': [f"[{datetime.now().strftime('%H:%M:%S')}] Job restored from checkpoint."],
                'result': None,
                'config': copy.deepcopy(job['config']),
                'interrupted': False,
                'output_filepath': job['config'].get('output_filepath'),
                'resume_from_index': checkpoint_data['resume_from_index']
            }

        return True

    def delete_checkpoint(self, translation_id: str) -> bool:
        """
        Delete a checkpoint for a job.

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        # Remove from in-memory state if exists
        with self._lock:
            if translation_id in self._translations:
                del self._translations[translation_id]

        # Delete from database
        return self.checkpoint_manager.delete_checkpoint(translation_id)

    def get_checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager instance"""
        return self.checkpoint_manager

    def get_glossary_store(self) -> "GlossaryStore":
        """Return the shared GlossaryStore, instantiating it on first use.

        A single store is shared by the glossary blueprint and the translation
        handler so we don't end up with multiple stores each leaking
        per-thread SQLite connections.
        """
        if self.glossary_store is not None:
            return self.glossary_store
        with self._glossary_lock:
            if self.glossary_store is None:
                from src.core.glossary import GlossaryStore
                self.glossary_store = GlossaryStore()
            return self.glossary_store

    def close_glossary_store(self) -> None:
        """Close every connection held by the shared GlossaryStore, if any."""
        with self._glossary_lock:
            store = self.glossary_store
            self.glossary_store = None
        if store is not None:
            try:
                store.close_all()
            except Exception:
                pass


# Global instance
_state_manager = TranslationStateManager()


def get_state_manager() -> TranslationStateManager:
    """Get the global state manager instance"""
    return _state_manager


def get_glossary_store() -> "GlossaryStore":
    """Return the process-wide shared GlossaryStore."""
    return _state_manager.get_glossary_store()


@atexit.register
def _shutdown_glossary_store() -> None:
    """Close all GlossaryStore connections on interpreter shutdown."""
    try:
        _state_manager.close_glossary_store()
    except Exception:
        pass