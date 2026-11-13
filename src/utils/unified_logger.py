"""
Unified logging system for TranslateBookWithLLM
Provides consistent logging across CLI, Web, and all file types
"""
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from enum import Enum


class LogLevel(Enum):
    """Log levels with priority values"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogType(Enum):
    """Types of log messages for special handling"""
    GENERAL = "general"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOKEN_USAGE = "token_usage"
    PROGRESS = "progress"
    CHUNK_INFO = "chunk_info"
    FILE_OPERATION = "file_operation"
    TRANSLATION_START = "translation_start"
    TRANSLATION_END = "translation_end"
    ERROR_DETAIL = "error_detail"


class Colors:
    """ANSI color codes for terminal output"""
    # Check if colors should be disabled
    NO_COLOR = os.environ.get('NO_COLOR') is not None or not sys.stdout.isatty()

    YELLOW = '' if NO_COLOR else '\033[93m'       # Pour les headers
    WHITE = '' if NO_COLOR else '\033[97m'        # Pour le texte principal
    GRAY = '' if NO_COLOR else '\033[90m'         # Pour les infos techniques
    ORANGE = '' if NO_COLOR else '\033[38;5;214m' # Orange clair - INPUT vers LLM
    GREEN = '' if NO_COLOR else '\033[92m'        # Vert clair - OUTPUT du LLM
    RED = '' if NO_COLOR else '\033[91m'          # Rouge - ERREURS
    ENDC = '' if NO_COLOR else '\033[0m'          # Reset

    @classmethod
    def disable(cls):
        """Disable all colors"""
        cls.YELLOW = cls.WHITE = cls.GRAY = cls.ORANGE = cls.GREEN = cls.RED = cls.ENDC = ''


class UnifiedLogger:
    """
    Unified logger that provides consistent logging across all interfaces
    """
    
    def __init__(self, 
                 name: str = "TranslateBookWithLLM",
                 console_output: bool = True,
                 enable_colors: bool = True,
                 min_level: LogLevel = LogLevel.INFO,
                 web_callback: Optional[Callable] = None,
                 storage_callback: Optional[Callable] = None):
        """
        Initialize the unified logger
        
        Args:
            name: Logger name/identifier
            console_output: Whether to output to console
            enable_colors: Whether to use colored output
            min_level: Minimum log level to display
            web_callback: Callback for web interface (WebSocket emission)
            storage_callback: Callback for storing logs (e.g., in memory)
        """
        self.name = name
        self.console_output = console_output
        self.enable_colors = enable_colors
        self.min_level = min_level
        self.web_callback = web_callback
        self.storage_callback = storage_callback
        
        # Translation state
        self.translation_state = {
            'current_chunk': 0,
            'total_chunks': 0,
            'source_lang': '',
            'target_lang': '',
            'file_type': '',
            'model': '',
            'start_time': None,
            'in_progress': False
        }
        
        if not enable_colors:
            Colors.disable()
    
    def _format_timestamp(self) -> str:
        """Format current timestamp"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print_separator(self, char: str = '=', length: int = 80, color: str = Colors.GRAY):
        """Print a colored separator line"""
        if self.console_output:
            print(f"{color}{char * length}{Colors.ENDC}")
    
    def _format_console_message(self, level: LogLevel, message: str, 
                               log_type: LogType = LogType.GENERAL,
                               data: Optional[Dict[str, Any]] = None) -> str:
        """Format message for console output"""
        timestamp = self._format_timestamp()
        
        # Color mapping
        level_colors = {
            LogLevel.DEBUG: Colors.GRAY,
            LogLevel.INFO: Colors.WHITE,
            LogLevel.WARNING: Colors.YELLOW,
            LogLevel.ERROR: Colors.RED,
            LogLevel.CRITICAL: Colors.RED
        }
        
        color = level_colors.get(level, Colors.WHITE)
        
        # Special formatting for different log types
        if log_type == LogType.LLM_REQUEST:
            return self._format_llm_request(data or {})
        elif log_type == LogType.LLM_RESPONSE:
            return self._format_llm_response(data or {})
        elif log_type == LogType.PROGRESS:
            return self._format_progress(data or {})
        elif log_type == LogType.TRANSLATION_START:
            return self._format_translation_start(message, data or {})
        elif log_type == LogType.TRANSLATION_END:
            return self._format_translation_end(message, data or {})
        elif log_type == LogType.ERROR_DETAIL:
            return self._format_error_detail(message, data or {})
        elif log_type == LogType.TOKEN_USAGE:
            return self._format_token_usage(message, data or {})
        else:
            # General message format
            level_str = f"[{level.name}]" if level != LogLevel.INFO else ""
            return f"{color}[{timestamp}] {level_str} {message}{Colors.ENDC}"
    
    def _format_llm_request(self, data: Dict[str, Any]) -> str:
        """Format LLM request with full details"""
        output = []
        
        # Une seule ligne de séparation avant le "SENDING TO LLM"
        output.append(f"{Colors.YELLOW}{'=' * 80}{Colors.ENDC}")
        timestamp = self._format_timestamp()
        output.append(f"{Colors.YELLOW}[{timestamp}] SENDING TO LLM{Colors.ENDC}")
        
        # Chunk info
        if self.translation_state['in_progress']:
            current = self.translation_state['current_chunk']
            total = self.translation_state['total_chunks']
            percentage = (current / total * 100) if total > 0 else 0
            output.append(f"{Colors.YELLOW}Chunk: {current}/{total} ({percentage:.1f}% complete){Colors.ENDC}")
        
        # Model info (en gris)
        if 'model' in data:
            output.append(f"{Colors.GRAY}Model: {data['model']}{Colors.ENDC}")
        
        # Full prompt only in debug mode for console
        # (UI always receives the full data via web_callback)
        if self.min_level == LogLevel.DEBUG:
            output.append(f"\n{Colors.ORANGE}RAW PROMPT (INPUT):{Colors.ENDC}")
            if 'system_prompt' in data or 'user_prompt' in data:
                if data.get('system_prompt'):
                    output.append(f"{Colors.GRAY}[SYSTEM]{Colors.ENDC}")
                    output.append(f"{Colors.ORANGE}{data.get('system_prompt', '')}{Colors.ENDC}")
                if data.get('user_prompt'):
                    output.append(f"{Colors.GRAY}[USER]{Colors.ENDC}")
                    output.append(f"{Colors.ORANGE}{data.get('user_prompt', '')}{Colors.ENDC}")
            else:
                output.append(f"{Colors.ORANGE}{data.get('prompt', '')}{Colors.ENDC}")

        return '\n'.join(output)
    
    def _format_llm_response(self, data: Dict[str, Any]) -> str:
        """Format LLM response with full details"""
        # In non-debug mode, return empty string (no output)
        # Token usage is already logged separately
        if self.min_level != LogLevel.DEBUG:
            return ""

        # Debug mode: show full details
        output = []
        timestamp = self._format_timestamp()
        output.append(f"{Colors.GREEN}[{timestamp}] LLM RESPONSE (OUTPUT){Colors.ENDC}")

        if 'execution_time' in data:
            output.append(f"{Colors.GRAY}Execution time: {data['execution_time']:.2f} seconds{Colors.ENDC}")

        output.append(f"\n{Colors.GREEN}RAW RESPONSE:{Colors.ENDC}")
        output.append(f"{Colors.GREEN}{data.get('response', '')}{Colors.ENDC}")

        return '\n'.join(output)
    
    def _format_progress(self, data: Dict[str, Any]) -> str:
        """Format progress summary"""
        output = []
        
        percentage = data.get('percentage', 0)
        current = data.get('current', self.translation_state['current_chunk'])
        total = data.get('total', self.translation_state['total_chunks'])
        
        output.append(f"\n{Colors.WHITE}PROGRESS: {current}/{total} chunks ({percentage:.1f}%){Colors.ENDC}")
        
        # Progress bar simple
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        output.append(f"{Colors.WHITE}[{bar}] {percentage:.1f}%{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_translation_start(self, message: str, data: Dict[str, Any]) -> str:
        """Format translation start message"""
        output = []
        
        output.append(f"{Colors.YELLOW}TRANSLATION STARTED{Colors.ENDC}")
        
        # Update translation state
        self.translation_state.update({
            'source_lang': data.get('source_lang', 'Unknown'),
            'target_lang': data.get('target_lang', 'Unknown'),
            'file_type': data.get('file_type', 'Unknown'),
            'model': data.get('model', 'Unknown'),
            'total_chunks': data.get('total_chunks', 0),
            'current_chunk': 0,
            'start_time': datetime.now(),
            'in_progress': True
        })
        
        output.append(f"{Colors.WHITE}File Type: {self.translation_state['file_type']}{Colors.ENDC}")
        output.append(f"{Colors.WHITE}Languages: {self.translation_state['source_lang']} → {self.translation_state['target_lang']}{Colors.ENDC}")
        output.append(f"{Colors.GRAY}Model: {self.translation_state['model']}{Colors.ENDC}")
        if self.translation_state['total_chunks'] > 0:
            output.append(f"{Colors.WHITE}Total Chunks: {self.translation_state['total_chunks']}{Colors.ENDC}")
        
        return '\n'.join(output)
    
    def _format_translation_end(self, message: str, data: Dict[str, Any]) -> str:
        """Format translation end message"""
        output = []
        
        output.append(f"\n{Colors.WHITE}TRANSLATION COMPLETE{Colors.ENDC}")
        
        # Calculate duration
        if self.translation_state['start_time']:
            duration = datetime.now() - self.translation_state['start_time']
            output.append(f"{Colors.GRAY}Duration: {duration}{Colors.ENDC}")
        
        if 'output_file' in data:
            output.append(f"{Colors.WHITE}Output saved to: {data['output_file']}{Colors.ENDC}")
        
        # Statistics
        if 'stats' in data:
            stats = data['stats']
            output.append(f"{Colors.WHITE}Completed chunks: {stats.get('completed', 0)}{Colors.ENDC}")
            if stats.get('failed', 0) > 0:
                output.append(f"{Colors.YELLOW}Failed chunks: {stats['failed']}{Colors.ENDC}")
        
        # Reset state
        self.translation_state['in_progress'] = False
        
        return '\n'.join(output)
    
    def _format_error_detail(self, message: str, data: Dict[str, Any]) -> str:
        """Format detailed error message"""
        output = []

        timestamp = self._format_timestamp()
        output.append(f"{Colors.RED}[{timestamp}] ERROR: {message}{Colors.ENDC}")

        if 'details' in data:
            output.append(f"{Colors.RED}Details: {data['details']}{Colors.ENDC}")
        if 'chunk' in data:
            output.append(f"{Colors.RED}Chunk: {data['chunk']}{Colors.ENDC}")

        return '\n'.join(output)

    def _format_token_usage(self, message: str, data: Dict[str, Any]) -> str:
        """Format token usage information with progress"""
        prompt_tokens = data.get('prompt_tokens', 0)
        response_tokens = data.get('response_tokens', 0)
        total_tokens = data.get('total_tokens', 0)
        num_ctx = data.get('num_ctx', 0)

        # Calculate context usage percentage
        usage_pct = (total_tokens / num_ctx * 100) if num_ctx > 0 else 0

        # Color based on usage level
        if usage_pct > 90:
            color = Colors.YELLOW  # Warning: near limit
        else:
            color = Colors.GRAY

        # Build progress prefix if translation is in progress
        progress_str = ""
        if self.translation_state['in_progress']:
            current = self.translation_state['current_chunk']
            total = self.translation_state['total_chunks']
            if total > 0:
                progress_pct = current / total * 100
                progress_str = f"[{current}/{total} {progress_pct:.0f}%] "

        return (f"{color}{progress_str}tokens: {prompt_tokens}+{response_tokens}="
                f"{total_tokens}/{num_ctx}{Colors.ENDC}")

    def log(self, level: LogLevel, message: str,
            log_type: LogType = LogType.GENERAL,
            data: Optional[Dict[str, Any]] = None):
        """
        Main logging method

        Args:
            level: Log level
            message: Log message
            log_type: Type of log for special formatting
            data: Additional data for the log entry
        """
        # Check minimum level
        if level.value < self.min_level.value:
            return

        # Update chunk counter for LLM requests
        if log_type == LogType.LLM_REQUEST and self.translation_state['in_progress']:
            self.translation_state['current_chunk'] += 1
        
        # Format for console
        if self.console_output:
            try:
                console_msg = self._format_console_message(level, message, log_type, data)
                if console_msg:  # Only print if there's actual content
                    print(console_msg, flush=True)
            except UnicodeEncodeError:
                # Handle Unicode errors on Windows (cp1252 codec issues)
                try:
                    # Try to encode/decode safely, replacing problematic characters
                    safe_message = message.encode('ascii', 'replace').decode('ascii')
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] {safe_message}", flush=True)
                except Exception:
                    # Last resort: just print a generic message
                    print(f"[LOG] Unicode encoding error - message suppressed", flush=True)
            except Exception:
                # Fallback to simple message if formatting fails
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    safe_message = str(message).encode('ascii', 'replace').decode('ascii')
                    print(f"[{timestamp}] {safe_message}", flush=True)
                except Exception:
                    print(f"[LOG] Error displaying message", flush=True)
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.name,
            'type': log_type.value,
            'message': message,
            'data': data or {}
        }
        
        # Web callback (for WebSocket)
        if self.web_callback:
            self.web_callback(log_entry)
        
        # Storage callback (for in-memory storage)
        if self.storage_callback:
            self.storage_callback(log_entry)
    
    # Convenience methods
    def debug(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.DEBUG, message, log_type, data)
    
    def info(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.INFO, message, log_type, data)
    
    def warning(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.WARNING, message, log_type, data)
    
    def error(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.ERROR, message, log_type, data)
    
    def critical(self, message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
        self.log(LogLevel.CRITICAL, message, log_type, data)
    
    def update_total_chunks(self, total: int):
        """Update total chunks count"""
        self.translation_state['total_chunks'] = total

    def update_progress(self, completed: int, total: int):
        """Update progress from stats callback"""
        self.translation_state['current_chunk'] = completed
        self.translation_state['total_chunks'] = total
        if total > 0:
            self.translation_state['in_progress'] = True
    
    def create_legacy_callback(self):
        """
        Create a legacy callback function for backward compatibility
        Returns a function that matches the old log_callback signature
        """
        def legacy_callback(message: str, details: str = "", data: Optional[Dict[str, Any]] = None):
            # Map old message types to new log types and levels
            if data and isinstance(data, dict):
                log_type = data.get('type')
                if log_type == 'llm_request':
                    self.log(LogLevel.DEBUG, "LLM Request", LogType.LLM_REQUEST, data)
                elif log_type == 'llm_response':
                    self.log(LogLevel.DEBUG, "LLM Response", LogType.LLM_RESPONSE, data)
                elif log_type == 'progress':
                    self.log(LogLevel.INFO, "Progress Update", LogType.PROGRESS, data)
                else:
                    self.info(details or message, data=data)
            else:
                # Map specific message patterns
                if message == "token_usage":
                    # Parse token usage from details string
                    # Format: "Tokens: prompt=X, response=Y, total=Z (num_ctx=W)"
                    import re
                    token_data = {}
                    prompt_match = re.search(r'prompt=(\d+)', details)
                    response_match = re.search(r'response=(\d+)', details)
                    total_match = re.search(r'total=(\d+)', details)
                    ctx_match = re.search(r'num_ctx=(\d+)', details)
                    if prompt_match:
                        token_data['prompt_tokens'] = int(prompt_match.group(1))
                    if response_match:
                        token_data['response_tokens'] = int(response_match.group(1))
                    if total_match:
                        token_data['total_tokens'] = int(total_match.group(1))
                    if ctx_match:
                        token_data['num_ctx'] = int(ctx_match.group(1))
                    self.log(LogLevel.INFO, "Token Usage", LogType.TOKEN_USAGE, token_data)
                elif "error" in message.lower():
                    self.error(details or message)
                elif "warning" in message.lower():
                    self.warning(details or message)
                elif message == "txt_translation_info_chunks1":
                    # Extract chunk count
                    import re
                    match = re.search(r'(\d+)\s+main segments', details)
                    if match:
                        self.update_total_chunks(int(match.group(1)))
                    self.info(details)
                elif message == "txt_translation_loop_start":
                    self.translation_state['in_progress'] = True
                    self.info(details)
                else:
                    self.info(details or message)
        
        return legacy_callback


# Global logger instance
_global_logger = None


def get_logger(name: str = "TranslateBookWithLLM", **kwargs) -> UnifiedLogger:
    """
    Get or create the global logger instance

    Args:
        name: Logger name
        **kwargs: Additional arguments for UnifiedLogger

    Returns:
        UnifiedLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = UnifiedLogger(name, **kwargs)
    else:
        # Update callbacks if provided (for multi-job scenarios)
        if 'web_callback' in kwargs:
            _global_logger.web_callback = kwargs['web_callback']
        if 'storage_callback' in kwargs:
            _global_logger.storage_callback = kwargs['storage_callback']
    return _global_logger


def setup_cli_logger(enable_colors: bool = True) -> UnifiedLogger:
    """Setup logger for CLI usage"""
    # Import here to avoid circular dependencies
    from src.config import DEBUG_MODE

    return get_logger(
        console_output=True,
        enable_colors=enable_colors,
        min_level=LogLevel.DEBUG if DEBUG_MODE else LogLevel.INFO
    )


def setup_web_logger(web_callback: Callable, storage_callback: Callable) -> UnifiedLogger:
    """Setup logger for web interface usage"""
    # Import here to avoid circular dependencies
    from src.config import DEBUG_MODE

    return get_logger(
        console_output=True,  # Also output to console for debugging
        enable_colors=True,   # Colors work in console even for web
        min_level=LogLevel.DEBUG if DEBUG_MODE else LogLevel.INFO,
        web_callback=web_callback,
        storage_callback=storage_callback
    )


# === Module-level convenience functions ===

def log(level: LogLevel, message: str,
        log_type: LogType = LogType.GENERAL,
        data: Optional[Dict[str, Any]] = None):
    """
    Module-level logging function using the global logger.

    Args:
        level: Log level
        message: Log message
        log_type: Type of log for special formatting
        data: Additional data for the log entry
    """
    logger = get_logger()
    logger.log(level, message, log_type, data)


def debug(message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
    """Log debug message using global logger."""
    log(LogLevel.DEBUG, message, log_type, data)


def info(message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
    """Log info message using global logger."""
    log(LogLevel.INFO, message, log_type, data)


def warning(message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
    """Log warning message using global logger."""
    log(LogLevel.WARNING, message, log_type, data)


def error(message: str, log_type: LogType = LogType.GENERAL, data: Optional[Dict[str, Any]] = None):
    """Log error message using global logger."""
    log(LogLevel.ERROR, message, log_type, data)