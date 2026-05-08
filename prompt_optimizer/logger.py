"""
Colored console logger for the prompt optimizer.

Provides rich visual feedback with colors to differentiate:
- Ollama (translation) outputs
- OpenRouter (evaluation) outputs
- OpenRouter (mutation) outputs
- System messages
"""

import sys
from typing import Optional
from enum import Enum


class Color(Enum):
    """ANSI color codes."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Standard colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def supports_color() -> bool:
    """Check if the terminal supports color."""
    # Windows 10+ supports ANSI colors
    if sys.platform == "win32":
        try:
            import os
            # Enable ANSI escape sequences on Windows
            os.system("")
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


USE_COLORS = supports_color()


def c(text: str, *colors: Color) -> str:
    """Apply colors to text."""
    if not USE_COLORS:
        return text
    color_codes = "".join(color.value for color in colors)
    return f"{color_codes}{text}{Color.RESET.value}"


def box(text: str, color: Color = Color.WHITE, title: str = "") -> str:
    """Create a boxed text output."""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines) if lines else 0
    if title:
        max_len = max(max_len, len(title) + 4)

    border_color = color
    top = c(f"{'=' * (max_len + 4)}", border_color)
    bottom = c(f"{'=' * (max_len + 4)}", border_color)

    if title:
        title_line = c(f"  {title}  ".center(max_len + 4, '='), border_color, Color.BOLD)
    else:
        title_line = top

    boxed_lines = [title_line]
    for line in lines:
        padded = line.ljust(max_len)
        boxed_lines.append(c("| ", border_color) + padded + c(" |", border_color))
    boxed_lines.append(bottom)

    return '\n'.join(boxed_lines)


class ConsoleLogger:
    """
    Rich console logger with colored output for different LLM operations.
    """

    # Color schemes for different sources
    COLORS = {
        'ollama': Color.CYAN,
        'openrouter': Color.MAGENTA,
        'mutation': Color.YELLOW,
        'system': Color.WHITE,
        'success': Color.GREEN,
        'error': Color.RED,
        'warning': Color.BRIGHT_YELLOW,
        'info': Color.BRIGHT_WHITE,
        'debug': Color.BRIGHT_BLACK,
    }

    ICONS = {
        'ollama': '',
        'openrouter': '',
        'mutation': '',
        'system': '',
        'success': '',
        'error': '',
        'warning': '',
        'translate': '',
        'evaluate': '',
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return time.strftime("%H:%M:%S")

    def _print(self, message: str):
        """Print message to console."""
        print(message, flush=True)

    def header(self, text: str, color: Color = Color.BRIGHT_WHITE):
        """Print a header."""
        self._print("")
        self._print(c(f"{'=' * 70}", color))
        self._print(c(f"  {text}", color, Color.BOLD))
        self._print(c(f"{'=' * 70}", color))

    def subheader(self, text: str, color: Color = Color.WHITE):
        """Print a subheader."""
        self._print(c(f"\n--- {text} ---", color))

    def info(self, message: str):
        """Print info message."""
        ts = c(f"[{self._timestamp()}]", Color.DIM)
        level = c("[INFO ]", Color.BRIGHT_WHITE)
        self._print(f"{ts} {level} {message}")

    def debug(self, message: str):
        """Print debug message."""
        if not self.verbose:
            return
        ts = c(f"[{self._timestamp()}]", Color.DIM)
        level = c("[DEBUG]", Color.BRIGHT_BLACK)
        self._print(f"{ts} {level} {c(message, Color.DIM)}")

    def success(self, message: str):
        """Print success message."""
        ts = c(f"[{self._timestamp()}]", Color.DIM)
        level = c("[ OK  ]", Color.GREEN, Color.BOLD)
        self._print(f"{ts} {level} {c(message, Color.GREEN)}")

    def warning(self, message: str):
        """Print warning message."""
        ts = c(f"[{self._timestamp()}]", Color.DIM)
        level = c("[WARN ]", Color.YELLOW, Color.BOLD)
        self._print(f"{ts} {level} {c(message, Color.YELLOW)}")

    def error(self, message: str):
        """Print error message."""
        ts = c(f"[{self._timestamp()}]", Color.DIM)
        level = c("[ERROR]", Color.RED, Color.BOLD)
        self._print(f"{ts} {level} {c(message, Color.RED)}")

    def ollama_request(self, model: str, system_prompt: str, user_prompt: str):
        """Log Ollama translation request."""
        self._print("")
        self._print(c(f"  OLLAMA REQUEST ({model})", Color.CYAN, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.CYAN))

        # System prompt (truncated)
        sys_preview = system_prompt[:200].replace('\n', ' ')
        if len(system_prompt) > 200:
            sys_preview += "..."
        self._print(c("  System: ", Color.CYAN) + c(sys_preview, Color.DIM))

        # User prompt (truncated)
        user_preview = user_prompt[:150].replace('\n', ' ')
        if len(user_prompt) > 150:
            user_preview += "..."
        self._print(c("  User: ", Color.CYAN) + user_preview)

    def ollama_response(self, translation: str, elapsed_ms: int, tokens: int = 0):
        """Log Ollama translation response."""
        self._print(c(f"  OLLAMA RESPONSE ({elapsed_ms}ms, ~{tokens} tokens)", Color.CYAN, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.CYAN))

        # Show translation (truncated if long)
        lines = translation.split('\n')
        for i, line in enumerate(lines[:5]):
            prefix = "  " if i > 0 else "  "
            display_line = line[:100] + "..." if len(line) > 100 else line
            self._print(c(prefix, Color.CYAN) + display_line)

        if len(lines) > 5:
            self._print(c(f"  ... ({len(lines) - 5} more lines)", Color.DIM))

        self._print("")

    def openrouter_eval_request(self, model: str, source_text: str, translation: str):
        """Log OpenRouter evaluation request."""
        self._print(c(f"  OPENROUTER EVAL REQUEST ({model})", Color.MAGENTA, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.MAGENTA))

        # Source preview
        src_preview = source_text[:100].replace('\n', ' ')
        if len(source_text) > 100:
            src_preview += "..."
        self._print(c("  Source: ", Color.MAGENTA) + c(src_preview, Color.DIM))

        # Translation preview
        trans_preview = translation[:100].replace('\n', ' ')
        if len(translation) > 100:
            trans_preview += "..."
        self._print(c("  Translation: ", Color.MAGENTA) + trans_preview)

    def openrouter_eval_response(
        self,
        accuracy: float,
        fluency: float,
        style: float,
        overall: float,
        feedback: str,
        elapsed_ms: int,
        cost: float
    ):
        """Log OpenRouter evaluation response."""
        self._print(c(f"  OPENROUTER EVAL RESPONSE ({elapsed_ms}ms, ${cost:.4f})", Color.MAGENTA, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.MAGENTA))

        # Scores with color coding
        def score_color(score: float) -> Color:
            if score >= 8:
                return Color.GREEN
            elif score >= 6:
                return Color.YELLOW
            else:
                return Color.RED

        scores = [
            ("Accuracy", accuracy),
            ("Fluency", fluency),
            ("Style", style),
            ("Overall", overall)
        ]

        score_str = "  "
        for name, score in scores:
            color = score_color(score)
            score_str += f"{name}: {c(f'{score:.1f}', color)}  "

        self._print(score_str)

        # Feedback
        feedback_preview = feedback[:150].replace('\n', ' ')
        if len(feedback) > 150:
            feedback_preview += "..."
        self._print(c("  Feedback: ", Color.MAGENTA) + c(feedback_preview, Color.DIM))
        self._print("")

    def mutation_request(self, strategy: str, parent_id: str, parent_fitness: float):
        """Log mutation request."""
        self._print("")
        self._print(c(f"  MUTATION REQUEST ({strategy.upper()})", Color.YELLOW, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.YELLOW))
        self._print(c(f"  Parent: {parent_id} (fitness: {parent_fitness:.3f})", Color.YELLOW))

    def mutation_context(self, feedbacks: list):
        """Log mutation context (feedbacks being sent)."""
        if not feedbacks:
            return

        self._print(c("  Feedbacks provided:", Color.YELLOW))
        for fb in feedbacks[:3]:  # Show max 3
            score_color = Color.GREEN if fb.score >= 7 else (Color.YELLOW if fb.score >= 5 else Color.RED)
            self._print(
                c(f"    - {fb.text_title}: ", Color.DIM) +
                c(f"{fb.score:.1f}", score_color) +
                c(f" ({fb.feedback[:50]}...)" if fb.feedback else "", Color.DIM)
            )

    def mutation_response(self, new_prompt: str, elapsed_ms: int, token_change: int):
        """Log mutation response."""
        sign = "+" if token_change > 0 else ""
        change_color = Color.RED if token_change > 50 else (Color.GREEN if token_change < 0 else Color.WHITE)

        self._print(c(f"  MUTATION RESPONSE ({elapsed_ms}ms, tokens: {sign}{token_change})", Color.YELLOW, Color.BOLD))
        self._print(c("  " + "-" * 50, Color.YELLOW))

        # Show new prompt (first 300 chars)
        preview = new_prompt[:300].replace('\n', '\n  ')
        self._print(c("  ", Color.YELLOW) + preview)
        if len(new_prompt) > 300:
            self._print(c(f"  ... ({len(new_prompt) - 300} more chars)", Color.DIM))

        self._print("")

    def fitness_summary(self, prompt_id: str, fitness: float, train_scores: list, test_scores: list):
        """Log fitness calculation summary."""
        train_avg = sum(train_scores) / len(train_scores) if train_scores else 0
        test_avg = sum(test_scores) / len(test_scores) if test_scores else 0
        gap = train_avg - test_avg

        fitness_color = Color.GREEN if fitness >= 7 else (Color.YELLOW if fitness >= 5 else Color.RED)

        self._print(
            c(f"  Fitness [{prompt_id}]: ", Color.WHITE) +
            c(f"{fitness:.3f}", fitness_color, Color.BOLD) +
            c(f" (train: {train_avg:.2f}, test: {test_avg:.2f}, gap: {gap:.2f})", Color.DIM)
        )

    def iteration_summary(self, iteration: int, total: int, best_fitness: float, avg_fitness: float, elapsed: float):
        """Log iteration summary."""
        self._print("")
        self._print(c(f"  ITERATION {iteration}/{total} COMPLETE", Color.BRIGHT_WHITE, Color.BOLD))
        self._print(c("  " + "=" * 50, Color.BRIGHT_WHITE))

        fitness_color = Color.GREEN if best_fitness >= 7 else (Color.YELLOW if best_fitness >= 5 else Color.RED)

        self._print(
            c("  Best: ", Color.WHITE) + c(f"{best_fitness:.3f}", fitness_color, Color.BOLD) +
            c(f"  Avg: {avg_fitness:.3f}  Time: {elapsed:.1f}s", Color.DIM)
        )
        self._print("")

    def final_summary(self, best_prompt: str, best_fitness: float, total_cost: float, mutation_history: list):
        """Log final optimization summary."""
        self._print("")
        self.header("OPTIMIZATION COMPLETE", Color.GREEN)

        self._print(c(f"\n  Best Fitness: ", Color.WHITE) + c(f"{best_fitness:.3f}", Color.GREEN, Color.BOLD))
        self._print(c(f"  Total Cost: ", Color.WHITE) + c(f"${total_cost:.4f}", Color.CYAN))
        self._print(c(f"  Mutation Path: ", Color.WHITE) + c(" -> ".join(mutation_history) or "base", Color.YELLOW))

        self._print(c("\n  BEST PROMPT:", Color.GREEN, Color.BOLD))
        self._print(c("  " + "-" * 60, Color.GREEN))

        for line in best_prompt.split('\n'):
            self._print(c("  ", Color.GREEN) + line)

        self._print(c("  " + "-" * 60, Color.GREEN))
        self._print("")


# Global logger instance
_logger: Optional[ConsoleLogger] = None


def get_logger(verbose: bool = True) -> ConsoleLogger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = ConsoleLogger(verbose=verbose)
    return _logger


def log_callback_factory(verbose: bool = True):
    """Create a log callback function for use with optimizers."""
    logger = get_logger(verbose)

    def callback(level: str, message: str):
        level = level.lower()
        if level == "info":
            logger.info(message)
        elif level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "success":
            logger.success(message)
        else:
            logger.info(message)

    return callback
