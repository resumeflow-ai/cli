"""
Rich terminal UI components for beautiful CLI output.
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich import box

from . import __version__
from .config import is_dev_mode

# Console instance for consistent output
console = Console()

# Brand colors
BRAND_PRIMARY = "#6366f1"  # Indigo
BRAND_SUCCESS = "#22c55e"  # Green
BRAND_WARNING = "#eab308"  # Yellow
BRAND_ERROR = "#ef4444"    # Red
BRAND_INFO = "#06b6d4"     # Cyan
BRAND_MUTED = "#6b7280"    # Gray


def print_header() -> None:
    """Print the branded CLI header."""
    header_text = Text()
    header_text.append("🚀  ", style="bold")
    header_text.append("ResumeFlow CLI", style=f"bold {BRAND_PRIMARY}")
    header_text.append(f" v{__version__}", style=BRAND_MUTED)
    
    panel = Panel(
        header_text,
        box=box.ROUNDED,
        border_style=BRAND_PRIMARY,
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold {BRAND_SUCCESS}]✓[/] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold {BRAND_ERROR}]✗[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold {BRAND_WARNING}]⚠[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[{BRAND_INFO}]ℹ[/] {message}")


def print_step(message: str, completed: bool = False, pending: bool = False) -> None:
    """Print a step indicator."""
    if completed:
        symbol = f"[bold {BRAND_SUCCESS}]✓[/]"
    elif pending:
        symbol = f"[{BRAND_MUTED}]○[/]"
    else:
        symbol = f"[bold {BRAND_PRIMARY}]●[/]"
    
    console.print(f"  ├── {symbol} {message}")


def print_last_step(message: str, completed: bool = False, pending: bool = False) -> None:
    """Print the last step indicator."""
    if completed:
        symbol = f"[bold {BRAND_SUCCESS}]✓[/]"
    elif pending:
        symbol = f"[{BRAND_MUTED}]○[/]"
    else:
        symbol = f"[bold {BRAND_PRIMARY}]●[/]"
    
    console.print(f"  └── {symbol} {message}")


def print_next_steps(steps: list[str]) -> None:
    """Print next steps section."""
    console.print()
    console.print("[bold]Next steps:[/]")
    for step in steps:
        console.print(f"  [bold {BRAND_INFO}]→[/] {step}")


def create_repo_table() -> Table:
    """Create a table for repository analysis display."""
    table = Table(
        box=box.ROUNDED,
        border_style=BRAND_MUTED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Repository", style="bold")
    table.add_column("Language", style=BRAND_INFO)
    table.add_column("Commits", justify="right")
    table.add_column("Age", justify="right")
    table.add_column("Status", justify="center")
    return table


def add_repo_row(
    table: Table,
    name: str,
    language: str | None = None,
    commits: int | None = None,
    age: str | None = None,
    status: str = "pending",
) -> None:
    """Add a row to the repository table."""
    status_map = {
        "pending": f"[{BRAND_MUTED}]○ Pending[/]",
        "analyzing": f"[bold {BRAND_PRIMARY}]● Analyzing...[/]",
        "completed": f"[bold {BRAND_SUCCESS}]✓[/]",
        "skipped": f"[{BRAND_MUTED}]⊘ skipped[/]",
        "error": f"[bold {BRAND_ERROR}]✗ Error[/]",
    }
    
    table.add_row(
        name,
        language or "-",
        str(commits) if commits else "-",
        age or "-",
        status_map.get(status, status),
    )


def create_profile_table() -> Table:
    """Create a table for profile listing."""
    table = Table(
        box=box.ROUNDED,
        border_style=BRAND_MUTED,
        show_header=True,
        header_style="bold",
    )
    table.add_column("Profile", style="bold")
    table.add_column("Projects", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")
    return table


def create_analysis_progress() -> Progress:
    """Create a progress bar for analysis."""
    return Progress(
        SpinnerColumn(style=BRAND_PRIMARY),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30, style=BRAND_MUTED, complete_style=BRAND_PRIMARY),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_simple_progress() -> Progress:
    """Create a simple progress indicator."""
    return Progress(
        SpinnerColumn(style=BRAND_PRIMARY),
        TextColumn("[bold]{task.description}"),
        console=console,
    )


def print_panel(title: str, content: str, border_color: str = BRAND_PRIMARY) -> None:
    """Print content in a panel."""
    panel = Panel(
        content,
        title=title,
        box=box.ROUNDED,
        border_style=border_color,
        padding=(1, 2),
    )
    console.print(panel)


def print_markdown(content: str) -> None:
    """Print markdown content."""
    md = Markdown(content)
    console.print(md)


def print_profile_preview(content: str, max_lines: int = 15) -> None:
    """Print a preview of a profile."""
    lines = content.split("\n")
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += "\n\n..."
    
    panel = Panel(
        Markdown(preview),
        box=box.ROUNDED,
        border_style=BRAND_PRIMARY,
        padding=(1, 2),
    )
    console.print(panel)


def print_auth_code(code: str) -> None:
    """Print the authentication code prominently."""
    code_text = Text(code, style=f"bold {BRAND_PRIMARY}")
    panel = Panel(
        code_text,
        box=box.ROUNDED,
        border_style=BRAND_PRIMARY,
        padding=(0, 4),
    )
    console.print()
    console.print(panel, justify="center")
    console.print()


def print_connection_status(connected: bool) -> None:
    """Print WebSocket connection status."""
    host = "localhost:8003" if is_dev_mode() else "resumeflow.dev"
    if connected:
        console.print(f"Connected to {host} [{BRAND_SUCCESS}]●[/]")
    else:
        console.print(f"Disconnected from {host} [{BRAND_ERROR}]●[/]")


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_duration(months: int) -> str:
    """Format duration in months to human readable string."""
    if months < 1:
        return "< 1 mo"
    elif months < 12:
        return f"{months} mo"
    else:
        years = months // 12
        remaining_months = months % 12
        if remaining_months == 0:
            return f"{years} yr"
        return f"{years} yr {remaining_months} mo"


class _AnimatedRenderable:
    """A renderable wrapper that calls the render function on each refresh."""
    
    def __init__(self, render_func):
        self.render_func = render_func
    
    def __rich_console__(self, console, options):
        yield self.render_func()


class AnalysisDisplay:
    """Live display for analysis progress with animations."""
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PROGRESS_CHARS = ["░", "▒", "▓", "█"]
    
    def __init__(self):
        self.repos: list[dict] = []
        self.current_step = ""
        self.current_detail = ""
        self.current_repo = ""
        self.progress_pct: float = 0.0
        self.connected = False
        self.live: Live | None = None
        self._frame: int = 0
        self._tick: int = 0
    
    def _get_spinner(self) -> str:
        return self.SPINNER_FRAMES[self._frame % len(self.SPINNER_FRAMES)]
    
    def _get_progress_bar(self, width: int = 30) -> Text:
        filled = int(self.progress_pct / 100 * width)
        shimmer_pos = self._tick % (width + 5)
        
        bar = Text()
        for i in range(width):
            if i < filled:
                if i == shimmer_pos or i == shimmer_pos - 1:
                    bar.append("█", style=f"bold {BRAND_PRIMARY}")
                else:
                    bar.append("█", style=BRAND_PRIMARY)
            elif i == filled and filled < width:
                partial_idx = int((self.progress_pct % (100/width)) / (100/width) * 4)
                bar.append(self.PROGRESS_CHARS[min(partial_idx, 3)], style=BRAND_PRIMARY)
            else:
                bar.append("░", style=BRAND_MUTED)
        
        return bar
    
    def _get_animated_status(self, status: str) -> str:
        if status == "analyzing":
            spinner = self._get_spinner()
            colors = [BRAND_PRIMARY, "#818cf8", "#a5b4fc", "#818cf8"]
            color = colors[self._frame % len(colors)]
            return f"[bold {color}]{spinner} Analyzing[/]"
        elif status == "pending":
            dots = "." * ((self._tick // 2) % 4)
            return f"[{BRAND_MUTED}]○ Waiting{dots}[/]"
        elif status == "completed":
            return f"[bold {BRAND_SUCCESS}]✓ Done[/]"
        elif status == "skipped":
            return f"[{BRAND_MUTED}]⊘ Skipped[/]"
        elif status == "error":
            return f"[bold {BRAND_ERROR}]✗ Error[/]"
        return status
    
    def _render(self):
        self._frame = (self._frame + 1) % len(self.SPINNER_FRAMES)
        self._tick += 1
        
        lines = []
        
        if self.current_step:
            spinner = self._get_spinner()
            
            # Map step names to nice icons
            step_icons = {
                "Starting": "🚀",
                "Extracting": "📂",
                "Preparing": "📋",
                "Analyzing": "🔍",
                "Synthesizing": "✨",
                "Merging": "🔗",
                "Finalizing": "📝",
                "Complete": "✅",
            }
            icon = step_icons.get(self.current_step, "")
            
            step_text = self.current_step.replace("_", " ").title()
            status_line = Text()
            status_line.append(f"{spinner} ", style=f"bold {BRAND_PRIMARY}")
            if icon:
                status_line.append(f"{icon} ", style="")
            status_line.append(step_text, style="bold")
            
            if self.current_repo:
                status_line.append(" → ", style=BRAND_MUTED)
                status_line.append(self.current_repo, style=BRAND_INFO)
            
            lines.append(status_line)
            
            if self.current_detail:
                detail_line = Text()
                detail_line.append("   ", style="")
                detail_line.append(self.current_detail, style=BRAND_MUTED)
                lines.append(detail_line)
            
            lines.append(Text())
        
        # Always show progress bar once we have a step (even at 0%)
        if self.current_step:
            progress_line = Text()
            progress_line.append("   ")
            progress_line.append_text(self._get_progress_bar(width=40))
            progress_line.append(f" {self.progress_pct:.0f}%", style=f"bold {BRAND_PRIMARY}")
            lines.append(progress_line)
            lines.append(Text())
        
        return Group(*lines) if lines else Text("")
    
    def start(self) -> None:
        """Start the live display."""
        # Use _AnimatedRenderable so Rich calls _render() on each refresh cycle
        self.live = Live(
            _AnimatedRenderable(self._render), 
            console=console, 
            refresh_per_second=10,
            transient=False,
        )
        self.live.start()
    
    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
    
    def update_progress(
        self, 
        step: str | None = None,
        detail: str | None = None,
        repo: str | None = None,
        progress: float | None = None,
    ) -> None:
        """Update progress information."""
        if step is not None:
            self.current_step = step
        if detail is not None:
            self.current_detail = detail
        if repo is not None:
            self.current_repo = repo
        if progress is not None:
            self.progress_pct = progress
    
    def update_repo(self, name: str, **kwargs) -> None:
        """Update a repository's display status."""
        for repo in self.repos:
            if repo["name"] == name:
                repo.update(kwargs)
                break
        else:
            self.repos.append({"name": name, **kwargs})
    
    def set_repos(self, repos: list[dict]) -> None:
        """Set all repositories."""
        self.repos = repos

