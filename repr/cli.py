"""
Repr CLI - Main command-line interface.

Commands:
    analyze     Analyze repositories and generate profiles (auto-pushed)
    push        Push local profiles to repr.dev
    view        View the latest profile
    login       Authenticate with Repr
    logout      Clear authentication
    profiles    List saved profiles
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from . import __version__
from .config import (
    list_profiles,
    get_latest_profile,
    get_profile,
    get_profile_metadata,
    save_profile,
    save_repo_profile,
    get_sync_info,
    update_sync_info,
    clear_cache,
    is_authenticated,
    CONFIG_DIR,
    PROFILES_DIR,
    set_dev_mode,
    is_dev_mode,
    get_api_base,
    get_litellm_config,
    get_llm_config,
    set_llm_config,
    clear_llm_config,
)
from .discovery import discover_repos, is_config_only_repo, RepoInfo
from .extractor import get_primary_language, detect_languages
from .auth import AuthFlow, AuthError, logout as auth_logout, get_current_user
from .api import push_repo_profile, APIError
from .openai_analysis import analyze_repo_openai, DEFAULT_EXTRACTION_MODEL, DEFAULT_SYNTHESIS_MODEL
from .ui import (
    console,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_next_steps,
    print_panel,
    print_markdown,
    print_auth_code,
    create_profile_table,
    create_simple_progress,
    format_bytes,
    AnalysisDisplay,
    BRAND_PRIMARY,
    BRAND_SUCCESS,
    BRAND_MUTED,
)
from .analyzer import analyze_repo
from .highlights import get_highlights

# Create Typer app
app = typer.Typer(
    name="repr",
    help="🚀 Privacy-first CLI that analyzes your code and generates a developer profile.",
    add_completion=False,
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        console.print(f"Repr CLI v{__version__}")
        raise typer.Exit()


def dev_callback(value: bool):
    if value:
        set_dev_mode(True)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    dev: bool = typer.Option(
        False,
        "--dev",
        callback=dev_callback,
        is_eager=True,
        help="Use localhost backend (http://localhost:8003).",
    ),
):
    """Repr CLI - Privacy-first developer profile generator."""
    pass


@app.command()
def analyze(
    paths: List[Path] = typer.Argument(
        ...,
        help="Paths to directories containing git repositories.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show detailed logs during analysis.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Re-analyze all repositories (ignore cache).",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local LLM (Ollama) instead of cloud.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Local model to use (default: llama3.2).",
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        help="Custom local LLM API endpoint.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Stats only, no AI analysis (no push).",
    ),
    min_commits: int = typer.Option(
        10,
        "--min-commits",
        help="Minimum number of commits required.",
    ),
):
    """
    Analyze repositories and generate profiles (one per repo).
    
    Each repository is analyzed separately, saved as {repo}_{date}.md,
    and pushed to repr.dev immediately. Process halts on any failure.
    
    Examples:
        repr analyze ~/code                              # OpenAI (default)
        repr analyze ~/code --local --model llama3.2     # Local LLM (Ollama)
        repr analyze ~/code --offline                    # Stats only, no push
    """
    print_header()
    
    mode = "offline" if offline else ("local" if local else "openai")
    llm_config = get_llm_config()
    
    local_api_key = None
    local_base_url = None
    local_extraction_model = None
    local_synthesis_model = None
    
    # Require authentication for non-offline modes (auto-push)
    if mode != "offline":
        if not is_authenticated():
            print_error("Not logged in. Please run 'repr login' first.")
            raise typer.Exit(1)
    
    if mode == "openai":
        _, litellm_key = get_litellm_config()
        if not litellm_key:
            print_error("Not authenticated for LLM access. Please run 'repr login' first.")
            raise typer.Exit(1)
    
    if mode == "local":
        local_api_key = llm_config["local_api_key"] or "ollama"
        local_base_url = api_base or llm_config["local_api_url"] or "http://localhost:11434/v1"
        local_extraction_model = model or llm_config["extraction_model"] or "llama3.2"
        local_synthesis_model = model or llm_config["synthesis_model"] or "llama3.2"
        print_info(f"Using local LLM: {local_extraction_model} at {local_base_url}")
    
    if no_cache:
        clear_cache()
        if verbose:
            print_info("Cache cleared")
    
    console.print("Discovering repositories...")
    console.print()
    
    with create_simple_progress() as progress:
        task = progress.add_task("Scanning directories...", total=None)
        repos = discover_repos(root_paths=paths, use_cache=not no_cache, min_commits=min_commits)
    
    if not repos:
        print_warning("No repositories found in the specified paths.")
        print_info(f"Make sure the paths contain git repositories with at least {min_commits} commits.")
        raise typer.Exit(1)
    
    total_paths = sum(1 for p in paths)
    console.print(f"Found [bold]{len(repos)}[/bold] repositories in {total_paths} path(s)")
    console.print()
    
    for repo in repos:
        languages = detect_languages(repo.path)
        repo.languages = {lang: int(pct) for lang, pct in languages.items()}
        repo.primary_language = max(languages, key=languages.get) if languages else None
    
    filtered_repos = []
    skipped_repos = []
    
    for repo in repos:
        if is_config_only_repo(repo.path):
            skipped_repos.append(repo)
        else:
            filtered_repos.append(repo)
    
    if skipped_repos:
        console.print(f"Skipping {len(skipped_repos)} config-only repos: {', '.join(r.name for r in skipped_repos)}")
        console.print()
    
    if not filtered_repos:
        print_warning("No analyzable repositories found (all were config-only).")
        raise typer.Exit(1)
    
    console.print(f"Analyzing {len(filtered_repos)} repositories...")
    console.print()
    
    # Process each repo individually
    for i, repo in enumerate(filtered_repos, 1):
        console.print(f"[bold][{i}/{len(filtered_repos)}] {repo.name}[/bold]")
        
        repo_metadata = {
            "remote_url": repo.remote_url,
            "commit_count": repo.commit_count,
            "user_commit_count": repo.user_commit_count,
            "primary_language": repo.primary_language,
            "languages": repo.languages,
            "contributors": repo.contributors,
            "first_commit_date": repo.first_commit_date.isoformat() if repo.first_commit_date else None,
            "last_commit_date": repo.last_commit_date.isoformat() if repo.last_commit_date else None,
            "is_fork": repo.is_fork,
            "description": repo.description,
            "frameworks": repo.frameworks,
            "has_tests": repo.has_tests,
            "has_ci": repo.has_ci,
        }
        
        # Generate profile for this single repo
        if mode == "offline":
            profile_content = _generate_single_repo_offline_profile(repo, verbose=verbose)
        else:
            profile_content = asyncio.run(_run_single_repo_analysis(
                repo,
                api_key=local_api_key,
                base_url=local_base_url,
                extraction_model=local_extraction_model or llm_config["extraction_model"],
                synthesis_model=local_synthesis_model or llm_config["synthesis_model"],
                verbose=verbose,
            ))
        
        if not profile_content:
            print_error(f"Failed to generate profile for {repo.name}. Halting.")
            raise typer.Exit(1)
        
        # Save locally as {repo}_{date}.md
        profile_path = save_repo_profile(profile_content, repo.name, repo_metadata)
        console.print(f"  Saved: {profile_path.name}")
        
        # Push to backend (skip for offline mode)
        if mode != "offline":
            try:
                result = asyncio.run(push_repo_profile(profile_content, repo.name, repo_metadata))
                console.print(f"  [green]Pushed to repr.dev[/green]")
                
                # Mark as synced
                update_sync_info(profile_path.name)
            except (APIError, AuthError) as e:
                print_error(f"Failed to push {repo.name}: {e}. Halting.")
                raise typer.Exit(1)
        
        console.print()
    
    print_success(f"Completed: {len(filtered_repos)} repositories analyzed and pushed")
    console.print()
    print_next_steps(["Run [bold]repr profiles[/] to see all saved profiles"])


def _generate_single_repo_offline_profile(repo: RepoInfo, verbose: bool = False) -> str:
    """Generate offline profile for a single repository."""
    lines = [f"# {repo.name}", ""]
    
    if verbose:
        console.print(f"  Analyzing {repo.name}...", style=BRAND_MUTED)
    
    try:
        analysis = analyze_repo(repo.path)
        highlights = get_highlights(repo.path)
    except Exception as e:
        if verbose:
            console.print(f"    Error: {e}", style="red")
        return f"# {repo.name}\n\nFailed to analyze repository: {e}"
    
    meta = []
    if repo.primary_language:
        meta.append(f"**{repo.primary_language}**")
    meta.append(f"{repo.commit_count} commits")
    meta.append(repo.age_display)
    lines.append(" | ".join(meta))
    lines.append("")
    
    if highlights and highlights.get("domain"):
        lines.append(f"*{highlights['domain'].replace('-', ' ').title()} Application*")
        lines.append("")
    
    frameworks = analysis.get("frameworks", [])
    if frameworks:
        lines.append(f"**Frameworks:** {', '.join(frameworks)}")
    
    code_lines = analysis.get("summary", {}).get("code_lines", 0)
    lines.append(f"**Codebase:** {code_lines:,} lines of code")
    
    has_tests = analysis.get("testing", {}).get("has_tests", False)
    lines.append(f"**Testing:** {'Has tests' if has_tests else 'No tests detected'}")
    lines.append("")
    
    if highlights and highlights.get("features"):
        lines.append("## What was built")
        lines.append("")
        for f in highlights["features"][:6]:
            lines.append(f"- {f['description']}")
        lines.append("")
    
    if highlights and highlights.get("integrations"):
        integration_names = [i["name"] for i in highlights["integrations"][:8]]
        lines.append(f"**Integrations:** {', '.join(integration_names)}")
        lines.append("")
    
    if highlights and highlights.get("achievements"):
        lines.append("## Key contributions")
        lines.append("")
        for achievement in highlights["achievements"][:5]:
            clean = achievement.strip()
            if len(clean) > 100:
                clean = clean[:97] + "..."
            lines.append(f"- {clean}")
        lines.append("")
    
    lines.append("---")
    lines.append("*Generated by Repr CLI (offline mode)*")
    
    return "\n".join(lines)


async def _run_single_repo_analysis(
    repo: RepoInfo,
    api_key: str = None,
    base_url: str = None,
    extraction_model: str = None,
    synthesis_model: str = None,
    verbose: bool = False,
) -> str | None:
    """Run analysis for a single repository using OpenAI-compatible API."""
    display = AnalysisDisplay()
    display.repos.append({
        "name": repo.name,
        "language": repo.primary_language or "-",
        "commits": str(repo.commit_count),
        "age": repo.age_display,
        "status": "pending",
    })
    
    def progress_callback(step: str, detail: str, repo: str, progress: float):
        display.update_progress(step=step, detail=detail, repo=repo, progress=progress)
        for repo_data in display.repos:
            if repo_data["name"] == repo:
                if step == "Complete":
                    repo_data["status"] = "completed"
                elif step in ("Extracting", "Preparing", "Analyzing", "Synthesizing"):
                    repo_data["status"] = "analyzing"
    
    try:
        display.start()
        
        profile = await analyze_repo_openai(
            repo,
            api_key=api_key,
            base_url=base_url,
            extraction_model=extraction_model,
            synthesis_model=synthesis_model,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        
        display.stop()
        return profile
        
    except Exception as e:
        display.stop()
        print_error(f"Analysis failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc(), style="red dim")
        return None


@app.command()
def view(
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Output plain markdown (for piping).",
    ),
    profile_name: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="View a specific profile by name.",
    ),
):
    """
    View the latest profile in the terminal.
    
    Example:
        repr view
        repr view --profile myrepo_2025-12-26
        repr view --raw > profile.md
    """
    # Get profile path
    if profile_name:
        profile_path = get_profile(profile_name)
        if not profile_path:
            print_error(f"Profile not found: {profile_name}")
            raise typer.Exit(1)
    else:
        profile_path = get_latest_profile()
        if not profile_path:
            print_error("No profiles found.")
            print_info("Run 'repr analyze <paths>' to generate a profile.")
            raise typer.Exit(1)
    
    content = profile_path.read_text()
    
    if raw:
        # Raw output for piping
        print(content)
    else:
        # Pretty display
        sync_info = get_sync_info()
        is_synced = sync_info.get("last_profile") == profile_path.name
        
        status = f"[{BRAND_SUCCESS}]✓ Synced[/]" if is_synced else f"[{BRAND_MUTED}]○ Local only[/]"
        
        print_panel(
            f"Profile: {profile_path.name}",
            f"{status}",
            border_color=BRAND_PRIMARY,
        )
        console.print()
        print_markdown(content)
        console.print()
        console.print(f"[{BRAND_MUTED}]Press q to quit, ↑↓ to scroll[/]")


@app.command()
def login():
    """
    Authenticate with Repr using device code flow.
    
    Opens a browser where you can sign in, then automatically
    completes the authentication.
    
    Example:
        repr login
    """
    print_header()
    
    if is_authenticated():
        user = get_current_user()
        email = user.get("email", "unknown") if user else "unknown"
        print_info(f"Already authenticated as {email}")
        
        if not Confirm.ask("Re-authenticate?"):
            raise typer.Exit()
    
    console.print("[bold]🔐  Authenticate with Repr[/]")
    console.print()
    
    async def run_auth():
        flow = AuthFlow()
        
        def on_code_received(device_code):
            console.print("To sign in, visit:")
            console.print()
            console.print(f"  [bold {BRAND_PRIMARY}]{device_code.verification_url}[/]")
            console.print()
            console.print("And enter this code:")
            print_auth_code(device_code.user_code)
        
        def on_progress(remaining):
            mins = int(remaining // 60)
            secs = int(remaining % 60)
            # Use carriage return for updating in place
            console.print(
                f"\rWaiting for authorization... expires in {mins}:{secs:02d}  ",
                end="",
            )
        
        def on_success(token):
            console.print()
            console.print()
            print_success(f"Authenticated as {token.email}")
            console.print()
            console.print(f"  Token saved to {CONFIG_DIR / 'config.json'}")
        
        def on_error(error):
            console.print()
            print_error(str(error))
        
        flow.on_code_received = on_code_received
        flow.on_progress = on_progress
        flow.on_success = on_success
        flow.on_error = on_error
        
        try:
            await flow.run()
        except AuthError:
            raise typer.Exit(1)
    
    asyncio.run(run_auth())


@app.command()
def logout():
    """
    Clear authentication and logout.
    
    Example:
        repr logout
    """
    if not is_authenticated():
        print_info("Not currently authenticated.")
        raise typer.Exit()
    
    auth_logout()
    
    print_success("Logged out")
    console.print()
    console.print(f"  Token removed from {CONFIG_DIR / 'config.json'}")
    console.print(f"  Local profiles preserved in {PROFILES_DIR}")


@app.command()
def profiles():
    """
    List all saved profiles.
    
    Example:
        repr profiles
    """
    saved_profiles = list_profiles()
    
    if not saved_profiles:
        print_info("No profiles found.")
        print_info("Run 'repr analyze <paths>' to generate a profile.")
        raise typer.Exit()
    
    print_panel("Saved Profiles", "", border_color=BRAND_PRIMARY)
    console.print()
    
    table = create_profile_table()
    
    for profile in saved_profiles:
        status = f"[{BRAND_SUCCESS}]✓ synced[/]" if profile["synced"] else f"[{BRAND_MUTED}]○ local only[/]"
        
        table.add_row(
            profile["name"],
            str(profile["project_count"]),
            format_bytes(profile["size"]),
            status,
        )
    
    console.print(table)
    console.print()
    console.print(f"Storage: {PROFILES_DIR}")
    console.print()
    console.print("[bold]Commands:[/]")
    console.print(f"  repr view --profile {saved_profiles[0]['name']}")
    if not saved_profiles[0]["synced"]:
        console.print(f"  repr push --profile {saved_profiles[0]['name']}")


@app.command()
def push(
    profile_name: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Push a specific profile by name (default: all unsynced profiles).",
    ),
):
    """
    Push local profiles to repr.dev.
    
    By default, pushes all unsynced profiles. Use --profile to push a specific one.
    
    Examples:
        repr push                              # Push all unsynced profiles
        repr push --profile myrepo_2025-12-26  # Push specific profile
    """
    if not is_authenticated():
        print_error("Not logged in. Please run 'repr login' first.")
        raise typer.Exit(1)
    
    print_header()
    
    # Get profiles to push
    all_profiles = list_profiles()
    
    if not all_profiles:
        print_info("No profiles found.")
        print_info("Run 'repr analyze <paths>' to generate profiles first.")
        raise typer.Exit()
    
    if profile_name:
        # Push specific profile
        profiles_to_push = [p for p in all_profiles if p["name"] == profile_name]
        if not profiles_to_push:
            print_error(f"Profile not found: {profile_name}")
            raise typer.Exit(1)
    else:
        # Push all unsynced profiles
        profiles_to_push = [p for p in all_profiles if not p["synced"]]
        
        if not profiles_to_push:
            print_success("All profiles are already synced!")
            raise typer.Exit()
    
    console.print(f"Pushing {len(profiles_to_push)} profile(s) to repr.dev...")
    console.print()
    
    pushed_count = 0
    failed_count = 0
    
    for profile in profiles_to_push:
        # Read profile content
        content = profile["path"].read_text()
        
        # Load metadata if available
        metadata = get_profile_metadata(profile["name"])
        
        if metadata and "repo" in metadata:
            # This is a per-repo profile
            repo_metadata = metadata["repo"]
            repo_name = repo_metadata.get("repo_name") or profile["name"].rsplit("_", 1)[0]
            
            console.print(f"  Pushing {repo_name}...", end=" ")
            
            try:
                asyncio.run(push_repo_profile(content, repo_name, repo_metadata))
                console.print("[green]✓[/]")
                
                # Mark as synced
                update_sync_info(profile["name"])
                pushed_count += 1
                
            except (APIError, AuthError) as e:
                console.print(f"[red]✗ {e}[/]")
                failed_count += 1
        else:
            # Legacy profile without metadata - skip with warning
            console.print(f"  [yellow]⚠[/] Skipping {profile['name']} (no metadata)")
            console.print(f"    Re-analyze to create proper repo profiles")
            failed_count += 1
    
    console.print()
    
    if pushed_count > 0:
        print_success(f"Pushed {pushed_count} profile(s)")
    
    if failed_count > 0:
        print_warning(f"Failed to push {failed_count} profile(s)")
    
    if pushed_count > 0 and failed_count == 0:
        console.print()
        print_next_steps([
            "View your profiles at https://repr.dev/profile",
            "Run [bold]repr analyze <paths>[/] to update your profiles",
        ])


@app.command(name="config")
def show_config():
    """
    Show current configuration and API endpoints.
    
    Useful for debugging and verifying backend connection.
    
    Example:
        repr config
        repr --dev config
    """
    
    console.print("[bold]Repr CLI Configuration[/]")
    console.print()
    
    mode = "[green]Development (localhost)[/]" if is_dev_mode() else "[blue]Production[/]"
    console.print(f"  Mode:     {mode}")
    console.print(f"  API Base: {get_api_base()}")
    console.print()
    
    if is_authenticated():
        user = get_current_user()
        email = user.get("email", "unknown") if user else "unknown"
        console.print(f"  Auth:     [green]✓ Authenticated as {email}[/]")
    else:
        console.print(f"  Auth:     [yellow]○ Not authenticated[/]")
    
    console.print()
    
    # LLM Configuration
    llm_config = get_llm_config()
    console.print("[bold]LLM Configuration:[/]")
    extraction_model_display = llm_config['extraction_model'] or f"[dim]{DEFAULT_EXTRACTION_MODEL}[/]"
    synthesis_model_display = llm_config['synthesis_model'] or f"[dim]{DEFAULT_SYNTHESIS_MODEL}[/]"
    console.print(f"  Extraction Model: {extraction_model_display}")
    console.print(f"  Synthesis Model:  {synthesis_model_display}")
    console.print(f"  Local API URL: {llm_config['local_api_url'] or '[dim]not set[/]'}")
    console.print(f"  Local API Key: {'[green]✓ set[/]' if llm_config['local_api_key'] else '[dim]not set[/]'}")
    console.print()
    
    # Cloud LLM Access (authenticated via backend)
    _, litellm_key = get_litellm_config()
    console.print("[bold]Cloud LLM Access:[/]")
    console.print(f"  Status:  {'[green]✓ authenticated[/]' if litellm_key else '[dim]not authenticated - run repr login[/]'}")
    console.print()
    
    console.print(f"  Config:   {CONFIG_DIR / 'config.json'}")
    console.print(f"  Profiles: {PROFILES_DIR}")
    console.print()
    
    console.print("[bold]Environment Variables:[/]")
    console.print("  REPR_DEV=1         Enable dev mode (localhost)")
    console.print("  REPR_API_BASE=URL  Override API base URL")


@app.command(name="config-set")
def config_set(
    extraction_model: Optional[str] = typer.Option(
        None,
        "--extraction-model",
        help="Model for extracting accomplishments from commits (e.g., gpt-4o-mini, llama3.2).",
    ),
    synthesis_model: Optional[str] = typer.Option(
        None,
        "--synthesis-model",
        help="Model for synthesizing the final profile (e.g., gpt-4o, llama3.2).",
    ),
    local_api_url: Optional[str] = typer.Option(
        None,
        "--local-api-url",
        help="Local LLM API base URL (e.g., http://localhost:11434/v1).",
    ),
    local_api_key: Optional[str] = typer.Option(
        None,
        "--local-api-key",
        help="Local LLM API key (e.g., 'ollama' for Ollama).",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear all LLM configuration.",
    ),
):
    """
    Configure LLM models and local API settings.
    
    Settings are saved to ~/.repr/config.json and used as defaults
    for the analyze command.
    
    Examples:
        # Set models for cloud analysis
        repr config-set --extraction-model gpt-4o-mini --synthesis-model gpt-4o
        
        # Configure local LLM (Ollama)
        repr config-set --local-api-url http://localhost:11434/v1 --local-api-key ollama
        repr config-set --extraction-model llama3.2 --synthesis-model llama3.2
        
        # Configure local LLM (LM Studio)
        repr config-set --local-api-url http://localhost:1234/v1 --local-api-key lm-studio
        
        # Clear all LLM settings
        repr config-set --clear
    """
    if clear:
        clear_llm_config()
        print_success("LLM configuration cleared")
        return
    
    if not any([extraction_model, synthesis_model, local_api_url, local_api_key]):
        # No options provided, show current config
        llm_config = get_llm_config()
        console.print("[bold]Current LLM Configuration:[/]")
        console.print()
        console.print(f"  Extraction Model: {llm_config['extraction_model'] or '[dim]default[/]'}")
        console.print(f"  Synthesis Model:  {llm_config['synthesis_model'] or '[dim]default[/]'}")
        console.print(f"  Local API URL:    {llm_config['local_api_url'] or '[dim]not set[/]'}")
        console.print(f"  Local API Key:    {'[green]✓ set[/]' if llm_config['local_api_key'] else '[dim]not set[/]'}")
        console.print()
        console.print("Use [bold]repr config-set --help[/] to see available options.")
        return
    
    # Update configuration
    set_llm_config(
        extraction_model=extraction_model,
        synthesis_model=synthesis_model,
        local_api_url=local_api_url,
        local_api_key=local_api_key,
    )
    
    print_success("LLM configuration updated")
    console.print()
    
    # Show what was set
    if extraction_model:
        console.print(f"  Extraction Model: {extraction_model}")
    if synthesis_model:
        console.print(f"  Synthesis Model:  {synthesis_model}")
    if local_api_url:
        console.print(f"  Local API URL: {local_api_url}")
    if local_api_key:
        console.print(f"  Local API Key: [green]✓ set[/]")


# Entry point
if __name__ == "__main__":
    app()
