"""
Configuration management for ~/.repr/ directory.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================================
# API Configuration
# ============================================================================

# Environment detection
_DEV_MODE = os.getenv("REPR_DEV", "").lower() in ("1", "true", "yes")

# Production URLs
PROD_API_BASE = "https://api.repr.dev/api/cli"

# Local development URLs
LOCAL_API_BASE = "http://localhost:8003/api/cli"


def get_api_base() -> str:
    """Get the API base URL based on environment."""
    if env_url := os.getenv("REPR_API_BASE"):
        return env_url
    return LOCAL_API_BASE if _DEV_MODE else PROD_API_BASE


def is_dev_mode() -> bool:
    """Check if running in dev mode."""
    return _DEV_MODE


def set_dev_mode(enabled: bool) -> None:
    """Set dev mode programmatically (for CLI --dev flag)."""
    global _DEV_MODE
    _DEV_MODE = enabled


# ============================================================================
# File Configuration
# ============================================================================

CONFIG_DIR = Path.home() / ".repr"
CONFIG_FILE = CONFIG_DIR / "config.json"
PROFILES_DIR = CONFIG_DIR / "profiles"
CACHE_DIR = CONFIG_DIR / "cache"
REPO_HASHES_FILE = CACHE_DIR / "repo-hashes.json"

DEFAULT_CONFIG = {
    "version": 1,
    "auth": None,
    "settings": {
        "default_paths": ["~/code"],
        "skip_patterns": ["node_modules", "venv", ".venv", "vendor", "__pycache__", ".git"],
    },
    "sync": {
        "last_pushed": None,
        "last_profile": None,
    },
    "llm": {
        "extraction_model": None,  # Model for extracting accomplishments (e.g., "gpt-4o-mini", "llama3.2")
        "synthesis_model": None,  # Model for synthesizing profile (e.g., "gpt-4o", "llama3.2")
        "local_api_url": None,  # Local LLM API base URL (e.g., "http://localhost:11434/v1")
        "local_api_key": None,  # Local LLM API key (often "ollama" for Ollama)
    },
}


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    CONFIG_DIR.mkdir(exist_ok=True)
    PROFILES_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load configuration from disk, creating default if missing."""
    ensure_directories()
    
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            # Merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **config}
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to disk."""
    ensure_directories()
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)


def get_auth() -> dict[str, Any] | None:
    """Get authentication info if available."""
    config = load_config()
    return config.get("auth")


def set_auth(
    access_token: str,
    user_id: str,
    email: str,
    litellm_api_key: str | None = None,
) -> None:
    """Store authentication info."""
    config = load_config()
    config["auth"] = {
        "access_token": access_token,
        "user_id": user_id,
        "email": email,
        "authenticated_at": datetime.now().isoformat(),
    }
    if litellm_api_key:
        config["auth"]["litellm_api_key"] = litellm_api_key
    save_config(config)


def clear_auth() -> None:
    """Clear authentication info."""
    config = load_config()
    config["auth"] = None
    save_config(config)


def is_authenticated() -> bool:
    """Check if user is authenticated."""
    auth = get_auth()
    return auth is not None and auth.get("access_token") is not None


def get_access_token() -> str | None:
    """Get access token if authenticated."""
    auth = get_auth()
    return auth.get("access_token") if auth else None


def get_litellm_config() -> tuple[str | None, str | None]:
    """Get LiteLLM configuration if available.
    
    Returns:
        Tuple of (litellm_url, litellm_api_key)
    """
    auth = get_auth()
    if not auth:
        return None, None
    return auth.get("litellm_url"), auth.get("litellm_api_key")


# ============================================================================
# LLM Configuration
# ============================================================================

def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration.
    
    Returns:
        Dict with extraction_model, synthesis_model, local_api_url, local_api_key
    """
    config = load_config()
    llm = config.get("llm", {})
    return {
        "extraction_model": llm.get("extraction_model"),
        "synthesis_model": llm.get("synthesis_model"),
        "local_api_url": llm.get("local_api_url"),
        "local_api_key": llm.get("local_api_key"),
    }


def set_llm_config(
    extraction_model: str | None = None,
    synthesis_model: str | None = None,
    local_api_url: str | None = None,
    local_api_key: str | None = None,
) -> None:
    """Set LLM configuration.
    
    Only updates provided values, leaves others unchanged.
    """
    config = load_config()
    if "llm" not in config:
        config["llm"] = DEFAULT_CONFIG["llm"].copy()
    
    if extraction_model is not None:
        config["llm"]["extraction_model"] = extraction_model if extraction_model else None
    if synthesis_model is not None:
        config["llm"]["synthesis_model"] = synthesis_model if synthesis_model else None
    if local_api_url is not None:
        config["llm"]["local_api_url"] = local_api_url if local_api_url else None
    if local_api_key is not None:
        config["llm"]["local_api_key"] = local_api_key if local_api_key else None
    
    save_config(config)


def clear_llm_config() -> None:
    """Clear all LLM configuration."""
    config = load_config()
    config["llm"] = DEFAULT_CONFIG["llm"].copy()
    save_config(config)


def get_skip_patterns() -> list[str]:
    """Get list of patterns to skip during discovery."""
    config = load_config()
    return config.get("settings", {}).get("skip_patterns", DEFAULT_CONFIG["settings"]["skip_patterns"])


def update_sync_info(profile_name: str) -> None:
    """Update last sync information."""
    config = load_config()
    config["sync"] = {
        "last_pushed": datetime.now().isoformat(),
        "last_profile": profile_name,
    }
    save_config(config)


def get_sync_info() -> dict[str, Any]:
    """Get sync information."""
    config = load_config()
    return config.get("sync", {})


# Profile management

def list_profiles() -> list[dict[str, Any]]:
    """List all saved profiles with metadata, sorted by modification time (newest first)."""
    ensure_directories()
    
    profiles = []
    for profile_path in sorted(PROFILES_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        content = profile_path.read_text()
        
        # Extract basic stats from content
        project_count = content.count("## ") - 1  # Subtract header sections
        if project_count < 0:
            project_count = 0
        
        # Check if synced
        sync_info = get_sync_info()
        is_synced = sync_info.get("last_profile") == profile_path.name
        
        # Load metadata if exists
        metadata = get_profile_metadata(profile_path.stem)
        
        profiles.append({
            "name": profile_path.stem,
            "filename": profile_path.name,
            "path": profile_path,
            "size": profile_path.stat().st_size,
            "modified": datetime.fromtimestamp(profile_path.stat().st_mtime),
            "project_count": project_count,
            "synced": is_synced,
            "repos": metadata.get("repos", []) if metadata else [],
        })
    
    return profiles


def get_latest_profile() -> Path | None:
    """Get path to the latest profile."""
    profiles = list_profiles()
    return profiles[0]["path"] if profiles else None


def get_profile(name: str) -> Path | None:
    """Get path to a specific profile by name."""
    profile_path = PROFILES_DIR / f"{name}.md"
    return profile_path if profile_path.exists() else None


def get_profile_metadata(name: str) -> dict[str, Any] | None:
    """Get metadata for a specific profile by name."""
    metadata_path = PROFILES_DIR / f"{name}.meta.json"
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_profile(content: str, name: str | None = None, repos: list[dict[str, Any]] | None = None) -> Path:
    """Save a profile to disk with optional metadata (repos as rich objects)."""
    ensure_directories()
    
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d")
    
    # Handle duplicate names by adding suffix
    profile_path = PROFILES_DIR / f"{name}.md"
    counter = 1
    while profile_path.exists():
        profile_path = PROFILES_DIR / f"{name}-{counter}.md"
        counter += 1
    
    profile_path.write_text(content)
    
    # Save metadata if repos provided
    if repos is not None:
        metadata_path = profile_path.with_suffix('.meta.json')
        metadata = {
            "repos": repos,
            "created_at": datetime.now().isoformat(),
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return profile_path


def save_repo_profile(content: str, repo_name: str, repo_metadata: dict[str, Any]) -> Path:
    """Save a per-repo profile to disk as {repo_name}_{date}.md."""
    ensure_directories()
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    name = f"{repo_name}_{date_str}"
    
    profile_path = PROFILES_DIR / f"{name}.md"
    counter = 1
    while profile_path.exists():
        profile_path = PROFILES_DIR / f"{name}-{counter}.md"
        counter += 1
    
    profile_path.write_text(content)
    
    metadata_path = profile_path.with_suffix('.meta.json')
    metadata = {
        "repo": repo_metadata,
        "created_at": datetime.now().isoformat(),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return profile_path


# Cache management

def load_repo_hashes() -> dict[str, str]:
    """Load cached repository hashes."""
    ensure_directories()
    
    if not REPO_HASHES_FILE.exists():
        return {}
    
    try:
        with open(REPO_HASHES_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_repo_hashes(hashes: dict[str, str]) -> None:
    """Save repository hashes to cache."""
    ensure_directories()
    
    with open(REPO_HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def get_repo_hash(repo_path: str) -> str | None:
    """Get cached hash for a repository."""
    hashes = load_repo_hashes()
    return hashes.get(repo_path)


def set_repo_hash(repo_path: str, hash_value: str) -> None:
    """Set cached hash for a repository."""
    hashes = load_repo_hashes()
    hashes[repo_path] = hash_value
    save_repo_hashes(hashes)


def clear_cache() -> None:
    """Clear all cached data."""
    if REPO_HASHES_FILE.exists():
        REPO_HASHES_FILE.unlink()
