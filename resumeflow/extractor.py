"""
Extract signals from repositories - languages, dependencies, etc.
"""

import json
import re
from collections import Counter
from pathlib import Path

from pygments.lexers import get_lexer_for_filename, ClassNotFound


# Language detection by file extension
LANGUAGE_EXTENSIONS = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".swift": "Swift",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C",
    ".hpp": "C++",
    ".rb": "Ruby",
    ".php": "PHP",
    ".cs": "C#",
    ".scala": "Scala",
    ".clj": "Clojure",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".erl": "Erlang",
    ".hs": "Haskell",
    ".lua": "Lua",
    ".r": "R",
    ".R": "R",
    ".jl": "Julia",
    ".dart": "Dart",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".sql": "SQL",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
}

# Files to skip when detecting languages
SKIP_PATTERNS = {
    "node_modules",
    "venv",
    ".venv",
    "vendor",
    "__pycache__",
    ".git",
    "dist",
    "build",
    ".next",
    "target",
    "coverage",
}


def detect_languages(repo_path: Path) -> dict[str, float]:
    """
    Detect languages used in a repository.
    
    Args:
        repo_path: Path to repository
    
    Returns:
        Dictionary of language -> percentage
    """
    extension_counts: Counter[str] = Counter()
    total_files = 0
    
    try:
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Skip certain directories
            parts = file_path.relative_to(repo_path).parts
            if any(skip in parts for skip in SKIP_PATTERNS):
                continue
            
            # Get language from extension
            ext = file_path.suffix.lower()
            if ext in LANGUAGE_EXTENSIONS:
                extension_counts[ext] += 1
                total_files += 1
    except Exception:
        pass
    
    if total_files == 0:
        return {}
    
    # Convert to language percentages
    languages: Counter[str] = Counter()
    for ext, count in extension_counts.items():
        language = LANGUAGE_EXTENSIONS[ext]
        languages[language] += count
    
    # Calculate percentages
    result = {}
    for language, count in languages.most_common():
        percentage = (count / total_files) * 100
        if percentage >= 1:  # Only include if >= 1%
            result[language] = round(percentage, 1)
    
    return result


def get_primary_language(repo_path: Path) -> str | None:
    """
    Get the primary language of a repository.
    
    Args:
        repo_path: Path to repository
    
    Returns:
        Primary language name or None
    """
    languages = detect_languages(repo_path)
    if languages:
        return max(languages, key=languages.get)
    return None


def detect_dependencies(repo_path: Path) -> dict[str, list[str]]:
    """
    Detect dependencies from package files.
    
    Args:
        repo_path: Path to repository
    
    Returns:
        Dictionary of ecosystem -> list of dependency names
    """
    dependencies: dict[str, list[str]] = {}
    
    # Python - requirements.txt
    requirements_file = repo_path / "requirements.txt"
    if requirements_file.exists():
        deps = _parse_requirements_txt(requirements_file)
        if deps:
            dependencies["python"] = deps
    
    # Python - pyproject.toml
    pyproject_file = repo_path / "pyproject.toml"
    if pyproject_file.exists():
        deps = _parse_pyproject_toml(pyproject_file)
        if deps:
            dependencies.setdefault("python", []).extend(deps)
            dependencies["python"] = list(set(dependencies["python"]))
    
    # Node.js - package.json
    package_json = repo_path / "package.json"
    if package_json.exists():
        deps = _parse_package_json(package_json)
        if deps:
            dependencies["nodejs"] = deps
    
    # Go - go.mod
    go_mod = repo_path / "go.mod"
    if go_mod.exists():
        deps = _parse_go_mod(go_mod)
        if deps:
            dependencies["go"] = deps
    
    # Rust - Cargo.toml
    cargo_toml = repo_path / "Cargo.toml"
    if cargo_toml.exists():
        deps = _parse_cargo_toml(cargo_toml)
        if deps:
            dependencies["rust"] = deps
    
    # Ruby - Gemfile
    gemfile = repo_path / "Gemfile"
    if gemfile.exists():
        deps = _parse_gemfile(gemfile)
        if deps:
            dependencies["ruby"] = deps
    
    return dependencies


def _parse_requirements_txt(path: Path) -> list[str]:
    """Parse Python requirements.txt file."""
    deps = []
    try:
        content = path.read_text()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Extract package name (before version specifier)
            match = re.match(r"^([a-zA-Z0-9_-]+)", line)
            if match:
                deps.append(match.group(1).lower())
    except Exception:
        pass
    return deps


def _parse_pyproject_toml(path: Path) -> list[str]:
    """Parse Python pyproject.toml dependencies."""
    deps = []
    try:
        content = path.read_text()
        # Simple parsing - look for dependencies array
        in_deps = False
        for line in content.splitlines():
            if "dependencies" in line and "=" in line:
                in_deps = True
                continue
            if in_deps:
                if line.strip().startswith("]"):
                    in_deps = False
                    continue
                # Extract package name
                match = re.search(r'"([a-zA-Z0-9_-]+)', line)
                if match:
                    deps.append(match.group(1).lower())
    except Exception:
        pass
    return deps


def _parse_package_json(path: Path) -> list[str]:
    """Parse Node.js package.json dependencies."""
    deps = []
    try:
        content = json.loads(path.read_text())
        for key in ["dependencies", "devDependencies"]:
            if key in content:
                deps.extend(content[key].keys())
    except Exception:
        pass
    return deps


def _parse_go_mod(path: Path) -> list[str]:
    """Parse Go go.mod dependencies."""
    deps = []
    try:
        content = path.read_text()
        in_require = False
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if in_require:
                if line == ")":
                    in_require = False
                    continue
                # Extract module path
                parts = line.split()
                if parts:
                    deps.append(parts[0])
            elif line.startswith("require "):
                parts = line.split()
                if len(parts) >= 2:
                    deps.append(parts[1])
    except Exception:
        pass
    return deps


def _parse_cargo_toml(path: Path) -> list[str]:
    """Parse Rust Cargo.toml dependencies."""
    deps = []
    try:
        content = path.read_text()
        in_deps = False
        for line in content.splitlines():
            line = line.strip()
            if line == "[dependencies]" or line == "[dev-dependencies]":
                in_deps = True
                continue
            if line.startswith("[") and in_deps:
                in_deps = False
                continue
            if in_deps and "=" in line:
                name = line.split("=")[0].strip()
                if name and not name.startswith("#"):
                    deps.append(name)
    except Exception:
        pass
    return deps


def _parse_gemfile(path: Path) -> list[str]:
    """Parse Ruby Gemfile dependencies."""
    deps = []
    try:
        content = path.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("gem "):
                # Extract gem name
                match = re.search(r"gem ['\"]([^'\"]+)['\"]", line)
                if match:
                    deps.append(match.group(1))
    except Exception:
        pass
    return deps


def get_file_tree(repo_path: Path, max_depth: int = 3) -> dict:
    """
    Get the file tree structure of a repository.
    
    Args:
        repo_path: Path to repository
        max_depth: Maximum depth to traverse
    
    Returns:
        Nested dictionary representing file tree
    """
    def build_tree(path: Path, depth: int = 0) -> dict | str:
        if depth > max_depth:
            return "..."
        
        if path.is_file():
            return path.name
        
        result = {}
        try:
            for item in sorted(path.iterdir()):
                # Skip hidden and common skip patterns
                if item.name.startswith(".") or item.name in SKIP_PATTERNS:
                    continue
                
                if item.is_dir():
                    subtree = build_tree(item, depth + 1)
                    if subtree:  # Only include non-empty directories
                        result[item.name + "/"] = subtree
                else:
                    result[item.name] = None
        except PermissionError:
            pass
        
        return result
    
    return build_tree(repo_path)


def get_file_tree_flat(repo_path: Path, max_depth: int = 3) -> list[str]:
    """
    Get a flat list of file paths in the repository.
    
    Args:
        repo_path: Path to repository
        max_depth: Maximum depth to traverse
    
    Returns:
        List of relative file paths
    """
    files = []
    
    def walk(path: Path, depth: int = 0) -> None:
        if depth > max_depth:
            return
        
        try:
            for item in sorted(path.iterdir()):
                # Skip hidden and common skip patterns
                if item.name.startswith(".") or item.name in SKIP_PATTERNS:
                    continue
                
                rel_path = str(item.relative_to(repo_path))
                
                if item.is_dir():
                    files.append(rel_path + "/")
                    walk(item, depth + 1)
                else:
                    files.append(rel_path)
        except PermissionError:
            pass
    
    walk(repo_path)
    return files

