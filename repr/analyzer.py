"""
Deep code analysis module.

Performs sophisticated analysis of repositories beyond basic stats:
- Code complexity metrics
- Architecture pattern detection
- Framework/library detection from imports
- Code quality indicators
- Test coverage indicators
- Documentation analysis
- API surface detection
"""

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CodeMetrics:
    """Metrics for a single file."""
    path: str
    language: str
    lines_total: int = 0
    lines_code: int = 0
    lines_comment: int = 0
    lines_blank: int = 0
    functions: int = 0
    classes: int = 0
    imports: list[str] = field(default_factory=list)
    complexity: int = 0  # Cyclomatic complexity estimate
    avg_function_length: float = 0.0
    has_tests: bool = False
    has_docstrings: bool = False


@dataclass
class RepoAnalysis:
    """Complete analysis of a repository."""
    # Basic metrics
    total_files: int = 0
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    
    # Languages
    languages: dict[str, float] = field(default_factory=dict)
    
    # Architecture
    architecture_patterns: list[str] = field(default_factory=list)
    project_type: str = ""  # web-app, cli, library, api, etc.
    
    # Frameworks & libraries
    frameworks: list[str] = field(default_factory=list)
    notable_libraries: list[str] = field(default_factory=list)
    
    # Code quality
    avg_file_size: float = 0.0
    avg_function_length: float = 0.0
    avg_complexity: float = 0.0
    docstring_coverage: float = 0.0
    
    # Testing
    has_tests: bool = False
    test_files: int = 0
    test_ratio: float = 0.0  # test files / source files
    test_frameworks: list[str] = field(default_factory=list)
    
    # Documentation
    has_readme: bool = False
    readme_quality: str = ""  # minimal, basic, good, excellent
    has_docs_folder: bool = False
    has_changelog: bool = False
    has_contributing: bool = False
    
    # API/Routes
    api_endpoints: list[dict] = field(default_factory=list)
    
    # Key files
    key_files: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    
    # Imports analysis
    most_used_imports: list[tuple[str, int]] = field(default_factory=list)
    
    # File metrics (detailed)
    file_metrics: list[CodeMetrics] = field(default_factory=list)


# ============================================================================
# Framework Detection
# ============================================================================

FRAMEWORK_PATTERNS = {
    # Python Web
    "Django": [r"from django", r"import django", r"INSTALLED_APPS", r"urlpatterns"],
    "Flask": [r"from flask", r"import flask", r"Flask\(__name__\)"],
    "FastAPI": [r"from fastapi", r"import fastapi", r"FastAPI\(\)"],
    "Starlette": [r"from starlette", r"import starlette"],
    "Tornado": [r"from tornado", r"import tornado"],
    
    # Python Data/ML
    "NumPy": [r"import numpy", r"from numpy"],
    "Pandas": [r"import pandas", r"from pandas"],
    "PyTorch": [r"import torch", r"from torch"],
    "TensorFlow": [r"import tensorflow", r"from tensorflow"],
    "Scikit-learn": [r"from sklearn", r"import sklearn"],
    "Keras": [r"from keras", r"import keras"],
    
    # Python Testing
    "pytest": [r"import pytest", r"from pytest"],
    "unittest": [r"import unittest", r"from unittest"],
    
    # JavaScript/TypeScript
    "React": [r"from ['\"]react['\"]", r"import React", r"useState", r"useEffect"],
    "Vue": [r"from ['\"]vue['\"]", r"createApp", r"<template>"],
    "Angular": [r"@angular/core", r"@Component", r"@Injectable"],
    "Svelte": [r"<script.*>", r"\.svelte$"],
    "Next.js": [r"from ['\"]next", r"getServerSideProps", r"getStaticProps"],
    "Express": [r"from ['\"]express['\"]", r"require\(['\"]express['\"]"],
    "NestJS": [r"@nestjs/", r"@Controller", r"@Injectable"],
    
    # JavaScript Testing
    "Jest": [r"from ['\"]jest['\"]", r"describe\(", r"it\(", r"expect\("],
    "Mocha": [r"from ['\"]mocha['\"]", r"describe\(", r"it\("],
    "Vitest": [r"from ['\"]vitest['\"]"],
    
    # Go
    "Gin": [r"github.com/gin-gonic/gin"],
    "Echo": [r"github.com/labstack/echo"],
    "Fiber": [r"github.com/gofiber/fiber"],
    
    # Rust
    "Actix": [r"actix-web", r"actix_web"],
    "Rocket": [r"rocket::"],
    "Axum": [r"axum::"],
    
    # Ruby
    "Rails": [r"Rails\.", r"ActiveRecord", r"ActionController"],
    "Sinatra": [r"require ['\"]sinatra['\"]", r"Sinatra::"],
    
    # Database
    "SQLAlchemy": [r"from sqlalchemy", r"import sqlalchemy"],
    "Prisma": [r"@prisma/client", r"prisma\."],
    "TypeORM": [r"from ['\"]typeorm['\"]"],
    "Sequelize": [r"from ['\"]sequelize['\"]"],
    "Mongoose": [r"from ['\"]mongoose['\"]"],
}

PROJECT_TYPE_INDICATORS = {
    "web-app": ["src/pages", "src/views", "src/components", "templates/", "static/"],
    "api": ["routes/", "endpoints/", "api/", "controllers/", "handlers/"],
    "cli": ["cli.py", "main.py", "__main__.py", "bin/", "cmd/"],
    "library": ["src/lib/", "lib/", "setup.py", "pyproject.toml", "package.json"],
    "mobile": ["ios/", "android/", "App.tsx", "App.js"],
    "ml-project": ["notebooks/", "models/", "data/", "train.py", "model.py"],
    "monorepo": ["packages/", "apps/", "lerna.json", "pnpm-workspace.yaml"],
}

ARCHITECTURE_PATTERNS = {
    "mvc": ["models/", "views/", "controllers/"],
    "clean-architecture": ["domain/", "usecases/", "infrastructure/", "presentation/"],
    "hexagonal": ["adapters/", "ports/", "core/"],
    "layered": ["services/", "repositories/", "controllers/"],
    "microservices": ["docker-compose", "kubernetes", "k8s/"],
    "serverless": ["serverless.yml", "lambda/", "functions/"],
}


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_repository(repo_path: Path) -> RepoAnalysis:
    """
    Perform deep analysis of a repository.
    
    Args:
        repo_path: Path to the repository
    
    Returns:
        RepoAnalysis with comprehensive metrics
    """
    analysis = RepoAnalysis()
    
    # Collect all source files
    source_files = _collect_source_files(repo_path)
    
    # Analyze each file
    all_imports: Counter[str] = Counter()
    total_functions = 0
    total_complexity = 0
    files_with_docstrings = 0
    
    for file_path in source_files:
        metrics = _analyze_file(repo_path, file_path)
        if metrics:
            analysis.file_metrics.append(metrics)
            analysis.total_files += 1
            analysis.total_lines += metrics.lines_total
            analysis.code_lines += metrics.lines_code
            analysis.comment_lines += metrics.lines_comment
            
            for imp in metrics.imports:
                all_imports[imp] += 1
            
            total_functions += metrics.functions
            total_complexity += metrics.complexity
            
            if metrics.has_docstrings:
                files_with_docstrings += 1
            
            if metrics.has_tests:
                analysis.test_files += 1
    
    # Calculate averages
    if analysis.total_files > 0:
        analysis.avg_file_size = analysis.code_lines / analysis.total_files
        analysis.docstring_coverage = (files_with_docstrings / analysis.total_files) * 100
    
    if total_functions > 0:
        analysis.avg_complexity = total_complexity / total_functions
    
    # Language breakdown
    analysis.languages = _calculate_language_breakdown(analysis.file_metrics)
    
    # Detect frameworks
    analysis.frameworks = _detect_frameworks(repo_path, all_imports)
    analysis.notable_libraries = _detect_notable_libraries(all_imports)
    
    # Detect project type and architecture
    analysis.project_type = _detect_project_type(repo_path)
    analysis.architecture_patterns = _detect_architecture(repo_path)
    
    # Testing analysis
    analysis.has_tests = analysis.test_files > 0
    source_files_count = analysis.total_files - analysis.test_files
    if source_files_count > 0:
        analysis.test_ratio = analysis.test_files / source_files_count
    analysis.test_frameworks = _detect_test_frameworks(analysis.frameworks, all_imports)
    
    # Documentation analysis
    analysis.has_readme = _check_readme(repo_path)
    analysis.readme_quality = _assess_readme_quality(repo_path)
    analysis.has_docs_folder = (repo_path / "docs").is_dir()
    analysis.has_changelog = any(
        (repo_path / name).exists() 
        for name in ["CHANGELOG.md", "CHANGELOG", "HISTORY.md", "CHANGES.md"]
    )
    analysis.has_contributing = (repo_path / "CONTRIBUTING.md").exists()
    
    # API endpoints
    analysis.api_endpoints = _detect_api_endpoints(repo_path, analysis.frameworks)
    
    # Key files
    analysis.key_files = _identify_key_files(repo_path)
    analysis.entry_points = _identify_entry_points(repo_path)
    
    # Most used imports
    analysis.most_used_imports = all_imports.most_common(20)
    
    return analysis


def _collect_source_files(repo_path: Path) -> list[Path]:
    """Collect all source code files."""
    skip_dirs = {
        ".git", "node_modules", "venv", ".venv", "__pycache__",
        "dist", "build", ".next", "target", "coverage", ".tox",
        "eggs", "*.egg-info", ".mypy_cache", ".pytest_cache",
    }
    
    source_extensions = {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
        ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php", ".cs", ".scala", ".ex", ".exs",
    }
    
    files = []
    
    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Skip directories
        parts = file_path.relative_to(repo_path).parts
        if any(skip in parts for skip in skip_dirs):
            continue
        
        if file_path.suffix.lower() in source_extensions:
            files.append(file_path)
    
    return files


def _analyze_file(repo_path: Path, file_path: Path) -> CodeMetrics | None:
    """Analyze a single source file."""
    try:
        content = file_path.read_text(errors="ignore")
    except Exception:
        return None
    
    rel_path = str(file_path.relative_to(repo_path))
    language = _get_language(file_path)
    
    lines = content.split("\n")
    metrics = CodeMetrics(
        path=rel_path,
        language=language,
        lines_total=len(lines),
    )
    
    # Count line types
    in_multiline_comment = False
    in_multiline_string = False
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            metrics.lines_blank += 1
        elif _is_comment_line(stripped, language, in_multiline_comment):
            metrics.lines_comment += 1
        else:
            metrics.lines_code += 1
    
    # Language-specific analysis
    if language == "Python":
        metrics = _analyze_python_file(content, metrics)
    elif language in ("JavaScript", "TypeScript"):
        metrics = _analyze_js_file(content, metrics)
    elif language == "Go":
        metrics = _analyze_go_file(content, metrics)
    
    # Check if it's a test file
    metrics.has_tests = _is_test_file(rel_path, content)
    
    return metrics


def _analyze_python_file(content: str, metrics: CodeMetrics) -> CodeMetrics:
    """Deep analysis of Python file using AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return metrics
    
    function_lengths = []
    
    for node in ast.walk(tree):
        # Count functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics.functions += 1
            # Estimate function length
            if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                length = node.end_lineno - node.lineno + 1
                function_lengths.append(length)
            
            # Check for docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                metrics.has_docstrings = True
            
            # Estimate complexity (simplified)
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                      ast.With, ast.comprehension)):
                    metrics.complexity += 1
        
        # Count classes
        if isinstance(node, ast.ClassDef):
            metrics.classes += 1
            # Check for class docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
                metrics.has_docstrings = True
        
        # Extract imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                metrics.imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                metrics.imports.append(node.module.split(".")[0])
    
    if function_lengths:
        metrics.avg_function_length = sum(function_lengths) / len(function_lengths)
    
    return metrics


def _analyze_js_file(content: str, metrics: CodeMetrics) -> CodeMetrics:
    """Analyze JavaScript/TypeScript file using regex."""
    # Count functions (approximate)
    function_patterns = [
        r"function\s+\w+\s*\(",
        r"const\s+\w+\s*=\s*(?:async\s*)?\(",
        r"(?:async\s+)?function\s*\(",
        r"\w+\s*:\s*(?:async\s*)?\(",
        r"=>\s*{",
    ]
    for pattern in function_patterns:
        metrics.functions += len(re.findall(pattern, content))
    
    # Count classes
    metrics.classes = len(re.findall(r"class\s+\w+", content))
    
    # Extract imports
    import_patterns = [
        r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]",
        r"require\(['\"]([^'\"]+)['\"]\)",
    ]
    for pattern in import_patterns:
        for match in re.findall(pattern, content):
            # Clean up import path
            module = match.split("/")[0].replace("@", "")
            if module and not module.startswith("."):
                metrics.imports.append(module)
    
    # Estimate complexity (if/for/while/switch)
    complexity_patterns = [r"\bif\s*\(", r"\bfor\s*\(", r"\bwhile\s*\(", r"\bswitch\s*\("]
    for pattern in complexity_patterns:
        metrics.complexity += len(re.findall(pattern, content))
    
    return metrics


def _analyze_go_file(content: str, metrics: CodeMetrics) -> CodeMetrics:
    """Analyze Go file using regex."""
    # Count functions
    metrics.functions = len(re.findall(r"func\s+(?:\([^)]+\)\s*)?\w+\s*\(", content))
    
    # Count structs (Go's classes)
    metrics.classes = len(re.findall(r"type\s+\w+\s+struct", content))
    
    # Extract imports
    import_match = re.search(r"import\s*\((.*?)\)", content, re.DOTALL)
    if import_match:
        for line in import_match.group(1).split("\n"):
            match = re.search(r"\"([^\"]+)\"", line)
            if match:
                pkg = match.group(1).split("/")[-1]
                metrics.imports.append(pkg)
    
    # Single imports
    for match in re.findall(r"import\s+\"([^\"]+)\"", content):
        pkg = match.split("/")[-1]
        metrics.imports.append(pkg)
    
    # Complexity
    metrics.complexity = len(re.findall(r"\bif\s+", content))
    metrics.complexity += len(re.findall(r"\bfor\s+", content))
    metrics.complexity += len(re.findall(r"\bswitch\s+", content))
    
    return metrics


def _get_language(file_path: Path) -> str:
    """Get language from file extension."""
    ext_map = {
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
    }
    return ext_map.get(file_path.suffix.lower(), "Unknown")


def _is_comment_line(line: str, language: str, in_multiline: bool) -> bool:
    """Check if a line is a comment."""
    if language == "Python":
        return line.startswith("#") or line.startswith('"""') or line.startswith("'''")
    elif language in ("JavaScript", "TypeScript", "Java", "Go", "Rust", "C", "C++"):
        return line.startswith("//") or line.startswith("/*") or line.startswith("*")
    elif language == "Ruby":
        return line.startswith("#")
    return False


def _is_test_file(path: str, content: str) -> bool:
    """Check if file is a test file."""
    path_lower = path.lower()
    
    # Check path patterns
    test_patterns = ["test_", "_test.", ".test.", "spec.", "_spec.", "/tests/", "/test/", "/__tests__/"]
    if any(p in path_lower for p in test_patterns):
        return True
    
    # Check content patterns
    test_content_patterns = [
        r"def test_\w+",
        r"@pytest",
        r"unittest\.TestCase",
        r"describe\(['\"]",
        r"it\(['\"]",
        r"expect\(",
        r"assert\s+",
    ]
    for pattern in test_content_patterns:
        if re.search(pattern, content):
            return True
    
    return False


def _calculate_language_breakdown(metrics: list[CodeMetrics]) -> dict[str, float]:
    """Calculate language percentage breakdown."""
    lang_lines: Counter[str] = Counter()
    total_lines = 0
    
    for m in metrics:
        if m.language != "Unknown":
            lang_lines[m.language] += m.lines_code
            total_lines += m.lines_code
    
    if total_lines == 0:
        return {}
    
    return {
        lang: round((lines / total_lines) * 100, 1)
        for lang, lines in lang_lines.most_common()
        if (lines / total_lines) >= 0.01  # At least 1%
    }


def _detect_frameworks(repo_path: Path, all_imports: Counter) -> list[str]:
    """Detect frameworks used in the repository."""
    detected = []
    
    # Check file contents for framework patterns
    source_files = list(repo_path.rglob("*.py")) + list(repo_path.rglob("*.js")) + \
                   list(repo_path.rglob("*.ts")) + list(repo_path.rglob("*.tsx"))
    
    sample_content = ""
    for f in source_files[:50]:  # Sample first 50 files
        try:
            sample_content += f.read_text(errors="ignore")
        except Exception:
            pass
    
    # Also check package files
    for pf in ["package.json", "requirements.txt", "pyproject.toml", "Cargo.toml", "go.mod"]:
        pkg_file = repo_path / pf
        if pkg_file.exists():
            try:
                sample_content += pkg_file.read_text()
            except Exception:
                pass
    
    for framework, patterns in FRAMEWORK_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sample_content, re.IGNORECASE):
                if framework not in detected:
                    detected.append(framework)
                break
    
    return detected


def _detect_notable_libraries(all_imports: Counter) -> list[str]:
    """Detect notable libraries from imports."""
    notable = {
        # Python
        "requests": "HTTP client",
        "aiohttp": "Async HTTP",
        "celery": "Task queue",
        "redis": "Redis client",
        "pydantic": "Data validation",
        "sqlalchemy": "ORM",
        "alembic": "Migrations",
        "boto3": "AWS SDK",
        "opencv": "Computer vision",
        "pillow": "Image processing",
        "matplotlib": "Plotting",
        "seaborn": "Statistical viz",
        # JavaScript
        "axios": "HTTP client",
        "lodash": "Utilities",
        "moment": "Date handling",
        "dayjs": "Date handling",
        "rxjs": "Reactive programming",
        "socket.io": "WebSockets",
        "graphql": "GraphQL",
        "apollo": "GraphQL client",
        "redux": "State management",
        "zustand": "State management",
        "tanstack": "Data fetching",
        "tailwindcss": "CSS framework",
    }
    
    detected = []
    for imp, count in all_imports.items():
        imp_lower = imp.lower()
        for lib, desc in notable.items():
            if lib in imp_lower and lib not in detected:
                detected.append(lib)
    
    return detected[:15]  # Top 15


def _detect_project_type(repo_path: Path) -> str:
    """Detect the type of project."""
    files_and_dirs = set()
    
    for p in repo_path.iterdir():
        files_and_dirs.add(p.name)
    
    for p in repo_path.rglob("*"):
        try:
            rel = str(p.relative_to(repo_path))
            files_and_dirs.add(rel)
        except Exception:
            pass
    
    scores: Counter[str] = Counter()
    
    for project_type, indicators in PROJECT_TYPE_INDICATORS.items():
        for indicator in indicators:
            for item in files_and_dirs:
                if indicator.lower() in item.lower():
                    scores[project_type] += 1
    
    if scores:
        return scores.most_common(1)[0][0]
    return "unknown"


def _detect_architecture(repo_path: Path) -> list[str]:
    """Detect architecture patterns."""
    detected = []
    
    dirs = set()
    for p in repo_path.rglob("*"):
        if p.is_dir():
            dirs.add(p.name.lower() + "/")
    
    for pattern_name, indicators in ARCHITECTURE_PATTERNS.items():
        matches = sum(1 for ind in indicators if any(ind.lower() in d for d in dirs))
        if matches >= len(indicators) * 0.5:  # At least 50% match
            detected.append(pattern_name)
    
    return detected


def _detect_test_frameworks(frameworks: list[str], imports: Counter) -> list[str]:
    """Detect testing frameworks."""
    test_frameworks = ["pytest", "unittest", "Jest", "Mocha", "Vitest"]
    return [f for f in test_frameworks if f in frameworks]


def _check_readme(repo_path: Path) -> bool:
    """Check if repository has a README."""
    readme_names = ["README.md", "README.rst", "README.txt", "README"]
    return any((repo_path / name).exists() for name in readme_names)


def _assess_readme_quality(repo_path: Path) -> str:
    """Assess README quality."""
    readme_names = ["README.md", "README.rst", "README.txt", "README"]
    
    for name in readme_names:
        readme_path = repo_path / name
        if readme_path.exists():
            try:
                content = readme_path.read_text()
                lines = len(content.split("\n"))
                
                # Check for common sections
                has_install = bool(re.search(r"(?i)install|setup|getting started", content))
                has_usage = bool(re.search(r"(?i)usage|example|how to", content))
                has_api = bool(re.search(r"(?i)api|reference|documentation", content))
                has_license = bool(re.search(r"(?i)license|licence", content))
                has_badges = bool(re.search(r"\[!\[", content))  # Markdown badges
                
                score = sum([has_install, has_usage, has_api, has_license, has_badges])
                
                if lines < 20:
                    return "minimal"
                elif score >= 4 and lines > 100:
                    return "excellent"
                elif score >= 2 and lines > 50:
                    return "good"
                else:
                    return "basic"
            except Exception:
                pass
    
    return "none"


def _detect_api_endpoints(repo_path: Path, frameworks: list[str]) -> list[dict]:
    """Detect API endpoints/routes."""
    endpoints = []
    
    # FastAPI patterns
    fastapi_patterns = [
        (r"@(?:app|router)\.(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]", "FastAPI"),
    ]
    
    # Flask patterns
    flask_patterns = [
        (r"@(?:app|blueprint)\.(route|get|post|put|delete)\(['\"]([^'\"]+)['\"]", "Flask"),
    ]
    
    # Express patterns
    express_patterns = [
        (r"(?:app|router)\.(get|post|put|patch|delete)\(['\"]([^'\"]+)['\"]", "Express"),
    ]
    
    patterns = []
    if "FastAPI" in frameworks:
        patterns.extend(fastapi_patterns)
    if "Flask" in frameworks:
        patterns.extend(flask_patterns)
    if "Express" in frameworks or "NestJS" in frameworks:
        patterns.extend(express_patterns)
    
    # If no framework detected, try all patterns
    if not patterns:
        patterns = fastapi_patterns + flask_patterns + express_patterns
    
    for file_path in list(repo_path.rglob("*.py")) + list(repo_path.rglob("*.js")) + list(repo_path.rglob("*.ts")):
        try:
            content = file_path.read_text(errors="ignore")
            for pattern, framework in patterns:
                for match in re.findall(pattern, content, re.IGNORECASE):
                    if isinstance(match, tuple):
                        method, path = match
                    else:
                        method, path = "GET", match
                    
                    endpoints.append({
                        "method": method.upper(),
                        "path": path,
                        "file": str(file_path.relative_to(repo_path)),
                    })
        except Exception:
            pass
    
    # Deduplicate and limit
    seen = set()
    unique_endpoints = []
    for ep in endpoints:
        key = (ep["method"], ep["path"])
        if key not in seen:
            seen.add(key)
            unique_endpoints.append(ep)
    
    return unique_endpoints[:50]  # Limit to 50


def _identify_key_files(repo_path: Path) -> list[str]:
    """Identify key files in the repository."""
    key_patterns = [
        "main.py", "app.py", "index.py", "server.py", "cli.py",
        "main.js", "index.js", "app.js", "server.js",
        "main.ts", "index.ts", "app.ts", "server.ts",
        "main.go", "main.rs",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "Makefile", "justfile",
        ".env.example", ".env.sample",
        "setup.py", "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
    ]
    
    found = []
    for pattern in key_patterns:
        for file_path in repo_path.rglob(pattern):
            rel_path = str(file_path.relative_to(repo_path))
            if rel_path not in found:
                found.append(rel_path)
    
    return found[:20]


def _identify_entry_points(repo_path: Path) -> list[str]:
    """Identify likely entry points."""
    entry_patterns = [
        "main.py", "__main__.py", "app.py", "cli.py", "server.py", "run.py",
        "index.js", "main.js", "app.js", "server.js",
        "index.ts", "main.ts", "app.ts", "server.ts",
        "main.go", "cmd/main.go",
        "main.rs", "src/main.rs",
    ]
    
    found = []
    for pattern in entry_patterns:
        for file_path in repo_path.rglob(pattern):
            rel_path = str(file_path.relative_to(repo_path))
            # Prefer root-level entry points
            if "/" not in rel_path or rel_path.startswith("src/") or rel_path.startswith("cmd/"):
                found.append(rel_path)
    
    return found[:5]


# ============================================================================
# Public API
# ============================================================================

def analyze_repo(repo_path: Path) -> dict[str, Any]:
    """
    Analyze a repository and return results as a dictionary.
    
    This is the main entry point for the analysis tool.
    """
    analysis = analyze_repository(repo_path)
    
    return {
        "summary": {
            "total_files": analysis.total_files,
            "total_lines": analysis.total_lines,
            "code_lines": analysis.code_lines,
            "comment_lines": analysis.comment_lines,
            "project_type": analysis.project_type,
        },
        "languages": analysis.languages,
        "frameworks": analysis.frameworks,
        "notable_libraries": analysis.notable_libraries,
        "architecture": analysis.architecture_patterns,
        "quality": {
            "avg_file_size_lines": round(analysis.avg_file_size, 1),
            "avg_complexity": round(analysis.avg_complexity, 2),
            "docstring_coverage_pct": round(analysis.docstring_coverage, 1),
        },
        "testing": {
            "has_tests": analysis.has_tests,
            "test_files": analysis.test_files,
            "test_ratio": round(analysis.test_ratio, 2),
            "test_frameworks": analysis.test_frameworks,
        },
        "documentation": {
            "has_readme": analysis.has_readme,
            "readme_quality": analysis.readme_quality,
            "has_docs_folder": analysis.has_docs_folder,
            "has_changelog": analysis.has_changelog,
            "has_contributing": analysis.has_contributing,
        },
        "api_endpoints": analysis.api_endpoints[:20],  # Limit for readability
        "key_files": analysis.key_files,
        "entry_points": analysis.entry_points,
        "most_used_imports": [
            {"name": name, "count": count}
            for name, count in analysis.most_used_imports[:15]
        ],
    }


def get_code_snippets(
    repo_path: Path,
    max_snippets: int = 5,
    max_lines: int = 50,
) -> list[dict[str, Any]]:
    """
    Extract interesting code snippets from a repository.
    
    Looks for:
    - Main entry points
    - Key functions/classes
    - Interesting patterns
    """
    snippets = []
    
    # Find interesting files
    interesting_files = [
        "main.py", "app.py", "cli.py", "server.py",
        "models.py", "schema.py", "routes.py", "handlers.py",
        "index.ts", "App.tsx", "main.ts",
    ]
    
    for pattern in interesting_files:
        for file_path in repo_path.rglob(pattern):
            if len(snippets) >= max_snippets:
                break
            
            try:
                content = file_path.read_text(errors="ignore")
                lines = content.split("\n")
                
                # Take first N lines or find an interesting section
                snippet_lines = lines[:max_lines]
                
                snippets.append({
                    "path": str(file_path.relative_to(repo_path)),
                    "content": "\n".join(snippet_lines),
                    "total_lines": len(lines),
                    "truncated": len(lines) > max_lines,
                })
            except Exception:
                pass
    
    return snippets

