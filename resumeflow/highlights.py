"""
Extract resume-worthy highlights from repositories.

Goes beyond stats to identify:
- What was built (features, systems)
- Technical challenges solved
- Integrations implemented
- Impact and scale indicators
- Domain expertise demonstrated
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from git import Repo


# ============================================================================
# Feature & Integration Detection
# ============================================================================

# What they BUILT (features/systems)
FEATURES = {
    "authentication": {
        "patterns": [
            r"auth", r"login", r"logout", r"signup", r"register",
            r"oauth", r"jwt", r"session", r"password", r"2fa", r"mfa",
            r"sso", r"saml", r"ldap", r"token",
        ],
        "description": "Authentication system",
        "skills": ["Security", "OAuth", "JWT"],
    },
    "payment": {
        "patterns": [
            r"payment", r"checkout", r"billing", r"invoice",
            r"subscription", r"stripe", r"paypal", r"charge",
            r"refund", r"transaction", r"pricing",
        ],
        "description": "Payment processing",
        "skills": ["Payment APIs", "PCI compliance"],
    },
    "real_time": {
        "patterns": [
            r"websocket", r"socket\.io", r"realtime", r"real-time",
            r"live\s*update", r"push\s*notification", r"sse",
            r"pubsub", r"broadcast", r"channel",
        ],
        "description": "Real-time communication",
        "skills": ["WebSockets", "Event-driven architecture"],
    },
    "search": {
        "patterns": [
            r"elasticsearch", r"solr", r"algolia", r"meilisearch",
            r"full.?text.?search", r"search.?index", r"fuzzy.?search",
        ],
        "description": "Search functionality",
        "skills": ["Search engines", "Full-text search"],
    },
    "file_upload": {
        "patterns": [
            r"upload", r"s3", r"blob", r"storage", r"presigned",
            r"multipart", r"file.?handling", r"asset",
        ],
        "description": "File upload system",
        "skills": ["Cloud storage", "File handling"],
    },
    "email": {
        "patterns": [
            r"sendgrid", r"mailgun", r"ses", r"smtp",
            r"email.?template", r"newsletter", r"transactional",
        ],
        "description": "Email system",
        "skills": ["Email APIs", "Template systems"],
    },
    "notifications": {
        "patterns": [
            r"notification", r"push", r"firebase.?cloud", r"fcm",
            r"apns", r"alert", r"in.?app.?message",
        ],
        "description": "Notification system",
        "skills": ["Push notifications", "Real-time updates"],
    },
    "caching": {
        "patterns": [
            r"redis", r"memcache", r"cache", r"memoiz",
            r"ttl", r"invalidat", r"cdn",
        ],
        "description": "Caching layer",
        "skills": ["Redis", "Performance optimization"],
    },
    "queue": {
        "patterns": [
            r"celery", r"rq", r"bullmq", r"rabbitmq", r"sqs",
            r"background.?job", r"worker", r"task.?queue", r"async.?task",
        ],
        "description": "Background job system",
        "skills": ["Message queues", "Async processing"],
    },
    "api": {
        "patterns": [
            r"rest.?api", r"graphql", r"openapi", r"swagger",
            r"endpoint", r"rate.?limit", r"api.?version",
        ],
        "description": "API development",
        "skills": ["REST API", "API design"],
    },
    "database": {
        "patterns": [
            r"migration", r"schema", r"model", r"orm",
            r"query.?optim", r"index", r"foreign.?key",
            r"transaction", r"rollback",
        ],
        "description": "Database design",
        "skills": ["Database design", "SQL", "ORM"],
    },
    "ci_cd": {
        "patterns": [
            r"github.?action", r"jenkins", r"circleci", r"travis",
            r"deploy", r"pipeline", r"ci.?cd", r"docker",
            r"kubernetes", r"helm", r"terraform",
        ],
        "description": "CI/CD pipeline",
        "skills": ["DevOps", "CI/CD", "Docker"],
    },
    "testing": {
        "patterns": [
            r"test.?suite", r"unit.?test", r"integration.?test",
            r"e2e", r"coverage", r"mock", r"fixture",
        ],
        "description": "Test suite",
        "skills": ["Testing", "TDD"],
    },
    "analytics": {
        "patterns": [
            r"analytics", r"tracking", r"metrics", r"dashboard",
            r"report", r"insight", r"visualization",
        ],
        "description": "Analytics system",
        "skills": ["Data analysis", "Visualization"],
    },
    "ml_ai": {
        "patterns": [
            r"machine.?learning", r"neural", r"train", r"model",
            r"predict", r"tensorflow", r"pytorch", r"sklearn",
            r"llm", r"gpt", r"embedding", r"vector",
        ],
        "description": "ML/AI features",
        "skills": ["Machine Learning", "AI"],
    },
    "scraping": {
        "patterns": [
            r"scrape", r"crawl", r"spider", r"beautifulsoup",
            r"selenium", r"playwright", r"puppeteer",
        ],
        "description": "Web scraping",
        "skills": ["Web scraping", "Data extraction"],
    },
    "security": {
        "patterns": [
            r"encrypt", r"decrypt", r"hash", r"salt",
            r"csrf", r"xss", r"sanitiz", r"vulnerability",
            r"permission", r"role.?based", r"rbac", r"acl",
        ],
        "description": "Security implementation",
        "skills": ["Security", "Encryption"],
    },
}

# Third-party integrations
INTEGRATIONS = {
    # Payment
    "Stripe": [r"stripe", r"sk_live", r"pk_live"],
    "PayPal": [r"paypal", r"braintree"],
    "Square": [r"square", r"squareup"],
    
    # Cloud
    "AWS": [r"aws", r"boto3", r"s3", r"ec2", r"lambda", r"dynamodb", r"sqs", r"sns"],
    "Google Cloud": [r"gcp", r"google.cloud", r"bigquery", r"firestore"],
    "Azure": [r"azure", r"microsoft.azure"],
    "Vercel": [r"vercel", r"@vercel"],
    "Cloudflare": [r"cloudflare", r"cf-"],
    
    # Database
    "PostgreSQL": [r"postgres", r"psycopg", r"pg_"],
    "MongoDB": [r"mongodb", r"mongoose", r"pymongo"],
    "Redis": [r"redis", r"ioredis"],
    "Elasticsearch": [r"elasticsearch", r"elastic.co"],
    
    # Communication
    "Twilio": [r"twilio"],
    "SendGrid": [r"sendgrid"],
    "Mailgun": [r"mailgun"],
    "Slack": [r"slack", r"slack-sdk"],
    "Discord": [r"discord", r"discord.py"],
    
    # Auth
    "Auth0": [r"auth0"],
    "Firebase Auth": [r"firebase.auth", r"firebase/auth"],
    "Okta": [r"okta"],
    "Clerk": [r"@clerk"],
    "Supabase": [r"supabase"],
    
    # AI/ML
    "OpenAI": [r"openai", r"gpt-4", r"gpt-3"],
    "Anthropic": [r"anthropic", r"claude"],
    "Hugging Face": [r"huggingface", r"transformers"],
    "LangChain": [r"langchain"],
    
    # Analytics
    "Segment": [r"segment", r"analytics.js"],
    "Mixpanel": [r"mixpanel"],
    "Amplitude": [r"amplitude"],
    "PostHog": [r"posthog"],
    
    # Monitoring
    "Sentry": [r"sentry", r"@sentry"],
    "Datadog": [r"datadog", r"dd-trace"],
    "New Relic": [r"newrelic"],
    
    # Other
    "GitHub API": [r"github.api", r"octokit", r"pygithub"],
    "Shopify": [r"shopify"],
    "Contentful": [r"contentful"],
    "Algolia": [r"algolia", r"algoliasearch"],
}

# Scale/impact indicators
SCALE_INDICATORS = {
    "high_traffic": [
        r"rate.?limit", r"throttl", r"load.?balanc", r"horizontal.?scal",
        r"replica", r"shard", r"partition", r"concurrent",
    ],
    "data_intensive": [
        r"batch.?process", r"etl", r"pipeline", r"stream",
        r"million", r"billion", r"terabyte", r"petabyte",
    ],
    "performance": [
        r"optimi[zs]", r"latency", r"throughput", r"benchmark",
        r"profil", r"memory.?leak", r"garbage.?collect",
    ],
    "reliability": [
        r"fault.?toleran", r"failover", r"redundan", r"backup",
        r"disaster.?recovery", r"high.?availability", r"sla",
    ],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Highlight:
    """A resume-worthy highlight."""
    category: str  # feature, integration, achievement
    title: str
    description: str
    evidence: list[str] = field(default_factory=list)  # file paths, commit messages
    skills: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1 how confident we are


@dataclass
class RepoHighlights:
    """Highlights extracted from a repository."""
    repo_name: str
    repo_path: str
    
    # What was built
    features: list[Highlight] = field(default_factory=list)
    integrations: list[Highlight] = field(default_factory=list)
    
    # Impact
    scale_indicators: list[str] = field(default_factory=list)
    
    # From commits
    key_achievements: list[str] = field(default_factory=list)
    
    # Domain
    domain: str = ""  # e-commerce, fintech, healthcare, etc.
    
    # Summary for LLM
    summary_context: str = ""


# ============================================================================
# Extraction Functions
# ============================================================================

def extract_highlights(repo_path: Path) -> RepoHighlights:
    """
    Extract resume-worthy highlights from a repository.
    
    Analyzes:
    - Code patterns to identify features built
    - Imports and configs for integrations
    - Commit messages for achievements
    - README for project purpose
    """
    highlights = RepoHighlights(
        repo_name=repo_path.name,
        repo_path=str(repo_path),
    )
    
    # Collect all source content for analysis
    all_content = _collect_source_content(repo_path)
    
    # Detect features
    highlights.features = _detect_features(all_content, repo_path)
    
    # Detect integrations
    highlights.integrations = _detect_integrations(all_content, repo_path)
    
    # Detect scale indicators
    highlights.scale_indicators = _detect_scale(all_content)
    
    # Analyze commits for achievements
    highlights.key_achievements = _extract_achievements_from_commits(repo_path)
    
    # Detect domain
    highlights.domain = _detect_domain(all_content, repo_path)
    
    # Build summary context for LLM
    highlights.summary_context = _build_summary_context(highlights, repo_path)
    
    return highlights


def _collect_source_content(repo_path: Path) -> str:
    """Collect content from source files for pattern matching."""
    content_parts = []
    
    skip_dirs = {
        ".git", "node_modules", "venv", ".venv", "__pycache__",
        "dist", "build", ".next", "target", "coverage",
    }
    
    extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"}
    
    # Also check config files
    config_files = [
        "package.json", "requirements.txt", "pyproject.toml",
        "Cargo.toml", "go.mod", "Gemfile", "docker-compose.yml",
        "Dockerfile", ".env.example", "README.md",
    ]
    
    for config in config_files:
        config_path = repo_path / config
        if config_path.exists():
            try:
                content_parts.append(config_path.read_text(errors="ignore"))
            except Exception:
                pass
    
    # Sample source files (limit to avoid memory issues)
    file_count = 0
    max_files = 100
    
    for file_path in repo_path.rglob("*"):
        if file_count >= max_files:
            break
        
        if not file_path.is_file():
            continue
        
        parts = file_path.relative_to(repo_path).parts
        if any(skip in parts for skip in skip_dirs):
            continue
        
        if file_path.suffix.lower() in extensions:
            try:
                content = file_path.read_text(errors="ignore")
                # Limit per file
                content_parts.append(content[:50000])
                file_count += 1
            except Exception:
                pass
    
    return "\n".join(content_parts)


def _detect_features(content: str, repo_path: Path) -> list[Highlight]:
    """Detect features built based on code patterns."""
    detected = []
    content_lower = content.lower()
    
    for feature_id, feature_info in FEATURES.items():
        matches = 0
        evidence = []
        
        for pattern in feature_info["patterns"]:
            found = re.findall(pattern, content_lower)
            if found:
                matches += len(found)
                # Find file evidence
                for f in _find_files_with_pattern(repo_path, pattern):
                    if f not in evidence:
                        evidence.append(f)
        
        if matches >= 3:  # Minimum threshold
            confidence = min(1.0, matches / 20)  # More matches = higher confidence
            
            detected.append(Highlight(
                category="feature",
                title=feature_info["description"],
                description=f"Implemented {feature_info['description'].lower()}",
                evidence=evidence[:5],  # Top 5 files
                skills=feature_info["skills"],
                confidence=confidence,
            ))
    
    # Sort by confidence
    detected.sort(key=lambda x: x.confidence, reverse=True)
    
    return detected[:10]  # Top 10 features


def _detect_integrations(content: str, repo_path: Path) -> list[Highlight]:
    """Detect third-party integrations."""
    detected = []
    content_lower = content.lower()
    
    for integration, patterns in INTEGRATIONS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                evidence = _find_files_with_pattern(repo_path, pattern)
                
                detected.append(Highlight(
                    category="integration",
                    title=integration,
                    description=f"Integrated {integration}",
                    evidence=evidence[:3],
                    skills=[integration],
                    confidence=0.8,
                ))
                break  # Only add once per integration
    
    return detected


def _detect_scale(content: str) -> list[str]:
    """Detect scale/impact indicators."""
    indicators = []
    content_lower = content.lower()
    
    for indicator_type, patterns in SCALE_INDICATORS.items():
        for pattern in patterns:
            if re.search(pattern, content_lower):
                indicators.append(indicator_type)
                break
    
    return indicators


def _find_files_with_pattern(repo_path: Path, pattern: str) -> list[str]:
    """Find files containing a pattern."""
    files = []
    skip_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return files
    
    for file_path in repo_path.rglob("*"):
        if len(files) >= 10:
            break
        
        if not file_path.is_file():
            continue
        
        parts = file_path.relative_to(repo_path).parts
        if any(skip in parts for skip in skip_dirs):
            continue
        
        try:
            content = file_path.read_text(errors="ignore")
            if regex.search(content):
                files.append(str(file_path.relative_to(repo_path)))
        except Exception:
            pass
    
    return files


def _extract_achievements_from_commits(repo_path: Path) -> list[str]:
    """Extract key achievements from commit messages."""
    achievements = []
    
    # Keywords that indicate achievements
    achievement_patterns = [
        r"implement(?:ed|s)?\s+(.+)",
        r"add(?:ed|s)?\s+(.+)",
        r"create(?:d|s)?\s+(.+)",
        r"build(?:s)?\s+(.+)",
        r"introduc(?:e|ed|es)\s+(.+)",
        r"develop(?:ed|s)?\s+(.+)",
        r"design(?:ed|s)?\s+(.+)",
        r"integrat(?:e|ed|es)\s+(.+)",
        r"migrat(?:e|ed|es)\s+(.+)",
        r"optimi[zs](?:e|ed|es)\s+(.+)",
        r"refactor(?:ed|s)?\s+(.+)",
        r"fix(?:ed|es)?\s+(.+)",
        r"improv(?:e|ed|es)\s+(.+)",
    ]
    
    try:
        repo = Repo(repo_path)
        
        # Get commit messages
        commit_messages = []
        for commit in repo.iter_commits(max_count=200):
            msg = commit.message.strip().split("\n")[0]  # First line
            if len(msg) > 10:  # Skip short messages
                commit_messages.append(msg)
        
        # Extract achievements
        seen = set()
        for msg in commit_messages:
            msg_lower = msg.lower()
            
            # Skip common non-achievement messages
            if any(skip in msg_lower for skip in [
                "merge", "update readme", "fix typo", "wip", "temp",
                "clean up", "formatting", "lint", "bump version",
            ]):
                continue
            
            for pattern in achievement_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    achievement = match.group(1).strip()
                    # Clean up
                    achievement = re.sub(r"\s+", " ", achievement)
                    achievement = achievement[:100]  # Limit length
                    
                    if achievement and achievement not in seen and len(achievement) > 5:
                        seen.add(achievement)
                        achievements.append(msg)  # Use original message
                        break
        
    except Exception:
        pass
    
    # Return most meaningful achievements
    # Prioritize longer, more descriptive commits
    achievements.sort(key=lambda x: len(x), reverse=True)
    
    return achievements[:15]  # Top 15


def _detect_domain(content: str, repo_path: Path) -> str:
    """Detect the domain/industry of the project."""
    content_lower = content.lower()
    
    domains = {
        "e-commerce": [r"shop", r"cart", r"checkout", r"product", r"inventory", r"order"],
        "fintech": [r"payment", r"banking", r"transaction", r"wallet", r"financial"],
        "healthcare": [r"patient", r"medical", r"health", r"doctor", r"appointment"],
        "education": [r"course", r"student", r"lesson", r"quiz", r"learning"],
        "social": [r"post", r"comment", r"follow", r"friend", r"feed", r"like"],
        "productivity": [r"task", r"project", r"calendar", r"reminder", r"schedule"],
        "media": [r"video", r"stream", r"playlist", r"podcast", r"audio"],
        "developer-tools": [r"cli", r"sdk", r"api", r"framework", r"library"],
        "real-estate": [r"property", r"listing", r"agent", r"rent", r"tenant"],
        "travel": [r"booking", r"hotel", r"flight", r"destination", r"itinerary"],
        "food": [r"restaurant", r"menu", r"delivery", r"recipe", r"order"],
        "fitness": [r"workout", r"exercise", r"gym", r"fitness", r"health"],
        "analytics": [r"dashboard", r"metric", r"report", r"insight", r"chart"],
    }
    
    scores = Counter()
    for domain, patterns in domains.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, content_lower))
            scores[domain] += matches
    
    if scores:
        top_domain = scores.most_common(1)[0]
        if top_domain[1] >= 5:  # Minimum threshold
            return top_domain[0]
    
    return ""


def _build_summary_context(highlights: RepoHighlights, repo_path: Path) -> str:
    """Build a summary context for LLM to generate narrative."""
    parts = []
    
    # README content
    readme_content = ""
    for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
        readme_path = repo_path / readme_name
        if readme_path.exists():
            try:
                readme_content = readme_path.read_text()[:2000]
                break
            except Exception:
                pass
    
    if readme_content:
        parts.append(f"README:\n{readme_content}\n")
    
    # Features
    if highlights.features:
        feature_list = ", ".join(f.title for f in highlights.features[:5])
        parts.append(f"Features built: {feature_list}")
    
    # Integrations
    if highlights.integrations:
        integration_list = ", ".join(i.title for i in highlights.integrations[:5])
        parts.append(f"Integrations: {integration_list}")
    
    # Domain
    if highlights.domain:
        parts.append(f"Domain: {highlights.domain}")
    
    # Scale
    if highlights.scale_indicators:
        parts.append(f"Scale indicators: {', '.join(highlights.scale_indicators)}")
    
    # Key commits
    if highlights.key_achievements:
        parts.append(f"Key achievements from commits:\n" + "\n".join(f"- {a}" for a in highlights.key_achievements[:10]))
    
    return "\n\n".join(parts)


# ============================================================================
# Public API
# ============================================================================

def get_highlights(repo_path: Path) -> dict[str, Any]:
    """
    Get highlights as a dictionary for easy consumption.
    
    Returns structured data about:
    - Features built
    - Integrations used
    - Technical achievements
    - Skills demonstrated
    """
    highlights = extract_highlights(repo_path)
    
    # Collect all skills
    all_skills = set()
    for f in highlights.features:
        all_skills.update(f.skills)
    for i in highlights.integrations:
        all_skills.update(i.skills)
    
    return {
        "features": [
            {
                "name": f.title,
                "description": f.description,
                "confidence": round(f.confidence, 2),
                "skills": f.skills,
                "evidence_files": f.evidence[:3],
            }
            for f in highlights.features
        ],
        "integrations": [
            {
                "name": i.title,
                "description": i.description,
            }
            for i in highlights.integrations
        ],
        "achievements": highlights.key_achievements[:10],
        "domain": highlights.domain,
        "scale_indicators": highlights.scale_indicators,
        "skills_demonstrated": sorted(all_skills),
        "summary_context": highlights.summary_context,
    }


def format_highlights_for_resume(highlights: dict[str, Any]) -> str:
    """
    Format highlights as resume bullet points.
    """
    bullets = []
    
    # Features as accomplishments
    for feature in highlights.get("features", [])[:5]:
        if feature["confidence"] >= 0.5:
            skills = ", ".join(feature["skills"][:3]) if feature["skills"] else ""
            bullet = f"• {feature['description']}"
            if skills:
                bullet += f" ({skills})"
            bullets.append(bullet)
    
    # Integrations
    integrations = highlights.get("integrations", [])
    if integrations:
        integration_names = [i["name"] for i in integrations[:5]]
        bullets.append(f"• Integrated with {', '.join(integration_names)}")
    
    # Scale indicators
    scale = highlights.get("scale_indicators", [])
    if "high_traffic" in scale:
        bullets.append("• Built for high-traffic production environment")
    if "performance" in scale:
        bullets.append("• Implemented performance optimizations")
    if "reliability" in scale:
        bullets.append("• Designed for high availability and fault tolerance")
    
    return "\n".join(bullets)

