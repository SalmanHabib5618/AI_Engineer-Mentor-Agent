"""AI Engineer Mentor: beginner-first, intent-aware mentor workflow.

Compatible with a project model_provider.py exposing ModelProvider,
available_provider(mode), ProviderError, and provider.generate(system_prompt, prompt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from model_provider import ModelProvider, available_provider, ProviderError


@dataclass
class Route:
    """Lightweight routing metadata for the app UI."""
    conversation_mode: str
    intent: str
    requires_model: bool
    note: str = ""


MAX_MESSAGE_LENGTH = 12_000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_LENGTH = 900

GREETING_PATTERNS = (
    r"^(hi|hello|hey|hey there|hi mentor|hello mentor)[!. ]*$",
    r"^(good morning|good afternoon|good evening|good night)[!. ]*$",
    r"^(thanks|thank you|thx|ok|okay|got it|bye|goodbye)[!. ]*$",
)

HAPPY_TERMS = (
    "i'm happy", "i am happy", "i did it", "i finally", "i passed",
    "i succeeded", "i finished", "i completed", "great news", "i got the job",
    "i solved it", "i understand now", "i got it working", "i'm excited",
    "i am excited",
)

SUPPORT_TERMS = (
    "i'm sad", "i am sad", "i feel sad", "i'm stressed", "i am stressed",
    "i'm frustrated", "i am frustrated", "i feel stupid", "i can't do this",
    "i cannot do this", "i give up", "i feel hopeless", "i'm overwhelmed",
    "i am overwhelmed", "i'm discouraged", "i am discouraged",
    "i don't understand anything",
)

INTENT_KEYWORDS = {
    "code": (
        "code", "python", "function", "script", "program", "syntax", "debug",
        "error", "exception", "bug", "traceback", "csv",
    ),
    "project": (
        "project", "build", "implement", "application", "app", "portfolio",
        "create a tool", "make an agent",
    ),
    "career": (
        "job", "career", "resume", "cv", "interview", "hiring", "job-ready",
        "job ready", "salary",
    ),
    "research": (
        "latest", "current", "today", "recent", "new version", "release",
        "compare sources", "official docs", "news",
    ),
    "quiz": (
        "quiz", "test me", "practice questions", "exercise",
        "test my knowledge", "give me questions",
    ),
    "roadmap": (
        "roadmap", "what should i learn", "next lesson", "study plan",
        "learn today", "my progress", "learning plan",
    ),
    "concept": (
        "what is", "explain", "teach me", "how does", "difference between",
        "meaning of", "why does",
    ),
}

SYSTEM_PROMPT = """
You are AI Engineer Mentor, a patient and practical guide helping a complete
beginner start a journey toward becoming an AI Engineer.

LEARNER PROFILE
- Treat the learner as starting from scratch.
- Do not assume they know Python, programming, data science, machine learning,
  APIs, command lines, Git, or AI terminology.
- Do not assume what they have already learned, built, completed, or understood.
- Use simple, respectful English. Briefly explain necessary technical words.
- Be encouraging without being childish, exaggerated, or overly motivational.

MOST IMPORTANT: UNDERSTAND THE CURRENT REQUEST
- First determine what the user is asking for in the current message.
- The current message is the primary source of intent.
- Conversation history is context, not a reason to ignore a new request.
- Any task category supplied by the application is only a weak hint and may be
  wrong. Never let it override the user's actual message.
- Do not turn every question into a lesson, roadmap, Python task, or AI topic.
- Answer the actual question and follow the requested format.
- If the user asks for a direct answer, give the answer first.
- If the user asks for code, provide relevant code and explain how to run it when useful.
- If the user asks to fix code, identify the likely cause and make a targeted correction.
- If the user asks to modify a file, preserve its architecture and interfaces unless
  a change is necessary.
- If the user asks for a project, give a realistic plan or implementation.
- If the user asks for a learning plan, make it practical and beginner-friendly.
- For latest/current information, do not pretend to have live research unless a
  research tool or reliable current source was actually used.
- If the user shares an achievement, respond naturally.
- If the user is discouraged, acknowledge it and offer one manageable next step.
- If the user only greets you or says thanks, respond briefly and naturally.

TEACHING STYLE
- Explain one main concept at a time.
- Prefer a small example before a large or abstract explanation.
- When teaching a process, use numbered steps.
- When presenting code, explain what it does and how to verify it.
- Avoid dumping alternatives unless useful or requested.
- Ask at most one concise clarification question when a missing detail prevents
  a useful answer. Otherwise make a reasonable assumption and state it.
- Do not claim code was run, deployment succeeded, files changed, or sources were
  checked unless that actually happened.
- Be honest about uncertainty and limitations.
- Never reveal API keys, secrets, or private configuration values.
"""


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    """Match phrases with simple word boundaries."""
    padded = f" {text.lower()} "
    return any(f" {term.lower()} " in padded for term in terms)


def _technical_intent(text: str) -> Optional[str]:
    """Guess a category for UI metadata; this is not true semantic classification."""
    for intent, keywords in INTENT_KEYWORDS.items():
        if _contains_any(text, keywords):
            return intent
    return None


def route_message(query: str) -> Route:
    """Route short social messages locally; send other requests to the model."""
    if not isinstance(query, str) or not query.strip():
        return Route("learning", "unknown", False, "empty")

    clean = query.strip()
    if len(clean) > MAX_MESSAGE_LENGTH:
        return Route("learning", "unknown", False, "too_long")

    lower = clean.lower()
    if any(re.fullmatch(p, lower, flags=re.IGNORECASE) for p in GREETING_PATTERNS):
        return Route("social", "conversation", False)

    happy = _contains_any(lower, HAPPY_TERMS)
    supportive = _contains_any(lower, SUPPORT_TERMS)
    guessed_intent = _technical_intent(lower)

    # If emotion and a task appear together, preserve the full request for the model.
    if happy and not guessed_intent:
        return Route("happy", "conversation", False)
    if supportive and not guessed_intent:
        return Route("supportive", "conversation", False)

    return Route("learning", guessed_intent or "general_request", True)


def social_response(query: str, route: Route) -> str:
    """Short responses for simple messages that do not need a model."""
    lower = query.lower().strip()

    if route.conversation_mode == "social":
        if "thank" in lower or "thx" in lower:
            return (
                "You're welcome! When you're ready, tell me what you'd like help "
                "with, and we'll take it one step at a time."
            )
        if any(word in lower for word in ("bye", "goodbye", "good night")):
            return "Take care! Come back whenever you're ready to continue."
        return (
            "Hello! I'm your AI Engineer Mentor. We'll start from the basics and "
            "work forward one step at a time. What would you like help with today?"
        )

    if route.conversation_mode == "happy":
        return (
            "That's great news! You put effort into this, and it's worth taking a "
            "moment to appreciate your progress. What helped you get there?"
        )

    if route.conversation_mode == "supportive":
        return (
            "I'm sorry this feels difficult right now. You don't need to understand "
            "everything at once. Tell me which part is confusing, and we'll work "
            "through just that part together."
        )

    return "I'm here to help. What would you like to work on?"


def _format_history(
    recent_history: Optional[Sequence[Mapping[str, Any]]],
) -> str:
    """Format a bounded amount of recent conversation as context."""
    if not recent_history:
        return ""

    lines = []
    for message in list(recent_history)[-MAX_HISTORY_MESSAGES:]:
        role = str(message.get("role", "user")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(
                f"{role}: {content[:MAX_HISTORY_MESSAGE_LENGTH]}"
            )

    if not lines:
        return ""

    return (
        "\n\nRecent conversation context (may be incomplete; use only if relevant):\n"
        + "\n".join(lines)
    )


def _task_hint(intent: str) -> str:
    """Return a non-binding hint for the model."""
    hints = {
        "code": "Possibly a programming or code-related request.",
        "project": "Possibly a project-building request.",
        "career": "Possibly a career-related request.",
        "research": "Possibly a current-information or research request.",
        "quiz": "Possibly a quiz or practice request.",
        "roadmap": "Possibly a learning-plan or progress request.",
        "concept": "Possibly an explanation request.",
        "conversation": "Possibly a general conversation.",
        "general_request": "No reliable task category was guessed.",
        "unknown": "No reliable task category was guessed.",
    }
    return hints.get(intent, "No reliable task category was guessed.")


def answer(
    query: str,
    mode: str = "Automatic",
    provider: Optional[ModelProvider] = None,
    recent_history: Optional[list[dict[str, str]]] = None,
) -> tuple[str, Route, str]:
    """Generate a response; return (response_text, route_metadata, provider_name)."""
    if not isinstance(query, str):
        query = str(query) if query is not None else ""

    route = route_message(query)

    if route.note == "too_long":
        return (
            f"Your message is longer than this mentor currently accepts "
            f"({MAX_MESSAGE_LENGTH:,} characters). Please split it into smaller parts.",
            route,
            "No model used",
        )

    if not query.strip():
        return (
            "Tell me a little about what you'd like help with, and we'll take it step by step.",
            route,
            "No model used",
        )

    if route.conversation_mode in {"social", "happy", "supportive"} and not route.requires_model:
        return social_response(query, route), route, "No model used"

    active_provider = None
    try:
        active_provider = provider or available_provider(mode)
        history_context = _format_history(recent_history)

        prompt = f"""
Interpret the user's actual request before choosing how to respond.

Non-binding task hint from a simple keyword router:
{_task_hint(route.intent)}

The hint may be wrong. Do not let it override the user's message.

Instructions for this response:
1. Identify what the user wants from the current message.
2. Answer directly and follow the requested format.
3. Use beginner-friendly language where explanation is needed, but do not add
   an unnecessary lesson to a simple request.
4. If the user asks for code, provide relevant code and run/setup instructions
   when useful.
5. If the user asks to debug or edit code, focus on the specific problem and
   preserve unrelated working behavior.
6. Use earlier conversation only if it helps answer the current request.
7. If a critical detail is missing, ask one concise question. Otherwise proceed
   with a sensible assumption and mention it briefly if needed.
8. Do not claim to have run code, checked live sources, changed files, or deployed
   anything unless you actually did so.
9. Never ask the user to paste API keys or secrets.

Current user message:
{query}
{history_context}
"""
        response = active_provider.generate(SYSTEM_PROMPT, prompt)

        if not isinstance(response, str) or not response.strip():
            return (
                "The AI model returned an empty response. Please try asking again "
                "with a little more detail.",
                route,
                getattr(active_provider, "name", "Unknown provider"),
            )

        return response.strip(), route, getattr(active_provider, "name", "Unknown provider")

    except ProviderError as exc:
        # A failed request may still have reached the provider, so do not claim
        # that no API usage occurred.
        return (
            "I couldn't get a response from the configured AI model.\n\n"
            f"Provider details: {exc}\n\n"
            "Check that your provider is configured correctly, that your API key "
            "is stored in Streamlit Secrets (not in your code), and that the "
            "selected model is available. Then try again.",
            route,
            "Model unavailable",
        )
    except Exception as exc:
        return (
            "Something unexpected happened while generating the response "
            f"({type(exc).__name__}). Please try again. If it keeps happening, "
            "share the error message with any API keys or secrets removed.",
            route,
            "Error",
        )
