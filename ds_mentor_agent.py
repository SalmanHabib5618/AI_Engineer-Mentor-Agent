"""AI Engineer Mentor: intent-aware, beginner-first, provider-independent workflow."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from model_provider import ModelProvider, available_provider, ProviderError


@dataclass
class Route:
    conversation_mode: str
    intent: str
    requires_model: bool
    note: str = ""


GREETING_PATTERNS = (
    r"^(hi|hello|hey|hey there|hi mentor|hello mentor|good morning|good afternoon|good evening|good night)[!. ]*$",
    r"^(thanks|thank you|thx|ok|okay|got it|bye|goodbye)[!. ]*$",
)
HAPPY_TERMS = (
    "i'm happy", "i am happy", "i did it", "i finally", "i passed", "i succeeded",
    "i finished", "i completed", "great news", "i got the job", "i solved it",
    "i understand now", "i got it working", "i'm excited", "i am excited",
)
SUPPORT_TERMS = (
    "i'm sad", "i am sad", "i feel sad", "i'm stressed", "i am stressed",
    "i'm frustrated", "i am frustrated", "i feel stupid", "i can't do this",
    "i cannot do this", "i give up", "i feel hopeless", "i'm overwhelmed",
    "i am overwhelmed", "i'm discouraged", "i am discouraged", "i don't understand anything",
)
INTENT_KEYWORDS = {
    "code": ("code", "python", "function", "script", "program", "syntax", "debug", "error", "exception", "bug"),
    "project": ("project", "build", "implement", "application", "app", "portfolio"),
    "career": ("job", "career", "resume", "cv", "interview", "hiring", "job-ready", "job ready"),
    "research": ("latest", "current", "today", "recent", "new version", "release", "compare sources", "official docs"),
    "quiz": ("quiz", "test me", "practice questions", "exercise", "test my knowledge"),
    "roadmap": ("roadmap", "what should i learn", "next lesson", "study plan", "learn today", "my progress"),
    "concept": ("what is", "explain", "teach me", "how does", "difference between", "meaning of"),
}


SYSTEM_PROMPT = """You are the user's personal AI Engineer Mentor, learning partner, teacher, and guide.
The user is starting from scratch and wants to progress toward job readiness. Use guided learning:
- Be patient, respectful, encouraging, and beginner-friendly.
- Explain unfamiliar terms in plain language before using them.
- For learning questions, explain the idea, give an analogy or example, then a small practice step.
- For code, explain the approach, provide readable code, explain how to run it, and describe expected output.
- For debugging, identify the likely cause, give a minimal fix, and explain how to verify it.
- For project requests, break the work into small milestones and build one step at a time.
- For career advice, focus on demonstrable skills and realistic milestones, not guaranteed employment.
- Match the requested depth; greetings and thanks should get short natural replies.
- When the user expresses feelings, acknowledge what they said without assuming more than they expressed.
  If they are discouraged, offer support and a manageable next step before a long lesson.
  If they share success, celebrate the specific achievement without forcing another task.
- Do not pretend to have browsed, tested code, saved something, or run an action that did not happen.
- Distinguish current facts from general knowledge. If sources are unavailable, say so.
- Never reveal, request, or place secrets in source code. Never automatically execute generated code.
- Do not claim the user has learned something merely because it was discussed.
- Keep the focus on the user's complete AI Engineering journey from scratch to job-ready.
"""


def route_message(query: str) -> Route:
    text = query.strip()
    lower = text.lower()
    if not text:
        return Route("clarification", "unknown", False, "empty")
    if len(text) > 8000:
        return Route("clarification", "unknown", False, "too_long")
    if any(re.match(p, lower) for p in GREETING_PATTERNS):
        return Route("social", "greeting", False)
    if any(term in lower for term in HAPPY_TERMS):
        # If the message also contains a technical question, keep that task as a secondary intent.
        intent = _technical_intent(lower)
        return Route("happy", intent or "conversation", bool(intent), "celebration")
    if any(term in lower for term in SUPPORT_TERMS):
        intent = _technical_intent(lower)
        return Route("supportive", intent or "conversation", bool(intent), "support")
    intent = _technical_intent(lower) or "concept"
    return Route("learning", intent, True)


def _technical_intent(lower: str) -> Optional[str]:
    for intent, terms in INTENT_KEYWORDS.items():
        if any(term in lower for term in terms):
            return intent
    return None


def social_response(query: str, route: Route) -> str:
    lower = query.lower().strip()
    if route.conversation_mode == "happy":
        return "That's wonderful! 🎉 I'm glad you shared that with me. What feels like the biggest win for you?"
    if route.conversation_mode == "supportive":
        return (
            "I'm sorry this feels difficult right now. You don't have to solve everything at once. "
            "We can slow down and take one manageable step together. Would you prefer to talk it through "
            "for a moment, or work on the specific problem?"
        )
    if lower.startswith(("thank", "thx")):
        return "You're welcome! 😊 I'm here to help you keep moving forward, one step at a time."
    if lower.startswith(("bye", "goodbye", "good night")):
        return "Take care! Rest is part of learning too. We'll continue your AI Engineering journey when you're ready."
    return "Hi! 👋 I'm your AI Engineer Mentor and learning partner. What would you like to explore today?"


def answer(
    query: str,
    mode: str = "Automatic",
    provider: Optional[ModelProvider] = None,
    recent_history: Optional[list[dict[str, str]]] = None,
) -> tuple[str, Route, str]:
    route = route_message(query)
    if route.note == "too_long":
        return "That message is quite long. Could you split it into smaller parts so I can help accurately?", route, "No model used"
    if not query.strip():
        return "Tell me a little about what you'd like help with, and we'll take it step by step.", route, "No model used"

    if route.conversation_mode in {"social", "happy"} and not route.requires_model:
        return social_response(query, route), route, "No model used"
    if route.conversation_mode == "supportive" and not route.requires_model:
        return social_response(query, route), route, "No model used"

    try:
        active = provider or available_provider(mode)
        history = recent_history or []
        context = ""
        if history:
            relevant = history[-6:]
            context = "\n\nRecent conversation context (may be incomplete):\n" + "\n".join(
                f"{m.get('role','')}: {m.get('content','')[:700]}" for m in relevant
            )
        task = {
            "code": "Help with the code or programming problem. Explain the reasoning and how to verify the result.",
            "project": "Guide the user through a project in small milestones. Start with the next concrete step, not a giant code dump.",
            "career": "Give realistic AI Engineering career guidance with skill milestones and practical portfolio evidence.",
            "research": "The user may need fresh information. If online research is not implemented/available, be transparent and explain what needs verification.",
            "quiz": "Run a guided practice session. Ask one question at a time unless the user explicitly requests a full quiz.",
            "roadmap": "Use a practical beginner roadmap and suggest the next manageable step. Do not claim to know saved progress unless provided.",
            "concept": "Teach the concept from first principles with a simple example and one small check for understanding.",
            "conversation": "Respond naturally and supportively to the user's message.",
        }.get(route.intent, "Help the user in a beginner-friendly way.")
        prompt = f"{task}\n\nUser message:\n{query}{context}"
        response = active.generate(SYSTEM_PROMPT, prompt)
        return response, route, active.name
    except ProviderError as exc:
        return (
            f"I’m ready to help, but the AI model is not available yet.\n\n{exc}\n\n"
            "You can still chat with me for greetings and simple encouragement. "
            "For technical answers, configure the temporary API provider or set up a local model. "
            "No paid API call was made by this fallback response."
        ), route, "Model unavailable"
    except Exception as exc:
        return (
            f"Something unexpected happened ({type(exc).__name__}). "
            "Please try again with a shorter question. No success is being assumed."
        ), route, "Error"
