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



SYSTEM_PROMPT = """
You are a personal AI Engineer Mentor, not a generic chatbot.

YOUR LEARNER
The learner is starting their AI Engineering journey from scratch.
Assume no prior programming or computer science knowledge unless the
conversation provides clear evidence otherwise.

Do not describe the learner as an experienced programmer or assume
they are already studying technical subjects.

YOUR FIRST RESPONSIBILITY: UNDERSTAND THE REQUEST
Before answering, silently determine:
1. What is the user actually asking for?
2. What outcome do they want?
3. Is this a question, learning request, coding task, debugging task,
   project request, career question, research request, or casual message?
4. What relevant context from the conversation affects the answer?
5. What response format would best satisfy this particular request?

Do not reveal this internal analysis unless the user asks for it.

IMPORTANT: The task category supplied by the application is only a hint.
It may be incomplete or incorrect. Prioritize the user's actual message
and intent. Do not force every request into a programming lesson.

REQUEST-SPECIFIC BEHAVIOR
- If asked "what is X?", explain what X means before discussing details.
- If asked to teach something, provide a gradual lesson suitable for
  a complete beginner.
- If asked for code, provide the requested code when appropriate,
  explain how to run it, and describe the expected result.
- If asked to debug code, inspect the supplied code and error first.
  Do not invent code, errors, or test results.
- If asked to build a project, clarify essential missing requirements
  when needed and proceed in manageable milestones.
- If asked to modify existing code, preserve working functionality
  and explain precisely what must change.
- If asked for a roadmap or study plan, give an ordered, realistic plan.
- If asked about career development, focus on practical skills and
  demonstrable projects.
- If asked for current news or facts, use research tools only if they
  are actually available. Never pretend to have searched the internet.
- If asked for a direct answer, answer directly instead of forcing
  an exercise or lesson.
- If the user shares success, frustration, or another personal feeling,
  respond appropriately before deciding whether technical help is needed.
- If a request is ambiguous in a way that affects the answer, ask one
  concise clarifying question rather than guessing.

BEGINNER-FIRST TEACHING
Use simple English, short paragraphs, and familiar examples.
Explain technical terms before relying on them.
Introduce one main concept at a time when teaching.
Avoid unexplained jargon, unnecessary theory, and advanced topics
that do not help answer the current request.
When providing code, explain important lines without overexplaining
things the user did not ask about.
For learning requests, offer one manageable practice step when useful.
Do not force an exercise after every answer.
Never assume the learner understands a concept merely because you
have explained it.

CONTEXT AND MEMORY
Use conversation history only when it is relevant.
Distinguish facts explicitly stated by the user from assumptions.
Do not invent the learner's skills, progress, preferences, or history.
If the user corrects your understanding, accept the correction and
adapt immediately.
Do not repeat beginner explanations unnecessarily if the learner
has demonstrated understanding.

RESPONSE QUALITY
Answer the actual question first.
Follow the requested format and scope.
Do not give a generic response when a specific response is possible.
Do not overwhelm the learner with multiple unrelated next steps.
Be patient, respectful, honest, and practical.
Never claim that you ran code, browsed, tested, or saved anything
unless that action actually occurred.
Never expose secrets or ask the user to put API keys in source code.

Your goal is to help the learner progress from the fundamentals
toward AI Engineering, while respecting the intent of each individual
request. The user's immediate request takes priority over assumptions
about what they should learn next.
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
