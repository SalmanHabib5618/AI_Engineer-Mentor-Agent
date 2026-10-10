# AI Engineer Mentor

A beginner-first personal mentor designed to guide the user from programming foundations toward job-ready AI Engineering skills.

## Current implementation: Phase 1 foundation

- Greeting, supportive, happy, learning, and technical response modes.
- Simple intent routing for concepts, coding, projects, career, research requests, quizzes, and roadmap questions.
- Replaceable model-provider interface.
- Temporary Gemini API provider, enabled only when a key is configured.
- Optional local Ollama provider (requires Ollama and a downloaded model).
- Automatic mode prefers a reachable local Ollama server; otherwise uses Gemini only if `GOOGLE_API_KEY` is configured.
- Local SQLite storage for conversation history and a basic learning-stage field.
- Clear saved chat/progress control.
- Graceful message when no model is configured.
- No online research implementation yet; current requests needing fresh information are told when verification is required.

## Requirements

- Python 3.10 or newer recommended.
- Windows, macOS, or Linux.
- Internet only if you choose the API provider or later add online research.
- Ollama is optional at this stage.

## Run locally

1. Extract this project folder.
2. Open a terminal in the project folder.
3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate it on Windows:

   ```powershell
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

## Configure temporary Gemini API

Set `GOOGLE_API_KEY` in your environment or in `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-key-here"
```

Never commit your key or share it in chat. Check your provider's current terms, free-tier limits, and pricing before use. The app does not promise that API access is free.

## Configure local Ollama (optional)

1. Install Ollama from its official website.
2. Download a model that fits your available memory and disk space.
3. Set `OLLAMA_MODEL` to the exact downloaded model name if it differs from the default.
4. Select `Local (Ollama)` or `Automatic` in the app.

The default model name is only a configuration default, not a recommendation that it will fit every computer. Test a small quantized model against your hardware before committing to it.

## Privacy

Conversation history and learning-stage data are stored in `data/mentor_memory.sqlite3` on this computer. The `data/` directory is excluded from Git by `.gitignore`. API mode sends prompts to the configured provider. Local mode sends prompts to the local Ollama server. Do not enter passwords or other secrets into the mentor.

## Known limitations in this first build

- Online multi-source research is not implemented yet.
- The learning roadmap is currently a basic saved stage, not a full curriculum tracker.
- Uploaded file analysis from the old UI is not included in this first rebuild.
- Emotional mode detection uses simple keyword rules and may miss subtle context.
- Automated tests and real model calls should be run before relying on the app.
- The UI still imports Google Fonts when internet is available; remove that import for stricter offline styling if desired.
