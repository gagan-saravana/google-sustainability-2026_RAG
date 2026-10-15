"""
app.py: NiceGUI chat interface for the Google 2026 Environmental Report RAG demo.

    python app.py            # then open http://localhost:8080

Design decisions:
* No accounts. NiceGUI's signed session cookie gives each browser an anonymous id
  (app.storage.browser['id']); limits and history live in server memory (src/limits.py).
* 3 questions per browser + a global daily cap (the real protection for the free Gemini quota).
  A failed model call refunds the question; double-submits are blocked.
* History = those few exchanges only. Follow-ups are "condensed" into standalone
  questions (src/condense.py) before retrieval. A page refresh restores the chat.
* Blocking work (Gemini + Chroma calls) runs via run.io_bound so the UI never freezes.
* LLM output is untrusted text: it is HTML-escaped before markdown rendering
  (and NiceGUI >= 3.7 sanitises ui.markdown as well).
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
from dataclasses import dataclass

from nicegui import app, run, ui

from src import config, limits
from src.condense import condense_question
from src.limits import DemoLimiter
from src.synthesize import Answer, answer_question

log = logging.getLogger("rag-demo")
limiter = DemoLimiter(config.MAX_PROMPTS_PER_BROWSER, config.DAILY_CAP, config.MAX_TRACKED_BROWSERS)

# Each of these is answered by the report (checked against the source text).
EXAMPLE_QUESTIONS = [
    "How much new clean energy did Google contract in 2025?",
    "What was Google's fleet-wide data center PUE in 2025?",
    "How much of its freshwater consumption did Google replenish?",
    "How much did the energy per Gemini prompt fall?",
]

NOTICES = {
    limits.BROWSER_LIMIT: (
        f"You've used all {config.MAX_PROMPTS_PER_BROWSER} demo questions. "
        "Thanks for trying it. The code and design notes are in the repository."
    ),
    limits.DAILY_CAP: "Today's demo capacity has been reached. Please come back tomorrow.",
}

CSS = """
:root { --bg:#09090b; --card:#111113; --border:#27272a; --fg:#fafafa; --muted:#a1a1aa; }
html, body, .q-page, .nicegui-content { background: var(--bg) !important; color: var(--fg);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.nicegui-content { padding: 0 !important; }
.shell { width:100%; max-width:42rem; margin:0 auto; height:100vh; height:100dvh; gap:0 !important; flex-wrap:nowrap; }
.header { width:100%; padding:16px 20px; border-bottom:1px solid var(--border); flex-wrap:nowrap;
  justify-content:space-between; align-items:center; }
.title { font-weight:600; font-size:15px; letter-spacing:-0.01em; }
.badge { font-size:12px; color:var(--muted); border:1px solid var(--border); border-radius:999px; padding:2px 10px; }
.chat { flex:1 1 0; min-height:0; width:100%; }
.chat .q-scrollarea__content { width:100%; }
.messages { width:100%; padding:20px; gap:14px !important; }
.bubble-user { align-self:flex-end; background:var(--fg); color:#09090b; border-radius:12px 12px 2px 12px;
  padding:9px 14px; max-width:85%; font-size:14px; line-height:1.5; white-space:pre-wrap; }
.bubble-bot { align-self:flex-start; width:100%; background:var(--card); border:1px solid var(--border);
  border-radius:12px; padding:14px 16px; gap:8px !important; font-size:14px; line-height:1.6; }
.bubble-bot p { margin:0 0 8px; }
.meta { font-size:12px; color:var(--muted); }
.thinking { gap:10px !important; align-items:center; flex-wrap:nowrap; }
.chips { gap:8px !important; }
.chip { text-transform:none !important; border-radius:8px !important; color:#d4d4d8 !important;
  font-size:12.5px; text-align:left; }
.sources { width:100%; border:1px solid var(--border); border-radius:8px; background:transparent; }
.sources .q-item { font-size:12.5px; color:var(--muted); min-height:36px; }
.source { width:100%; padding:8px 14px; gap:2px !important; border-top:1px solid var(--border); }
.source-title { font-weight:500; font-size:13px; }
.snippet { font-size:12.5px; color:#d4d4d8; white-space:pre-wrap; line-height:1.5; }
.composer { width:100%; padding:12px 20px 6px; gap:6px !important; border-top:1px solid var(--border); }
.composer-row { width:100%; flex-wrap:nowrap; gap:8px !important; align-items:center; }
.q-field--outlined .q-field__control { border-radius:10px; }
.q-field--outlined .q-field__control:before { border-color: var(--border) !important; }
.notice { width:100%; text-align:center; font-size:13px; color:var(--muted); padding:8px; }
.disclaimer { width:100%; text-align:center; font-size:11.5px; color:#71717a; padding:0 20px 12px; }
"""

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
"""


@dataclass
class Exchange:
    question: str      # what the user typed
    standalone: str    # what we actually searched for (differs on follow-ups)
    answer: Answer


def _escape(text: str) -> str:
    """Neutralise raw HTML in model output before markdown rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


def render_answer(bubble: ui.column, ex: Exchange) -> None:
    bubble.clear()
    with bubble:
        ui.markdown(_escape(ex.answer.text))
        if ex.standalone.strip().lower() != ex.question.strip().lower():
            ui.label(f"Searched for: {ex.standalone}").classes("meta")
        cited = ex.answer.cited_sources
        if cited:
            with ui.expansion(f"Sources ({len(cited)})", icon="article").classes("sources").props("dense dark"):
                for n, src in cited:
                    with ui.column().classes("source"):
                        ui.label(f"[{n}] {src.section_path or 'Report'}").classes("source-title")
                        ui.label(f"{src.type} · similarity {src.similarity:.2f}").classes("meta")
                        ui.label(src.text[:500] + ("…" if len(src.text) > 500 else "")).classes("snippet")


@ui.page("/")
def index() -> None:
    ui.dark_mode(True)
    ui.add_css(CSS)
    ui.add_head_html(HEAD_HTML)
    browser_id = app.storage.browser["id"]
    max_q = config.MAX_PROMPTS_PER_BROWSER
    chips: list[ui.button] = []

    # ---- layout ---------------------------------------------------------------
    with ui.column().classes("shell"):
        with ui.row().classes("header"):
            ui.label("Google 2026 Environmental Report").classes("title")
            badge = ui.label().classes("badge")
        scroll = ui.scroll_area().classes("chat")
        with scroll:
            messages = ui.column().classes("messages")
        composer = ui.column().classes("composer")
        ui.label(
            "Answers are generated from the report by an AI model and can contain mistakes. "
            "Check important figures against the source."
        ).classes("disclaimer")

    # ---- helpers ----------------------------------------------------------------
    def update_badge() -> None:
        badge.set_text(f"{limiter.remaining(browser_id)} of {max_q} questions left")

    async def scroll_to_bottom() -> None:
        try:
            await asyncio.sleep(0.05)
            scroll.scroll_to(percent=1.0)
        except Exception:   # purely cosmetic: never let it break a turn
            pass

    def set_controls(enabled: bool) -> None:
        for el in (question_input, send_button, *chips):
            el.enable() if enabled else el.disable()

    def lock_composer(reason: str) -> None:
        composer.clear()
        with composer:
            ui.label(NOTICES.get(reason, "The demo is unavailable right now.")).classes("notice")
        for chip in chips:
            chip.disable()

    # ---- the one place a question is handled -----------------------------------------
    async def submit(raw: str | None) -> None:
        text = (raw or "").strip()
        if not text:
            return
        if len(text) > config.MAX_QUESTION_CHARS:
            ui.notify(f"Please keep questions under {config.MAX_QUESTION_CHARS} characters.", type="warning")
            return
        reason = limiter.reserve(browser_id)       # server-side check BEFORE any API call
        if reason == limits.BUSY:
            ui.notify("Still working on your previous question.", type="info")
            return
        if reason:
            lock_composer(reason)
            update_badge()
            return

        committed = False
        bubble = None
        try:
            question_input.set_value("")
            set_controls(False)
            with messages:
                ui.label(text).classes("bubble-user")
                bubble = ui.column().classes("bubble-bot")
                with bubble:
                    with ui.row().classes("thinking"):
                        ui.spinner(size="sm")
                        ui.label("Searching the report…").classes("meta")
            await scroll_to_bottom()

            past = [(e.question, e.answer.text) for e in limiter.session(browser_id).history]
            standalone = await run.io_bound(condense_question, past, text)
            answer = await run.io_bound(answer_question, standalone)

            exchange = Exchange(text, standalone, answer)
            limiter.commit(browser_id, exchange)
            committed = True
            render_answer(bubble, exchange)
            update_badge()
            if limiter.remaining(browser_id) == 0:
                lock_composer(limits.BROWSER_LIMIT)
            else:
                set_controls(True)
            await scroll_to_bottom()
        except (Exception, SystemExit):
            log.exception("Question failed")
            try:
                if bubble is not None:
                    bubble.clear()
                    with bubble:
                        ui.label(
                            "Something went wrong while answering. "
                            "Your question was not counted. Please try again in a moment."
                        ).classes("meta")
                set_controls(True)
            except Exception:
                pass
        finally:
            if not committed:
                limiter.release(browser_id)         # refund the slot on any failure
                try:
                    update_badge()
                except Exception:
                    pass

    # ---- composer (input + send) ---------------------------------------------------------
    with composer:
        with ui.row().classes("composer-row"):
            question_input = (
                ui.input(placeholder="Ask about the report…")
                .props(f"outlined dense dark autofocus maxlength={config.MAX_QUESTION_CHARS}")
                .classes("grow")
            )
            send_button = ui.button("Send").props("unelevated no-caps color=white text-color=black")

    async def on_send() -> None:
        await submit(question_input.value)

    question_input.on("keydown.enter", on_send)
    send_button.on_click(on_send)

    # ---- opening message with the topic + clickable examples ---------------------------------
    with messages:
        with ui.column().classes("bubble-bot"):
            ui.markdown(
                "**Ask me about Google's 2026 Environmental Report.**\n\n"
                "Topics include clean energy, data center efficiency, AI's energy footprint, "
                f"water stewardship and emissions. You have {max_q} questions in this demo. Try one:"
            )
            with ui.row().classes("chips"):
                for q in EXAMPLE_QUESTIONS:
                    async def ask_example(q: str = q) -> None:
                        await submit(q)

                    chips.append(ui.button(q, on_click=ask_example).props("outline no-caps dense").classes("chip"))

        # restore this browser's earlier exchanges after a page refresh
        for ex in limiter.session(browser_id).history:
            ui.label(ex.question).classes("bubble-user")
            render_answer(ui.column().classes("bubble-bot"), ex)

    update_badge()
    if limiter.remaining(browser_id) == 0:
        lock_composer(limits.BROWSER_LIMIT)
    elif limiter.daily_remaining() == 0:
        lock_composer(limits.DAILY_CAP)


def main() -> None:
    # Fail fast with a clear message instead of on the first question.
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and paste your key.")
    from src.retrieve import open_collection

    print(f"Index OK: {open_collection().count()} chunks | "
          f"{config.MAX_PROMPTS_PER_BROWSER} questions/browser, {config.DAILY_CAP}/day overall")

    ui.run(
        title="Google 2026 Environmental Report Q&A",
        favicon="🌱",
        dark=True,
        reload=False,
        port=int(os.getenv("PORT", "8080")),
        storage_secret=config.STORAGE_SECRET or secrets.token_urlsafe(32),
    )


if __name__ == "__main__":
    main()
