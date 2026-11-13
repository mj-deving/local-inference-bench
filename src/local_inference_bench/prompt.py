"""The fixed-context prompt. Every runtime gets byte-identical messages.

Adapted from agentic-rag-mastra src/mastra/agents/instructions.ts (prompt v2). The only change:
the agent there calls a search tool; here the same top-5 chunks are pasted into the user message,
so the runtime is the only variable. The tool sentences are rewritten to refer to the excerpts.
"""

from .grading import REFUSAL_MARKER

SYSTEM = f"""Du beantwortest Fragen zur KI-Verordnung (EU) 2024/1689 ausschließlich anhand der Textstellen,
die dir in der Nachricht als Auszüge mit chunkId gegeben werden. Du gibst keine Rechtsberatung und nutzt kein eigenes Vorwissen.

Vorgehen:
- Beantworte die Frage knapp auf Deutsch, nur mit dem, was in den Auszügen steht.
- Belege jede Aussage direkt dahinter mit der chunkId in genau dieser Form: [chunk:<chunkId>], zum Beispiel [chunk:art50-p1].
- Zitiere nur chunkIds, die in den Auszügen vorkommen.
- Übernimm Zahlen, Beträge, Prozentsätze und Daten wörtlich aus der Textstelle.

Prüfung vor jeder Antwort:
- Thematische Nähe reicht nicht. Antworte nur, wenn eine Textstelle genau die gefragte Sache ausdrücklich regelt.
- Fragt die Frage nach einer Definition, einem Verfahren, einem Artikel oder einer Stelle, die in den Textstellen nicht wörtlich vorkommt, dann verweigere.
- Fragt die Frage nach Ausnahmen oder Bedingungen, nenne alle, die die Textstelle nennt, nicht nur eine.

Verweigerung:
- Wenn die Auszüge die Frage nicht beantworten, antwortest du mit genau einer Zeile:
  {REFUSAL_MARKER} Die abgerufenen Auszüge enthalten dazu keine Grundlage.
- Rate nicht und ergänze nichts aus Vorwissen, auch wenn du die Antwort zu kennen glaubst."""


def messages(question: str, chunks: list[dict], request_id: str) -> list[dict]:
    """request_id makes every request unique after the shared system prompt, so no runtime can
    serve a repeated request from an exact-prompt cache. The system prompt stays a cacheable prefix,
    as it would be in a real RAG service."""
    excerpts = "\n\n".join(f"chunkId: {c['id']}\n{c['text']}" for c in chunks)
    # "/no_think" is Qwen3's in-prompt switch; it travels identically to every runtime.
    user = f"Anfrage {request_id}\n\nAuszüge:\n\n{excerpts}\n\nFrage: {question} /no_think"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
