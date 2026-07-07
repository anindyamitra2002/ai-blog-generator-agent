# Changes in this update

## 1. Root cause of the hallucination

For a very recent/local/niche topic (e.g. "the Baruipur minor girl tragedy"),
the search tools often return little or no real evidence. The old
`synthesize_node` prompt said "do not invent facts," but that's a soft
instruction — smaller/free models (Gemini Flash free tier, qwen2.5:7b, or
whatever "auto" happens to route to on Omniroute) will still fall back on
their own training data to produce a plausible-sounding brief when the
evidence is thin, and that's exactly the "mixed up with other old
data"/hallucination you saw. Nothing in the old pipeline detected or flagged
this — it just quietly produced a brief and moved on.

## 2. What changed

### a) Category prompts are now real files, not a hardcoded dict
`agent/research_agent.py` used to embed a short, one-line "focus + trusted
domains" string per category directly in the module (`CATEGORY_INSTRUCTIONS`).
That's gone. Instead:

- `prompts/research/<category>.md` — 21 files, one per category — contain the
  full research/grounding instructions for that category (focus area, trusted
  domains, output structure, and the anti-hallucination rules below).
- `agent/research_agent.py` now has `load_category_prompt(category)`, which
  reads the matching file (falling back to the default category if missing).
- `prompts/_generate_prompts.py` is the generator that produced those 21
  files from a single template — rerun it if you need to change a category's
  focus/domains/rules everywhere consistently.
- The **short** directive (category + focus + trusted domains) is extracted
  from the file for the `plan` step (keeps planning prompts small); the
  **full** file — including the grounding rules — is used verbatim for the
  `synthesize` step, since that's where fabrication actually happens.

### b) Explicit, hard anti-hallucination rules in every category prompt
Every `prompts/research/*.md` file now has a
`CRITICAL GROUNDING & ANTI-HALLUCINATION RULES` section that says, in short:
use ONLY the raw search evidence given this run; never your own training
knowledge; if evidence is thin, say so explicitly instead of padding; every
cited URL must literally appear in the evidence, never a remembered/plausible
one.

The same "don't use outside knowledge, don't invent to fill gaps" language
was also added to `chains/outline_chain.py`, `chains/writer_chain.py`, and
`chains/editor_chain.py`, since those stages can fabricate too if the brief
has gaps.

### c) Evidence-sufficiency tracking (new safety net)
`agent/research_agent.py`'s `synthesize_node` now measures how much *genuine*
(non-error, non-empty) evidence was actually collected. If it's below
`MIN_EVIDENCE_CHARS` (env-configurable, default 400), the brief is flagged
`evidence_insufficient = True` and the synthesis prompt is told explicitly to
write an honest "nothing verified was found" brief rather than fill the gap.

`run_research()` now returns the full state dict (not just the brief string)
so callers can see this flag. `main.py` checks it after the research phase:
- If insufficient and `--force-thin` was **not** passed: prints a clear
  warning, prints the (honest, likely short) brief it did get, and **stops
  the pipeline before writing a post** — so you never silently get a
  hallucinated blog post for a topic search couldn't actually find.
- If `--force-thin` **is** passed: continues anyway with whatever the
  research phase produced.

### d) Memory (Mem0) removed from the pipeline
`main.py` no longer imports or calls `memory/mem0_store.py` at all — the
"check memory for similar past topics" and "record to memory" stages are
gone. That file is left in the repo for reference / future reintroduction,
but nothing wires it up anymore, per your request that memory not influence
new generations with old content. `config.py`'s `MEM0_USER_ID` is left as a
harmless unused setting with a comment explaining it's dormant.

## 3. New/changed files

- `prompts/research/*.md` — new, 21 files.
- `prompts/_generate_prompts.py` — new, the generator for the above.
- `agent/research_agent.py` — rewritten (loads prompt files, evidence
  tracking, `run_research()` now returns the full state dict).
- `agent/state.py` — added `evidence_insufficient: bool` field.
- `main.py` — rewritten (no memory stages, evidence-insufficient guard,
  new `--force-thin` flag, removed `--skip-memory`).
- `config.py` — added `MIN_EVIDENCE_CHARS`; `MEM0_USER_ID` kept but commented
  as dormant.
- `chains/outline_chain.py`, `chains/writer_chain.py`, `chains/editor_chain.py`
  — added explicit "don't use outside knowledge / don't invent to fill gaps"
  language.
- `.env.example` — documents the new `MIN_EVIDENCE_CHARS` var and notes Mem0
  vars are dormant.
- `memory/mem0_store.py` — unchanged, just no longer called from `main.py`.

## 4. Trying it

```bash
python main.py "Explain the Baruipur minor girl tragedy"
```

If search genuinely can't find coverage, you'll now see something like:

```
[warning] The search tools could not find substantive evidence for '...'.
Stopping here to avoid a hallucinated post. Re-run with --force-thin ...
--- Research brief so far ---
_Research compiled on July 07, 2026. Recency window: last 14 days for news._

## Definition
No verified information could be found in the available search sources for this topic as of July 07, 2026.
...
```

instead of a confidently-written but fabricated blog post.


Here the markdown building is not proper. Full links are also seen in the preview of the output post. MD. And it only used 4 sources. It must use at least 10+ sources. Also use the Exa search tool and LinkupSearchTool along with Travily. Don't use DuckDuckGo at first. It will be a fallback option. Use all the 10 source contents to write the blogs and mention the links in the source section.



Update necessary code files.