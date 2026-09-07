You are working on my Electrical Engineering minor project: **Grid Stability
Prediction**. Before doing anything else, fully read `AGENTS.md` (or `context.md`
if that's what it's named — project context, locked-in tech decisions, and scope)
and `roadmap.md` (the phase-by-phase plan with tasks and acceptance criteria).

Your job, across this session and future ones: take this project from zero to a
working, well-structured, defensible ML + power-systems pipeline — research,
simulation-based dataset generation, modeling, explainability, a small demo, and
eventually a report and two PowerPoint decks (a mid-sem progress deck now, a final
deck later). Follow `roadmap.md`'s phase order — don't skip ahead to modeling
before the simulation/dataset phase is solid.

**Ground rules:**

1. I'm not a power-systems expert, and no teammate is checking the physics side —
   you are the domain check. Before generating the dataset, do real research
   (literature + ANDES documentation) and briefly justify your scenario-generation
   and stability-labeling choices in a research-notes file. Don't guess at
   load-flow/dynamics parameters — ground them in ANDES's documented defaults/
   examples or in cited literature.
2. Confirm exact ANDES API details (bundled case file names/paths for the IEEE
   39-bus system, function signatures, etc.) against the installed package version
   and its docs rather than assuming — don't take file paths in `AGENTS.md` as
   gospel if the installed version disagrees.
3. **Priority right now is everything needed for the mid-sem PPT**: problem
   framing, literature summary, pipeline architecture, a working (even if
   small-scale) simulation + dataset, and initial EDA. Stop and build that PPT
   before going deeper into modeling — tell me explicitly when you reach that
   checkpoint, and show me the deck before considering it done.
3a. This project runs the full academic year, not just this semester — there are
   three checkpoints total: the mid-sem PPT (now), a midway PPT at the end of
   this semester (once baselines and the generalization experiments are done —
   this closes out the minor project), and a final PPT at the end of the year
   once this continues as the major project next semester. Don't rush toward
   "final" deliverables (CCT, explainability, demo, final report) before the
   midway checkpoint — those belong to Part B of `roadmap.md`.
4. Structure the repo for maintainability from the start, per `AGENTS.md`'s
   "production quality" section. Spending a bit more time on structure early beats
   redoing it later.
5. When you hit a genuine fork in the road — a modeling choice, a scope cut,
   something the roadmap leaves ambiguous — ask me rather than silently deciding.
   Don't ask about things `AGENTS.md` or `roadmap.md` already answer.
6. Keep `roadmap.md`'s checklist (or your own generated task list) updated as you
   go, so progress is visible at a glance.

Start with Phase 0 — setup and research — now.