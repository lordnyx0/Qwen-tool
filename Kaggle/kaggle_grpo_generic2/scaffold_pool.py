"""scaffold_pool.py — pool versionado de scaffolds por nivel (Ornith pilar 1).

Scaffold = plano curto (~64 tokens): estrategia + shortlist + criterio de
sucesso + politica de erro. O pool guarda o melhor por nivel (media rolante
de reward); o trainer refina a partir dele e persiste vencedores.
Puro (json): testavel sem modelo/GPU.
"""
import json
import os

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "data", "rl_scaffolds")

SEEDS = {
    "frontier_abstain": ("If the request is greeting, small-talk, opinion, or needs "
                         "no tool: answer directly, no tags. Only call a tool when the "
                         "request names a concrete object or action. Done when answered."),
    "frontier_python": ("Restate the computation. Write one print() program. "
                        "Run mentally once. Done when stdout matches the question."),
    "frontier_discrimination": ("Name the target object. Pick the tool whose description "
                                "covers it; ignore lookalike names. Fill required args only. "
                                "Done when selection matches description."),
    "frontier_multiturn": ("Track entities across turns. Reuse prior outputs, never "
                           "re-ask. On error output: change exactly one thing and retry "
                           "once. Done when final answers the last turn."),
}

REFINE_TPL = ("Task: {task}\nPrevious strategy: {scaffold}\n"
              "Refine the strategy in one short paragraph. "
              "No tool calls, no tags, plain text only.")

GENERIC_SEED = ("Identify the request. Pick the tool whose description covers it. "
                "Fill required args only. Verify the output matches the request. "
                "On error output, change one thing and retry once.")


def pool_path(level):
    return os.path.join(POOL_DIR, f"{level}.json")


def ensure_files(levels=None):
    """Cria pool dir + seeds (idempotente; nunca sobrescreve vencedor)."""
    os.makedirs(POOL_DIR, exist_ok=True)
    items = SEEDS.items() if levels is None else [(k, SEEDS.get(k, GENERIC_SEED)) for k in levels]
    for lvl, text in items:
        fp = pool_path(lvl)
        if not os.path.exists(fp):
            with open(fp, "w", encoding="utf-8") as f:
                json.dump({"best": {"text": text, "mean": 0.0, "n": 0},
                           "history": []}, f, ensure_ascii=False, indent=2)


def get_best(level):
    """Texto do melhor scaffold do nivel (ou seed se pool ausente)."""
    try:
        with open(pool_path(level), encoding="utf-8") as f:
            d = json.load(f)
        return d.get("best", {}).get("text", SEEDS.get(level, GENERIC_SEED))
    except (OSError, ValueError, KeyError):
        return SEEDS.get(level, GENERIC_SEED)


def build_refine_prompt(task_text, level):
    return REFINE_TPL.format(task=task_text[:800], scaffold=get_best(level)[:800])


def update(level, scaffold_text, reward, cap=50):
    """Atualiza media rolante; promove a best se superar. Retorna (is_best, mean)."""
    ensure_files([level])
    fp = pool_path(level)
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    hist = d.get("history", [])
    entry = None
    for h in hist:
        if h.get("text") == scaffold_text:
            entry = h
            break
    if entry is None:
        entry = {"text": scaffold_text, "mean": 0.0, "n": 0}
        hist.append(entry)
    entry["n"] += 1
    entry["mean"] += (reward - entry["mean"]) / entry["n"]
    hist = hist[-cap:]
    d["history"] = hist
    is_best = entry["mean"] > d.get("best", {}).get("mean", float("-inf"))
    if is_best:
        d["best"] = {"text": entry["text"], "mean": entry["mean"], "n": entry["n"]}
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return is_best, entry["mean"]
