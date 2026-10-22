"""P-Check multilingual stress test (no training, free GPU).

Question: how much does a personalized LLM judge lose when the user's past
conversations are not in English?

Pipeline (each stage is a CLI call so every model runs in its own process,
which frees the GPU cleanly and lets two GPUs work on halves in parallel):

  prepare   sample N PRISM test users, parse their histories
  translate translate histories EN -> KO / ZH / MS with NLLB-200
  gp        Llama-3.1-8B writes each user's General Profile (GP) from the
            history in each language version
  checklist base Llama-3.2-3B writes a P-Check style checklist per test pair
  judge     Llama-3.1-8B scores chosen and rejected responses, (a) plainly
            from the GP and (b) with the checklist
  eval      accuracy per language, bootstrap CIs, paired differences vs EN

Only the user's history is translated. The test query and the two candidate
answers stay in English, so any accuracy change comes from the history
language alone.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

REPO = os.environ.get("PCHECK_REPO", "P-Check_")
OUT = os.environ.get("PCHECK_OUT", "out")
N_USERS = int(os.environ.get("PCHECK_N_USERS", "100"))
SEED = 42

LANGS = ["en", "ko", "zh", "ms"]
VERSIONS = LANGS + ["mixed"]
# "mixed": past conversation i is in MIX_CYCLE[i % 4], like a user who
# switches language between sessions.
MIX_CYCLE = ["en", "zh", "ms", "ko"]
NLLB_CODES = {"ko": "kor_Hang", "zh": "zho_Hans", "ms": "zsm_Latn"}

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
CHECKLIST_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# 4-bit AWQ build of Llama-3.1-8B-Instruct: fits one 16 GB T4.
JUDGE_MODEL = os.environ.get("PCHECK_JUDGE", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

MAX_MODEL_LEN = 16384
GP_MAX_TOKENS = 1000
CHECKLIST_MAX_TOKENS = 2000
JUDGE_CK_MAX_TOKENS = 2500
JUDGE_PLAIN_MAX_TOKENS = 400
TEMPERATURE = 0.0  # the repo samples at 1.0; greedy keeps runs repeatable
CHUNK = 256  # prompts per vLLM call; results are saved after each chunk

GP_SYSTEM = "You are a helpful assistant."  # as in the repo's openai_async.py
CHECKLIST_SYSTEM = "You are a rigorous personalization checklist designer."  # repo
JUDGE_SYSTEM = ("You are a strict personalization verifier. Evaluate ONLY with the "
                "given inputs; do not use outside knowledge.")  # repo

# The trained P-Check model learned the output format from data. The base
# model has not, so the schema is spelled out.
CHECKLIST_SCHEMA_HINT = (
    "- Output format: return ONLY one JSON object, no other text, in this shape:\n"
    '  {"checklist": [{"evidence": "...", "facet": "...", "criterion": "...", '
    '"category": "Essential" | "Important" | "Optional"}, ...]}\n'
    "- Write the checklist in English.\n"
)

PLAIN_JUDGE_PROMPT = """##Task
Rate how well the [Candidate Model response] is personalized for this user, on a 1-10 scale. Use [User GP] and [Current User Query] to understand what this user wants. 10 = exactly what this user would prefer; 1 = clearly not what this user would prefer.

First give brief reasoning, then the score.

[User GP]: {GP}

[Current User Query]: {query}

[Candidate Model response]: {response}

Return ONLY a single JSON object with this shape (no extra text):
{"reasoning": "<brief reasoning>", "score": <1-10 score>}
[Verify Result] (Json format):"""


def p(*parts):
    return os.path.join(OUT, *parts)


def fill(template, **kw):
    """Replace {key} placeholders without touching other braces (JSON examples)."""
    return re.sub(r"\{(" + "|".join(map(re.escape, kw)) + r")\}",
                  lambda m: str(kw[m.group(1)]), template)


def read_jsonl(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# PRISM history parsing
# --------------------------------------------------------------------------

HIST_START = "[Interaction History of User]\n<previous interaction 1>:\n"
FOOTER_START = "[User’s General Profile] (GP) (Write only"
BLOCK_RE = re.compile(
    r"<previous interaction (\d+)>:\n\[User Query\]:(.*?)\n\n\[Chosen Model Response\]:(.*?)"
    r"\n\n\[Rejected Model Response\]:(.*?)\n\n(?=<previous interaction \d+>:\n|$)",
    re.S)


def parse_gp_input(model_input):
    """Split the repo's GP-generation prompt into header, interactions, footer."""
    h = model_input.index(HIST_START)
    f = model_input.index(FOOTER_START)
    header = model_input[:h] + "[Interaction History of User]\n"
    body = model_input[h + len("[Interaction History of User]\n"):f]
    footer = model_input[f:]
    # The body ends with "\n\n\n" + whatever padding; keep it exactly.
    m_end = re.search(r"\n*$", body)
    tail = body[m_end.start():]
    body_core = body[:m_end.start()] + "\n\n"
    inters = [{"query": q, "chosen": c, "reject": r}
              for _, q, c, r in BLOCK_RE.findall(body_core)]
    sep = tail[2:] if tail.startswith("\n\n") else tail
    return {"header": header, "interactions": inters, "sep": sep, "footer": footer}


def build_gp_input(parsed, interactions, start_index=1):
    blocks = []
    for k, it in enumerate(interactions):
        blocks.append(f"<previous interaction {k + start_index}>:\n[User Query]:{it['query']}\n\n"
                      f"[Chosen Model Response]:{it['chosen']}\n\n"
                      f"[Rejected Model Response]:{it['reject']}\n\n")
    return parsed["header"] + "".join(blocks) + parsed["sep"] + parsed["footer"]


# --------------------------------------------------------------------------
# Translation helpers
# --------------------------------------------------------------------------

LIST_PREFIX = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")
MAX_SEG_CHARS = 600


def split_segments(text):
    """Split text into (kind, value) pieces: ('keep', s) stays, ('tr', s) is translated."""
    out = []
    lines = text.split("\n")
    for li, line in enumerate(lines):
        if li:
            out.append(("keep", "\n"))
        if not re.search(r"[A-Za-z]", line):
            if line:
                out.append(("keep", line))
            continue
        m = LIST_PREFIX.match(line)
        if m:
            out.append(("keep", m.group(1)))
            line = line[m.end():]
        lead = re.match(r"^\s*", line).group(0)
        if lead:
            out.append(("keep", lead))
            line = line[len(lead):]
        if len(line) <= MAX_SEG_CHARS:
            out.append(("tr", line))
            continue
        sents = re.split(r"(?<=[.!?])\s+", line)
        buf = ""
        for s in sents:
            if buf and len(buf) + 1 + len(s) > MAX_SEG_CHARS:
                out.append(("tr", buf))
                out.append(("keep", " "))
                buf = s
            else:
                buf = f"{buf} {s}" if buf else s
        if buf:
            out.append(("tr", buf))
    return out


def join_segments(pieces, lang, cache):
    joiner_space = lang in ("ms", "en")
    res = []
    for kind, s in pieces:
        if kind == "keep":
            # CJK text does not use spaces between sentence chunks.
            res.append(s if (s != " " or joiner_space) else "")
        else:
            res.append(cache[s])
    return "".join(res)


def segments_to_translate(texts):
    segs = set()
    for t in texts:
        segs.update(s for k, s in split_segments(t) if k == "tr")
    return sorted(segs)


def translate_interaction(it, lang, caches):
    if lang == "en":
        return dict(it)
    return {k: join_segments(split_segments(it[k]), lang, caches[lang]) for k in ("query", "chosen", "reject")}


def version_interactions(interactions, version, caches):
    if version in LANGS:
        return [translate_interaction(it, version, caches) for it in interactions]
    return [translate_interaction(it, MIX_CYCLE[i % len(MIX_CYCLE)], caches)
            for i, it in enumerate(interactions)]


# --------------------------------------------------------------------------
# Language detection (rough, script + stopwords; good enough to tell which
# language the model answered in)
# --------------------------------------------------------------------------

MS_WORDS = {"yang", "dan", "untuk", "dengan", "tidak", "ini", "itu", "adalah", "dalam", "pengguna", "lebih", "anda"}
EN_WORDS = {"the", "and", "to", "of", "with", "is", "that", "for", "user", "prefers", "more", "this"}


def detect_lang(text):
    if not text:
        return "empty"
    hangul = len(re.findall(r"[가-힣]", text))
    cjk = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = hangul + cjk + latin or 1
    if hangul / total > 0.2:
        return "ko"
    if cjk / total > 0.2:
        return "zh"
    words = re.findall(r"[a-z]+", text.lower())
    ms = sum(w in MS_WORDS for w in words)
    en = sum(w in EN_WORDS for w in words)
    return "ms" if ms > en else "en"


# --------------------------------------------------------------------------
# Scoring (uses the repo's own parser and weighted sum)
# --------------------------------------------------------------------------

def repo_eval():
    sys.path.insert(0, os.path.join(REPO, "src"))
    import eval as pc_eval  # noqa: E402  (the repo's src/eval.py)
    return pc_eval


def parse_checklist(raw, pc_eval):
    obj = pc_eval.parse_output_safely(raw)
    if isinstance(obj, dict) and isinstance(obj.get("checklist"), list):
        items = [c for c in obj["checklist"] if isinstance(c, dict) and c.get("criterion")]
        return items or None
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):  # bare array
        items = [c for c in obj["results"] if isinstance(c, dict) and c.get("criterion")]
        return items or None
    return None


def plain_score(raw, pc_eval):
    obj = pc_eval.parse_output_safely(raw)
    if isinstance(obj, dict) and "score" in obj:
        try:
            return float(obj["score"])
        except (TypeError, ValueError):
            pass
    m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw or "")
    return float(m.group(1)) if m else None


def checklist_score(raw, checklist, pc_eval):
    return pc_eval.calculate_score(pc_eval.parse_output_safely(raw), checklist)


def bootstrap_ci(values, n=2000, seed=0):
    if not values:
        return (None, None)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_prepare(args):
    src = os.path.join(REPO, "data", "PRISM", "PRISM_GP_gen_test_user_input.json")
    data = json.load(open(src, encoding="utf-8"))
    rng = random.Random(SEED)
    picked = rng.sample(data, min(N_USERS, len(data)))
    users, pairs, iid = [], [], 0
    for u in picked:
        parsed = parse_gp_input(u["model_input"])
        assert build_gp_input(parsed, parsed["interactions"]) == u["model_input"], u["user_id"]
        users.append({"user_id": u["user_id"], **parsed})
        for t in u["test"]:
            iid += 1
            pairs.append({"instance_id": iid, "user_id": u["user_id"], "query": t["query"],
                          "chosen": t["chosen"], "reject": t["reject"]})
    os.makedirs(OUT, exist_ok=True)
    json.dump(users, open(p("users.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(pairs, open(p("pairs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    n_int = sum(len(u["interactions"]) for u in users)
    print(f"[prepare] {len(users)} users, {len(pairs)} test pairs, {n_int} past conversations")


def load_caches():
    caches = {"en": {}}
    for lang in NLLB_CODES:
        caches[lang] = {}
        for f in sorted(glob.glob(p(f"trans_{lang}.shard*.json"))):
            caches[lang].update(json.load(open(f, encoding="utf-8")))
    return caches


def stage_translate(args):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    users = json.load(open(p("users.json"), encoding="utf-8"))
    texts = [it[k] for u in users for it in u["interactions"] for k in ("query", "chosen", "reject")]
    segs = segments_to_translate(texts)
    mine = [s for i, s in enumerate(segs) if i % args.num_shards == args.shard]
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, torch_dtype=torch.float16).cuda().eval()
    for lang, code in NLLB_CODES.items():
        path = p(f"trans_{lang}.shard{args.shard}.json")
        cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        todo = sorted((s for s in mine if s not in cache), key=len)
        print(f"[translate] {lang}: {len(todo)} segments to go (shard {args.shard})", flush=True)
        bos = tok.convert_tokens_to_ids(code)
        for b in range(0, len(todo), 48):
            batch = todo[b:b + 48]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=1,
                                     max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 16))
            for s, o in zip(batch, tok.batch_decode(gen, skip_special_tokens=True)):
                cache[s] = o
            if (b // 48) % 20 == 0 or b + 48 >= len(todo):
                json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {lang}: {min(b + 48, len(todo))}/{len(todo)}", flush=True)
        json.dump(cache, open(path, "w", encoding="utf-8"), ensure_ascii=False)


def make_llm(model_id):
    from vllm import LLM
    return LLM(model=model_id, dtype="float16", max_model_len=MAX_MODEL_LEN,
               gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=SEED)


def run_chat(llm, items, key_fn, conv_fn, max_tokens, out_path, meta_fn):
    """Generate for items not yet in out_path, saving after each chunk."""
    from vllm import SamplingParams
    done = {r["key"] for r in read_jsonl(out_path)}
    todo = [it for it in items if key_fn(it) not in done]
    print(f"  {len(todo)} to generate ({len(items) - len(todo)} already saved)", flush=True)
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens, seed=SEED)
    for b in range(0, len(todo), CHUNK):
        batch = todo[b:b + CHUNK]
        outs = llm.chat([conv_fn(it) for it in batch], sp, use_tqdm=True)
        append_jsonl(out_path, [{"key": key_fn(it), **meta_fn(it), "output": o.outputs[0].text,
                                 "finish": o.outputs[0].finish_reason}
                                for it, o in zip(batch, outs)])
        print(f"  saved {min(b + CHUNK, len(todo))}/{len(todo)}", flush=True)


def msgs(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def stage_gp(args):
    users = json.load(open(p("users.json"), encoding="utf-8"))
    users = [u for i, u in enumerate(users) if i % args.num_shards == args.shard]
    caches = load_caches()
    llm = make_llm(JUDGE_MODEL)
    tok = llm.get_tokenizer()
    budget = MAX_MODEL_LEN - GP_MAX_TOKENS - 200
    items = []
    for u in users:
        vers = {v: version_interactions(u["interactions"], v, caches) for v in VERSIONS}
        # Keep the SAME past conversations in every version: drop the oldest
        # until the longest version fits the context window.
        start = 0
        while True:
            longest = max(len(tok.encode(build_gp_input(u, vers[v][start:], start + 1))) for v in VERSIONS)
            if longest <= budget or start >= len(u["interactions"]) - 1:
                break
            start += 1
        for v in VERSIONS:
            items.append({"user_id": u["user_id"], "version": v, "kept_from": start,
                          "n_kept": len(u["interactions"]) - start,
                          "prompt": build_gp_input(u, vers[v][start:], start + 1)})
    print(f"[gp] {len(items)} profiles (shard {args.shard})")
    run_chat(llm, items, lambda it: f"{it['user_id']}|{it['version']}",
             lambda it: msgs(GP_SYSTEM, it["prompt"]), GP_MAX_TOKENS,
             p(f"gp.shard{args.shard}.jsonl"),
             lambda it: {"user_id": it["user_id"], "version": it["version"],
                         "kept_from": it["kept_from"], "n_kept": it["n_kept"]})


def checklist_prompt():
    tmpl = open(os.path.join(REPO, "prompt", "Checklist_gen_train_prompt.txt"), encoding="utf-8").read()
    anchor = "\n\n[User's General Preference] (GP): {GP}"
    assert anchor in tmpl, "checklist prompt changed upstream"
    return tmpl.replace(anchor, "\n" + CHECKLIST_SCHEMA_HINT + anchor, 1)


def gp_table():
    return {(r["user_id"], r["version"]): r["output"].strip() for r in read_jsonl(p("gp.shard*.jsonl"))}


def stage_checklist(args):
    pairs = json.load(open(p("pairs.json"), encoding="utf-8"))
    gps = gp_table()
    tmpl = checklist_prompt()
    items = [{**pr, "version": v, "GP": gps[(pr["user_id"], v)]}
             for pr in pairs for v in VERSIONS if (pr["user_id"], v) in gps]
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    print(f"[checklist] {len(items)} checklists (shard {args.shard})")
    llm = make_llm(CHECKLIST_MODEL)
    run_chat(llm, items, lambda it: f"{it['instance_id']}|{it['version']}",
             lambda it: msgs(CHECKLIST_SYSTEM, fill(tmpl, GP=it["GP"], query=it["query"])),
             CHECKLIST_MAX_TOKENS, p(f"checklist.shard{args.shard}.jsonl"),
             lambda it: {"instance_id": it["instance_id"], "version": it["version"]})


def stage_judge(args):
    pc_eval = repo_eval()
    pairs = {pr["instance_id"]: pr for pr in json.load(open(p("pairs.json"), encoding="utf-8"))}
    gps = gp_table()
    ck_tmpl = open(os.path.join(REPO, "prompt", "Reward_infer_prompt.txt"), encoding="utf-8").read()
    plain, ck = [], []
    for (uid, v), gp in sorted(gps.items()):
        for pr in pairs.values():
            if pr["user_id"] != uid:
                continue
            for rt in ("chosen", "reject"):
                plain.append({**pr, "version": v, "GP": gp, "rt": rt})
    for r in read_jsonl(p("checklist.shard*.jsonl")):
        items = parse_checklist(r["output"], pc_eval)
        if not items:
            continue
        pr = pairs[r["instance_id"]]
        ck_str = "\n".join(f"criterion {j + 1}: {c['criterion']}" for j, c in enumerate(items))
        for rt in ("chosen", "reject"):
            ck.append({**pr, "version": r["version"], "GP": gps[(pr["user_id"], r["version"])],
                       "rt": rt, "ck_str": ck_str})
    plain = [it for i, it in enumerate(plain) if i % args.num_shards == args.shard]
    ck = [it for i, it in enumerate(ck) if i % args.num_shards == args.shard]
    key = lambda it: f"{it['instance_id']}|{it['version']}|{it['rt']}"  # noqa: E731
    meta = lambda it: {"instance_id": it["instance_id"], "version": it["version"], "rt": it["rt"]}  # noqa: E731
    llm = make_llm(JUDGE_MODEL)
    print(f"[judge] plain: {len(plain)} calls (shard {args.shard})")
    run_chat(llm, plain, key,
             lambda it: msgs(JUDGE_SYSTEM, fill(PLAIN_JUDGE_PROMPT, GP=it["GP"], query=it["query"],
                                                response=it[it["rt"]])),
             JUDGE_PLAIN_MAX_TOKENS, p(f"judge_plain.shard{args.shard}.jsonl"), meta)
    print(f"[judge] checklist: {len(ck)} calls (shard {args.shard})")
    run_chat(llm, ck, key,
             lambda it: msgs(JUDGE_SYSTEM, fill(ck_tmpl, GP=it["GP"], query=it["query"],
                                                checklist=it["ck_str"], response=it[it["rt"]])),
             JUDGE_CK_MAX_TOKENS, p(f"judge_ck.shard{args.shard}.jsonl"), meta)


def pair_outcomes(rows, score_fn):
    """{(instance_id, version): 1/0} for pairs where both scores parsed; ties count as wrong."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["instance_id"], r["version"])][r["rt"]] = r
    res, failed = {}, Counter()
    for k, d in by.items():
        if "chosen" not in d or "reject" not in d:
            continue
        sa, sb = score_fn(d["chosen"]), score_fn(d["reject"])
        if sa is None or sb is None:
            failed[k[1]] += 1
            continue
        res[k] = {"correct": int(sa > sb), "tie": int(sa == sb)}
    return res, failed


def summarize(outcomes, failed, n_expected):
    rows = {}
    en = {k[0]: o["correct"] for k, o in outcomes.items() if k[1] == "en"}
    for v in VERSIONS:
        vals = [o["correct"] for k, o in outcomes.items() if k[1] == v]
        ties = sum(o["tie"] for k, o in outcomes.items() if k[1] == v)
        acc = sum(vals) / len(vals) if vals else None
        lo, hi = bootstrap_ci(vals)
        diffs = [o["correct"] - en[k[0]] for k, o in outcomes.items() if k[1] == v and k[0] in en]
        dlo, dhi = bootstrap_ci(diffs) if v != "en" else (None, None)
        rows[v] = {"acc": acc, "ci": (lo, hi), "n": len(vals), "ties": ties,
                   "parse_fail": failed.get(v, 0), "expected": n_expected,
                   "diff_vs_en": (sum(diffs) / len(diffs)) if (diffs and v != "en") else None,
                   "diff_ci": (dlo, dhi)}
    return rows


def pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def stage_eval(args):
    pc_eval = repo_eval()
    pairs = json.load(open(p("pairs.json"), encoding="utf-8"))
    ck_rows = read_jsonl(p("checklist.shard*.jsonl"))
    ck_items = {(r["instance_id"], r["version"]): parse_checklist(r["output"], pc_eval) for r in ck_rows}

    plain_out, plain_fail = pair_outcomes(read_jsonl(p("judge_plain.shard*.jsonl")),
                                          lambda r: plain_score(r["output"], pc_eval))

    def ck_fn(r):
        items = ck_items.get((r["instance_id"], r["version"]))
        return checklist_score(r["output"], items, pc_eval) if items else None

    ck_out, ck_fail = pair_outcomes(read_jsonl(p("judge_ck.shard*.jsonl")), ck_fn)
    for (iid, v), items in ck_items.items():  # checklist itself failed to parse
        if not items:
            ck_fail[v] += 1

    names = {"en": "English", "ko": "Korean", "zh": "Chinese", "ms": "Malay",
             "mixed": "Mixed (EN/ZH/MS/KO by conversation)"}
    S = {"plain": summarize(plain_out, plain_fail, len(pairs)),
         "checklist": summarize(ck_out, ck_fail, len(pairs))}

    gp_rows = read_jsonl(p("gp.shard*.jsonl"))
    gp_lang = {v: Counter(detect_lang(r["output"]) for r in gp_rows if r["version"] == v) for v in VERSIONS}
    gp_cut = {v: sum(r["finish"] == "length" for r in gp_rows if r["version"] == v) for v in VERSIONS}

    lines = ["| History language | Plain judge acc % [95% CI] | vs EN (pts) | Judge + checklist acc % [95% CI] | vs EN (pts) | Pairs scored (plain / checklist) | Profile written in |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for v in VERSIONS:
        a, b = S["plain"][v], S["checklist"][v]

        def cell(s):
            return f"{pct(s['acc'])} [{pct(s['ci'][0])}–{pct(s['ci'][1])}]" if s["acc"] is not None else ""

        def dcell(s):
            if s["diff_vs_en"] is None:
                return "—"
            return f"{100 * s['diff_vs_en']:+.1f} [{100 * s['diff_ci'][0]:+.1f}, {100 * s['diff_ci'][1]:+.1f}]"

        langs = ", ".join(f"{k} {c}" for k, c in gp_lang[v].most_common())
        lines.append(f"| {names[v]} | {cell(a)} | {dcell(a)} | {cell(b)} | {dcell(b)} | "
                     f"{a['n']} / {b['n']} of {a['expected']} | {langs} |")
    notes = [
        "",
        "Ties count as wrong (as in the repo's eval.py). Pairs whose outputs could not be parsed are left out;",
        "see 'Pairs scored'. A 'vs EN' interval that includes 0 means the difference could be noise.",
        f"Plain-judge ties per version: " + ", ".join(f"{v} {S['plain'][v]['ties']}" for v in VERSIONS),
        f"Profiles cut off at {GP_MAX_TOKENS} tokens: " + ", ".join(f"{v} {gp_cut[v]}" for v in VERSIONS),
    ]
    md = "\n".join(lines + notes)
    open(p("results.md"), "w", encoding="utf-8").write(md + "\n")
    json.dump(S, open(p("results.json"), "w"), indent=2, default=str)
    print(md)


def stage_handcheck(args):
    """Side-by-side CSV of 20 random past conversations per language for checking by hand."""
    import csv
    users = json.load(open(p("users.json"), encoding="utf-8"))
    caches = load_caches()
    rng = random.Random(SEED + 1)
    flat = [(u["user_id"], i, it) for u in users for i, it in enumerate(u["interactions"])]
    with open(p("handcheck.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "user_id", "conversation", "field", "english", "translation",
                    "meaning_error (y/n)", "note"])
        for lang in NLLB_CODES:
            for uid, i, it in rng.sample(flat, min(20, len(flat))):
                tr = translate_interaction(it, lang, caches)
                for k in ("query", "chosen"):
                    w.writerow([lang, uid, i + 1, k, it[k], tr[k], "", ""])
    print(f"[handcheck] wrote {p('handcheck.csv')}")


STAGES = {"prepare": stage_prepare, "translate": stage_translate, "gp": stage_gp,
          "checklist": stage_checklist, "judge": stage_judge, "eval": stage_eval,
          "handcheck": stage_handcheck}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    STAGES[ap.parse_args().stage](ap.parse_args())
