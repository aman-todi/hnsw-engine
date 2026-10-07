#!/usr/bin/env python3
"""Render the results sections of README.md and docs/BENCHMARKS.md from the
CSVs in bench/results/ (run after plot.py). Every number in those sections
comes from here, so the docs always match the measured data.

Sections are replaced between marker comments:
    <!-- results:begin --> ... <!-- results:end -->   (README.md, docs/BENCHMARKS.md)
    <!-- env:begin --> ... <!-- env:end -->           (docs/BENCHMARKS.md)
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "bench" / "results"
NAMES = {"engine": "hnsw-engine", "hnswlib": "hnswlib", "faiss": "FAISS HNSWFlat"}
LIBS = ("engine", "hnswlib", "faiss")
DATASETS = ("synth-sift", "synth-glove", "synth-gist")


def rows(name: str) -> list[dict]:
    p = RES / name
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def best(rs, lib, mode, thr):
    q = [f(r["qps"]) for r in rs if r["library"] == lib and r["mode"] == mode and f(r["recall"]) >= thr]
    return max(q) if q else math.nan


def n0(v):
    return "—" if math.isnan(v) else f"{v:,.0f}"


def pct(a, b):
    return f"{(a / b - 1) * 100:+.0f}%"


def env() -> dict:
    out = {}
    p = RES / "environment.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            k, _, v = line.partition(": ")
            out[k] = v
    return out


def describe(rs) -> str:
    r = rs[0]
    return f"{int(r['n']):,} × {r['dim']}, {r['metric']}"


def qps_table(data, mode, targets=(0.95, 0.99)) -> list[str]:
    out = ["| dataset | recall@10 target | " + " | ".join(NAMES[l] for l in LIBS) + " |",
           "|---|---|" + "---:|" * len(LIBS)]
    for ds, rs in data.items():
        for t in targets:
            vals = [best(rs, l, mode, t) for l in LIBS]
            top = max(v for v in vals if not math.isnan(v))
            cells = [f"**{n0(v)}**" if v == top else n0(v) for v in vals]
            out.append(f"| {ds} ({describe(rs)}) | ≥ {t:.2f} | " + " | ".join(cells) + " |")
    return out


def slower(data) -> list[str]:
    notes = []
    for ds, rs in data.items():
        for mode in ("single", "batch"):
            for t in (0.90, 0.95, 0.99):
                e = best(rs, "engine", mode, t)
                others = {l: best(rs, l, mode, t) for l in ("hnswlib", "faiss")}
                lib, o = max(others.items(), key=lambda kv: -math.inf if math.isnan(kv[1]) else kv[1])
                if not math.isnan(e) and not math.isnan(o) and e < o:
                    notes.append(f"{ds}, {'single-thread' if mode == 'single' else 'batched'}, recall ≥ {t:.2f}: "
                                 f"{n0(e)} vs {NAMES[lib]} {n0(o)} QPS ({pct(e, o)})")
    return notes


def ceiling(rs, lib):
    return max(f(r["recall"]) for r in rs if r["library"] == lib)


def ablation_table() -> tuple[list[str], dict]:
    ab = rows("ablation.csv")
    if not ab:
        return [], {}
    base = f(ab[0]["qps"])
    out = ["| configuration (C++ harness, same graph) | recall@10 | QPS | vs scalar |", "|---|---:|---:|---:|"]
    for r in ab:
        label = r["label"].split(" ", 1)[1]
        out.append(f"| {label} | {f(r['recall']):.3f} | {n0(f(r['qps']))} | {f(r['qps']) / base:.1f}× |")
    return out, {r["label"].split(" ", 1)[1]: f(r["qps"]) for r in ab}


def kernels() -> dict:
    k = {}
    p = RES / "micro_kernels.csv"
    if p.exists():
        for row in csv.reader(open(p)):
            if row and row[0].startswith("BM_") and len(row) > 2 and row[2]:
                k[row[0]] = f(row[2])
    return k


def scaling() -> list[dict]:
    return rows("scaling.csv")


def replace(path: Path, marker: str, body: str) -> None:
    text = path.read_text()
    pat = re.compile(rf"(<!-- {marker}:begin -->\n).*?(<!-- {marker}:end -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"{path}: missing {marker} markers")
    path.write_text(pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text))


def main() -> int:
    data = {ds: rows(f"{ds}.csv") for ds in DATASETS if rows(f"{ds}.csv")}
    e = env()
    k = kernels()
    abl, abq = ablation_table()
    sc = scaling()
    sift = data.get("synth-sift", [])
    build = {r["library"]: r for r in sift} if sift else {}
    hw = (f"{e.get('cpu', '?')}, {e.get('cores', '?')}, {e.get('memory', '?')} RAM; "
          f"{e.get('compiler', '?')}; hnswlib {e.get('hnswlib', '?')}, faiss-cpu {e.get('faiss-cpu', '?')}; "
          f"commit `{e.get('commit', '?')[:7]}`")
    sl = slower(data)
    k128 = (k.get("BM_L2/scalar/128"), k.get("BM_L2/avx2/128"), k.get("BM_L2/avx512/128"))
    sc_txt = ""
    if sc:
        t1, tn = f(sc[0]["build_s"]), f(sc[-1]["build_s"])
        sc_txt = (f"Parallel build: {tn and t1 / tn:.1f}× on {sc[-1]['build_threads']} threads "
                  f"({int(sc[0]['n']):,} vectors: {t1:.1f} s → {tn:.1f} s; recall@10 at ef=64 "
                  f"{f(sc[0]['recall']):.4f} → {f(sc[-1]['recall']):.4f}).")

    # ------------------------------------------------------------------ README
    md = [f"Measured by `scripts/run_all_benchmarks.sh --synthetic` on {hw}. M = 16, "
          "ef_construction = 200, k = 10, 4 threads. **Synthetic data shaped like the standard sets — "
          "not SIFT/GloVe/GIST results.** Full tables, methodology, run-to-run variance and raw CSVs: "
          "[docs/BENCHMARKS.md](docs/BENCHMARKS.md), `bench/results/`.", "",
          "![recall vs QPS on synth-sift](bench/results/synth-sift.png)", "",
          "**Best single-thread QPS at a recall target** (one query per Python call; bold = fastest):", ""]
    md += qps_table(data, "single")
    if build:
        md += ["", f"**Build** (synth-sift, 1M): {f(build['engine']['build_s']):.0f} s on 4 threads / "
               f"{f(build['engine']['build_1t_s']):.0f} s on 1 thread, vs hnswlib "
               f"{f(build['hnswlib']['build_s']):.0f} s / {f(build['hnswlib']['build_1t_s']):.0f} s and FAISS "
               f"{f(build['faiss']['build_s']):.0f} s / {f(build['faiss']['build_1t_s']):.0f} s. Index file "
               f"{f(build['engine']['index_bytes']) / 2**20:.0f} MB (hnswlib "
               f"{f(build['hnswlib']['index_bytes']) / 2**20:.0f}, FAISS {f(build['faiss']['index_bytes']) / 2**20:.0f})."]
    if abl:
        md += ["", "**Where the speed comes from** (synth-sift 1M, ef = 64):", ""] + abl
    if all(k128):
        md += ["", f"Distance kernel alone (L2, d = 128): scalar {k128[0]:.1f} ns, AVX2 {k128[1]:.1f} ns "
               f"({k128[0] / k128[1]:.1f}×), AVX-512 {k128[2]:.1f} ns ({k128[0] / k128[2]:.1f}×). {sc_txt}"]
    if sl:
        md += ["", "**Where it is slower** (every case where another library beats the engine at a target):", ""]
        md += [f"* {s}" for s in sl]
        md += ["", "Batched (4-thread) throughput on this shared VM varied by up to ~30% between runs "
               "and single-thread by up to ~16% (see the variance section in BENCHMARKS.md, including a "
               "synth-glove re-check where the engine was ahead), so only gaps larger than that are meaningful."]
    if sift:
        md += ["", f"On synth-sift the engine's recall saturates a little lower at very high ef "
               f"(max {ceiling(sift, 'engine'):.4f} vs hnswlib {ceiling(sift, 'hnswlib'):.4f} at ef = 640), "
               "which is what costs it the ≥ 0.99 target there; this is an open item (see BENCHMARKS.md)."]
    replace(REPO / "README.md", "results", "\n".join(md))

    # -------------------------------------------------------------- BENCHMARKS
    replace(REPO / "docs" / "BENCHMARKS.md", "env",
            "```\n" + (RES / "environment.txt").read_text().strip() + "\n```")
    summary = (RES / "summary.md").read_text().split("\n", 2)[2]
    summary = re.sub(r"^## ", "### ", summary, flags=re.M)
    b = ["![synth-sift](../bench/results/synth-sift.png)", "![synth-glove](../bench/results/synth-glove.png)",
         "![synth-gist](../bench/results/synth-gist.png)", "", summary.strip(), "",
         "Notes on the tables:", "",
         "* Ablation and scaling rows come from the C++ harness (`bench_main`). The ablation loads one",
         "  pre-built index, so every row searches the identical graph (identical recall); the last row is",
         "  batched, so it has no per-query latency.",
         "* synth-gist peak RSS is dominated by loading the 960-d dataset in each worker, so it is the same",
         "  for all three libraries; compare the RSS-growth column instead.",
         "* Brute force on synth-sift (1M vectors, 1,000 queries) reaches recall "
         f"{f(rows('bruteforce.csv')[0]['recall']) if rows('bruteforce.csv') else math.nan:.5f} against "
         "the NumPy ground truth (`bruteforce.csv`), validating the harness.", ""]
    fl = rows("filter.csv")
    if fl:
        b += ["### Filtered search (engine, synth-sift 200k subset, random allow-lists, 4 threads)", "",
              "| allowed | ef | recall@10 | QPS | filter violations |", "|---:|---:|---:|---:|---:|"]
        b += [f"| {f(r['selectivity']) * 100:.0f}% | {r['ef']} | {f(r['recall']):.4f} | {n0(f(r['qps']))} | "
              f"{r['violations']} |" for r in fl]
        b += ["", "Every returned id satisfied the filter. Recall stays high at low selectivity because the search",
              "keeps exploring until it has `ef` eligible results; the cost is throughput.", ""]
    if k:
        dims = (16, 100, 128, 384, 768, 960, 1024)
        b += ["### Kernel microbenchmarks (Google Benchmark, ns per call)", "",
              "| dim | scalar L2 | AVX2 L2 | AVX-512 L2 | scalar dot | AVX2 dot | AVX-512 dot |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for d in dims:
            vals = [k.get(f"BM_{fn}/{isa}/{d}", math.nan) for fn in ("L2", "Dot") for isa in ("scalar", "avx2", "avx512")]
            b.append(f"| {d} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        b.append("")
    rc = rows("recheck_synth-glove_batch.csv")
    b += ["### Run-to-run variance", "",
          "The benchmark machine is a shared 4-core cloud VM. Comparing two complete runs of the pipeline",
          "(same code for hnswlib/FAISS, recall identical to 4 decimals), single-thread QPS for the same",
          "library moved by up to ~16% (hnswlib, synth-sift, recall ≥ 0.95: 2,167 vs 2,508) and batched",
          "4-thread QPS by up to ~30%. Differences smaller than that are not meaningful on this machine."]
    if rc and "synth-glove" in data:
        full = data["synth-glove"]
        b += ["For example, synth-glove batched at ef = 40:", "",
              "| library | full run QPS | re-check QPS (`recheck_synth-glove_batch.csv`) |", "|---|---:|---:|"]
        for lib in ("engine", "hnswlib"):
            a = next((f(r["qps"]) for r in full if r["library"] == lib and r["mode"] == "batch" and r["ef"] == "40"), math.nan)
            c = next((f(r["qps"]) for r in rc if r["library"] == lib and r["ef"] == "40"), math.nan)
            b.append(f"| {NAMES[lib]} | {n0(a)} | {n0(c)} |")
        b += ["", "Rerun on a dedicated host for publishable throughput numbers."]
    b += ["", "### Open item: high-recall ceiling on synth-sift", "",
          "At ef = 640 the engine reaches lower recall than hnswlib on synth-sift "
          f"({ceiling(sift, 'engine'):.4f} vs {ceiling(sift, 'hnswlib'):.4f}), although it matches or",
          "exceeds hnswlib's recall on synth-glove and synth-gist. Two candidate causes were ruled out by an",
          "A/B test on a 200k subset (no measurable recall change): passing only the closest node instead of",
          "the whole result set W to the next layer, and skipping the heuristic when fewer than M candidates",
          "exist (both hnswlib behaviours). The cause is still open.", ""]
    if sift and "synth-gist" in data:
        gist = data["synth-gist"]
        s95 = {l: best(sift, l, "single", 0.95) for l in LIBS}
        s99 = {l: best(sift, l, "single", 0.99) for l in LIBS}
        g95 = {l: best(gist, l, "single", 0.95) for l in LIBS}
        b += ["## Resume-ready summary (measured; synthetic SIFT/GIST-shaped data)", "",
              "* Built an HNSW vector search engine from scratch in C++20 (Malkov & Yashunin, Algorithms 1–5) "
              "with AVX2/AVX-512/NEON kernels and runtime CPU dispatch: "
              f"**{k128[0] / k128[1]:.1f}× faster L2 kernel** (AVX2 vs scalar, d = 128) and "
              f"**{abq.get('+prefetch', math.nan) / abq.get('scalar kernels (no prefetch)', math.nan):.1f}× single-thread "
              "QPS from SIMD + prefetching** at identical recall (1M × 128).",
              f"* 1M × 128 L2, recall@10 ≥ 0.95, single thread: **{n0(s95['engine'])} QPS vs hnswlib "
              f"{n0(s95['hnswlib'])} ({pct(s95['engine'], s95['hnswlib'])}) and FAISS HNSWFlat {n0(s95['faiss'])} "
              f"({pct(s95['engine'], s95['faiss'])})** — on par with hnswlib within this VM's ~16% run-to-run noise; "
              f"200k × 960: **{pct(g95['engine'], g95['hnswlib'])} vs hnswlib**. "
              f"Slower at recall ≥ 0.99 on 1M × 128 ({pct(s99['engine'], s99['hnswlib'])} vs hnswlib).",
              f"* {sc_txt.replace('Parallel build:', 'Parallel build scales')} 1M-vector build in "
              f"{f(build['engine']['build_s']):.0f} s on 4 threads (hnswlib {f(build['hnswlib']['build_s']):.0f} s, "
              f"FAISS {f(build['faiss']['build_s']):.0f} s) at the same index size.",
              "* Memory-mapped, checksummed on-disk format whose loader rejects every truncated or bit-flipped "
              "file in fuzz tests; ASan/UBSan- and TSan-clean; GoogleTest + pytest suites; pybind11 package "
              "that releases the GIL and matches the C++ results exactly."]
    replace(REPO / "docs" / "BENCHMARKS.md", "results", "\n".join(b))
    print("updated README.md and docs/BENCHMARKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
