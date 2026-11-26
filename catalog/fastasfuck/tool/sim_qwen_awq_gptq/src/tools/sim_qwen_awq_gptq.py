#!/usr/bin/env python3
"""
Held-out RX 7900 XTX simulator evaluating ZERO-SPEED-PENALTY AWQ smoothing +
full-rank ~61k-token GPTQ on Qwen 3.8 Flash Next (Thinking-Cap) at the EXACT
1_Current_Prod binary .faf layout:
  exp_gu=q4g128, exp_down=q4g128, gdn_in=q4g128,
  dense (gdn_out, attn_qkv, attn_o, sh_gu, sh_down, ple_kv)=q6g64,
  hc_down=q8g64, hc_up=q8g64, lm_head=q8g64, embed=bf16.

Zero runtime overhead, zero HIP kernel edits:
  1. gdn_in AWQ: folds 1/s into attn_hyper_connection.hc_norm.weight (used only by hc_mix_up),
     keeping mix_down/inject pre-folded with the ORIGINAL hc_norm.
  2. exp_gu/sh_gu/router AWQ: folds 1/s into mlp_hyper_connection.hc_norm.weight,
     scaling router (BF16), sh_gu (RTN q6g64), and exp_gu (q4g128) columns by s.
  3. exp_down AWQ: folds 1/s_{e} into up rows (640..1279) of exp_gu[e] (lossless in W_up
     thanks to independent per-row-group scales) and multiplies exp_down[e] columns by s_{e}.
  4. Never touches q6g64 or q8g64 with weighted clipping or GPTQ (pure RTN q6g64/q8g64).
"""

import os
os.environ.setdefault("ROCP_HSA_INTERCEPT", "0")
os.environ.setdefault("ROCP_TOOLS_LIB", "")
os.environ.setdefault("HSA_TOOLS_LIB", "")

import argparse
import concurrent.futures
import glob
import hashlib
import json
import math
import statistics
import sys
import time
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_qwen_quant as sq
import sim_qwen_calib as sc


ALPHAS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)

SENTINEL_PHRASES = (
    "Riemann-Lebesgue",
    "Riemann–Lebesgue",
    "execute_database_drop",
    "kernel-bypass HSA queue",
    "max_concurrent_tasks",
    "CycleDetectedError",
    "do_page_fault",
    "Deficit Round Robin",
    "async execute() method",
    "direct reclaim loops",
)


# ---------------------------------------------------------------------------
# 1. Diverse Strictly Disjoint ~61k-Token Calibration Corpus
# ---------------------------------------------------------------------------

def _read_arrow_table(path: str) -> pa.Table:
    with open(path, "rb") as f:
        try:
            return ipc.open_stream(f).read_all()
        except Exception:
            f.seek(0)
            return ipc.open_file(f).read_all()


def _verify_text_disjoint(text: str, eval_doc: str, label: str) -> None:
    for s in SENTINEL_PHRASES:
        assert s not in text, f"Sentinel phrase {s!r} leaked into calibration source ({label})"
    # Sliding 50-char window check against held-out CALIB_DOC
    for i in range(0, max(0, len(text) - 50), 25):
        seg = text[i : i + 50]
        if "<|im_" in seg or not seg.strip():
            continue
        assert seg not in eval_doc, f"Calibration overlap with CALIB_DOC ({label}): {seg!r}"


def _build_domain_chunks(tok, texts, target_count: int, seq_len: int, eval_doc: str, label: str):
    """Concatenates texts within a single domain and slices exact [seq_len]-token sequences."""
    all_ids = []
    sep_ids = tok.encode("\n")
    for idx, t in enumerate(texts):
        _verify_text_disjoint(t, eval_doc, f"{label}#{idx}")
        ids = tok.encode(t)
        all_ids.extend(ids)
        all_ids.extend(sep_ids)
        if len(all_ids) >= target_count * seq_len:
            break
    seqs = []
    pos = 0
    while len(seqs) < target_count and pos + seq_len <= len(all_ids):
        seqs.append(all_ids[pos : pos + seq_len])
        pos += seq_len
    return seqs


def _gather_swebench_texts() -> list[str]:
    texts = []
    arrow_paths = [
        "/home/cheblan/.cache/huggingface/datasets/princeton-nlp___swe-bench_lite/default/0.0.0/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2/swe-bench_lite-test.arrow",
        "/home/cheblan/.cache/huggingface/datasets/ScaleAI___swe-bench_pro/default/0.0.0/2d52cb3df914a3fcf80c7f66738b3a88ae37fc50/swe-bench_pro-test.arrow",
    ]
    for p in arrow_paths:
        if not os.path.exists(p):
            continue
        tbl = _read_arrow_table(p)
        cols = set(tbl.column_names)
        for i in range(0, min(tbl.num_rows, 90), 3):
            repo = str(tbl["repo"][i].as_py() or "")
            prob = str(tbl["problem_statement"][i].as_py() or "")[:2400] if "problem_statement" in cols else str(tbl["requirements"][i].as_py() or "")[:2400]
            patch = str(tbl["patch"][i].as_py() or "")[:3200]
            tpatch = str(tbl["test_patch"][i].as_py() or "")[:1600]
            if not prob or not patch:
                continue
            doc = (
                "<|im_start|>system\nYou are an autonomous software engineer debugging open-source repositories.<|im_end|>\n"
                f"<|im_start|>user\nRepository: `{repo}`\nIssue:\n{prob}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n"
                f"Inspecting the reported failure in `{repo}` and isolating the minimal backwards-compatible patch.\n"
                "</think>\n"
                f"```diff\n{patch}\n```\nTests verifying the change:\n```diff\n{tpatch}\n```<|im_end|>"
            )
            texts.append(doc)
    return texts


def _gather_code_texts(workspace_root: str) -> list[str]:
    texts = []
    subdirs = ["strixite", "DeepInfer", "halogen-flash-server"]
    exts = (".cpp", ".hpp", ".hip", ".py", ".rs")
    candidates = []
    for sd in subdirs:
        base = os.path.join(workspace_root, sd)
        if not os.path.exists(base):
            continue
        for dp, dn, fns in os.walk(base):
            dn[:] = sorted([d for d in dn if d not in (".git", "build", "build-rel", "__pycache__", ".venv", "target")])
            for fn in sorted(fns):
                if not fn.endswith(exts):
                    continue
                if fn.startswith(("sim_qwen", "sim_ling", "bench_")) or fn == "baseline_outputs.json":
                    continue
                p = os.path.join(dp, fn)
                sz = os.path.getsize(p)
                if 2000 <= sz <= 120_000:
                    candidates.append(p)
    # Stride evenly across repos for diversity
    step = max(1, len(candidates) // 45)
    for p in candidates[::step]:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        if any(s in raw for s in SENTINEL_PHRASES):
            continue
        sl = raw[1000:9000] if len(raw) > 2500 else raw[:7000]
        rel = os.path.relpath(p, workspace_root)
        ext = os.path.splitext(p)[1].lstrip(".")
        texts.append(
            "<|im_start|>user\nReview this systems module and explain its invariants:\n"
            f"File: `{rel}`\n```{ext}\n{sl}\n```<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nLet's trace ownership, synchronization, and boundary checks carefully.\n</think>\n"
            f"The module `{rel}` enforces bounds and deterministic state transitions across its hot path.<|im_end|>"
        )
    return texts


def _gather_docs_texts(workspace_root: str) -> list[str]:
    texts = []
    for p in sorted(glob.glob(os.path.join(workspace_root, "*.md")) + glob.glob(os.path.join(workspace_root, "strixite", "**", "*.md"), recursive=True)):
        if os.path.basename(p) == "AGENTS.md":
            continue
        sz = os.path.getsize(p)
        if not (1500 <= sz <= 100_000):
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        if any(s in raw for s in SENTINEL_PHRASES):
            continue
        texts.append(raw[:9000])
    # Add diverse neutral technical prose from /usr/share/doc (README/changelog/docs)
    usr_docs = []
    for pat in ("/usr/share/doc/*/README*", "/usr/share/doc/*/copyright"):
        usr_docs.extend(sorted(glob.glob(pat)))
    for p in usr_docs[::max(1, len(usr_docs) // 30)]:
        if p.endswith(".gz") or not os.path.isfile(p):
            continue
        if not (1200 <= os.path.getsize(p) <= 40_000):
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        if len(raw.strip()) > 600 and not any(s in raw for s in SENTINEL_PHRASES):
            texts.append(raw[:6000])
    return texts


def _russian_tech_texts() -> list[str]:
    return [
        (
            "<|im_start|>system\nТы — ведущий архитектор высоконагруженных систем хранения данных.<|im_end|>\n"
            "<|im_start|>user\nСравни журнально-структурированные деревья (LSM-tree) и классические B+ деревья "
            "при смешанной нагрузке запись/чтение на современных NVMe-накопителях с интерфейсом io_uring.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nРаскроем физику амплификации записи, уплотнения уровней (compaction) и "
            "асинхронного опроса очередей завершения без межъядерных блокировок.\n</think>\n"
            "### 1. Сравнительный анализ амплификации ввода-вывода\n"
            "Деревья LSM преобразуют случайные модификации в последовательную запись сегментов SSTable через буфер "
            "в оперативной памяти (MemTable) и журнал предзаписи (WAL). При уровневом уплотнении (leveled compaction) "
            "типичная амплификация записи составляет 10–25×, однако пропускная способность вставок на порядок выше, чем "
            "у B+ деревьев, где изменение одного ключа требует перезаписи целой страницы 4–16 КиБ на месте.\n\n"
            "### 2. Интеграция с `io_uring` и модель SQPOLL\n"
            "При использовании выделенных пар колец Submission Queue / Completion Queue для каждого рабочего потока "
            "устраняется конкуренция за глобальные мьютексы файловой системы. Режим `O_DIRECT` совместно с заранее "
            "зарегистрированными DMA-буферами (`IORING_REGISTER_BUFFERS`) исключает лишнее копирование в page cache ядра "
            "и снижает 99-й процентиль задержки чтения ниже 180 микросекунд.\n\n"
            "### 3. Фильтры Блума и кеширование блоков\n"
            "Для минимизации числа обращений к уровням L1–L4 применяются секционированные фильтры Блума (10 бит на ключ, "
            "вероятность ложного срабатывания ~0.8%) и закрепление индексных блоков в L2-кеше пользовательского пространства.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nРазбери устройство алгебраического многосеточного предобусловливателя (AMG) и метода "
            "сопряжённых градиентов (PCG) для разреженных СЛАУ в задачах вычислительной гидродинамики.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nОпишем разделение высокочастотных и гладких компонент невязки, "
            "построение операторов пролонгации и ограничения по Галеркину P^T A P.\n</think>\n"
            "Для симметричной положительно определённой матрицы $A \\in \\mathbb{R}^{n \\times n}$ скорость сходимости "
            "классического метода PCG ограничивается квадратным корнем из числа обусловленности $\\sqrt{\\kappa(M^{-1} A)}$. "
            "Гладильщики типа Якоби или симметричного Гаусса–Зейделя эффективно подавляют лишь высокочастотные колебания "
            "ошибки на текущей сетке; длинноволновые моды затухают критически медленно.\n\n"
            "Алгебраическая многосеточная иерархия оценивает сильные связи графа по порогу $|a_{ij}| \\ge \\theta \\max_{k \\neq i} |a_{ik}|$, "
            "выбирает грубые узлы (C-точки) и формирует грубосеточный оператор $A_c = R A P$. Один V-цикл AMG удерживает "
            "число итераций PCG практически независимым от измельчения расчётной сетки — в диапазоне 8–14 итераций.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nОбъясни работу алгоритмов контроля перегрузки BBRv2 и DCTCP в датацентровых сетях RDMA.<|im_end|>\n"
            "<|im_start|>assistant\nВ отличие от классического Cubic, реагирующего исключительно на потерю пакетов после "
            "переполнения буферов коммутатора (bufferbloat), протокол BBR оценивает произведение пропускной способности "
            "узкого места на минимальное время кругового обхода ($BtlBw \\times RTprop$) и удерживает объём данных в полёте "
            "около 1–2 BDP. Внутри ЦОД алгоритм DCTCP использует однобитную маркировку ECN на пороге $K_{ECN}$, "
            "вычисляя сглаженную долю отмеченных пакетов $\\alpha \\leftarrow (1 - g)\\alpha + g \\cdot F$, что плавно "
            "подстраивает окно передачи $cwnd$ без пикового обрушения загрузки канала.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nКак устроен SSA-граф в оптимизирующем компиляторе (LLVM) при векторизации циклов и "
            "анализе псевдонимов указателей (Alias Analysis)?<|im_end|>\n"
            "<|im_start|>assistant\nСтатическая однократная присваиваемость (SSA) гарантирует, что каждая виртуальная "
            "переменная определяется ровно в одной точке, а слияние потоков управления моделируется через фи-узлы (`phi`). "
            "Проход `LoopVectorize` опирается на `ScalarEvolution` для доказательства отсутствия переполнения индексных "
            "выражений и использует `ScopedNoAliasAA` либо генерацию runtime-проверок непересечения диапазонов указателей "
            "`[base_a, base_a + N*stride)` перед входом в векторное тело цикла.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nОпиши механизм планирования задач work-stealing в рантаймах уровня Tokio и Cilk-5.<|im_end|>\n"
            "<|im_start|>assistant\nКаждый рабочий поток поддерживает локальный двусторонний дек (Chase-Lev deque): "
            "собственные новые задачи помещаются и извлекаются со стороны `bottom` без атомарных CAS-операций на горячем "
            "пути (FIFO/LIFO для максимальной локальности кеша L1d). Когда локальный дек пустеет, поток-вор выбирает "
            "случайную жертву и забирает самую крупную задачу с конца `top`, используя одно атомарное сравнение с обменом "
            "и барьер `seq_cst` лишь при граничном конфликте из-за последнего элемента.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nПоясни принцип работы транзакционной памяти (MVCC + SSI) и предотвращение аномалии write skew в распределённых СУБД.<|im_end|>\n"
            "<|im_start|>assistant\nСериализуемая изоляция снимков (SSI) отслеживает антизависимости `rw-conflict` между "
            "конкурирующими транзакциями, читающими устаревшие версии ключей и изменяющими непересекающиеся строки. "
            "При обнаружении двух смежных рёбер `rw` в графе сериализации (опасной структуры) координатор принудительно "
            "откатывает одну из транзакций, гарантируя эквивалентность строго последовательному выполнению без глобальных блокировок чтения.\n<|im_end|>"
        ),
    ]


def _math_algo_texts() -> list[str]:
    return [
        (
            "<|im_start|>system\nYou are a formal mathematician and numerical analyst.<|im_end|>\n"
            "<|im_start|>user\nProve the Finite-Dimensional Spectral Theorem for real symmetric matrices and derive "
            "the Rayleigh quotient variational bounds.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nProceed by induction on dimension n, isolating an extreme value of the "
            "quadratic form on the unit sphere and exploiting invariant orthogonal complements.\n</think>\n"
            "**Theorem:** If $A = A^T \\in \\mathbb{R}^{n \\times n}$, there exists an orthogonal matrix $Q \\in O(n)$ "
            "and real diagonal matrix $\\Lambda = \\operatorname{diag}(\\lambda_1, \\dots, \\lambda_n)$ such that $A = Q \\Lambda Q^T$.\n\n"
            "**Proof:** Maximize $R(x) = x^T A x$ over the compact unit sphere $S^{n-1}$. By Weierstrass's theorem, the maximum "
            "$\\lambda_1$ is attained at some unit vector $q_1$. First-order optimality of $x^T A x - \\lambda(x^T x - 1)$ "
            "gives $2 A q_1 - 2 \\lambda_1 q_1 = 0$, hence $A q_1 = \\lambda_1 q_1$. Let $W = q_1^\\perp$. For any $w \\in W$, "
            "$q_1^T (A w) = (A q_1)^T w = \\lambda_1 (q_1^T w) = 0$, so $A$ leaves $W$ invariant. Inducting on $A|_W$ completes "
            "the orthonormal eigenbasis. Consequently, $\\lambda_{\\min} \\|x\\|_2^2 \\le x^T A x \\le \\lambda_{\\max} \\|x\\|_2^2$.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nDerive the Karush-Kuhn-Tucker (KKT) conditions and Slater's constraint qualification for "
            "convex programs, plus the dual of regularized ridge regression.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nCombine separating hyperplane geometry on the epigraph with subgradient optimality.\n</think>\n"
            "Consider minimizing convex $f_0(x)$ subject to $f_i(x) \\le 0$ ($i=1\\dots m$) and $Ax = b$. Define the Lagrangian "
            "$L(x, \\lambda, \\nu) = f_0(x) + \\sum_i \\lambda_i f_i(x) + \\nu^T (Ax - b)$ with $\\lambda \\ge 0$. Under Slater's "
            "condition (existence of a strictly feasible interior point), strong duality $p^* = d^*$ holds and any primal-dual optimum "
            "satisfies primal/dual feasibility, complementary slackness $\\lambda_i^* f_i(x^*) = 0$, and stationarity $0 \\in \\partial_x L$.\n"
            "For weighted Ridge $(W x - y)^T \\Omega (W x - y) + \\delta \\|x\\|_2^2$, normal equations yield "
            "$(W^T \\Omega W + \\delta I) x^* = W^T \\Omega y$, solved stably via Cholesky factorization $H = L L^T$.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nExplain the Cooley-Tukey radix-2 FFT butterfly factorization and prove its O(N log N) complexity, "
            "then apply Strassen-like block decomposition to triangular Schur updates.<|im_end|>\n"
            "<|im_start|>assistant\nSplit $X[k] = \\sum_{n=0}^{N-1} x[n] \\omega_N^{nk}$ into even $x[2m]$ and odd $x[2m+1]$ indices: "
            "$X[k] = E[k] + \\omega_N^k O[k]$ and $X[k + N/2] = E[k] - \\omega_N^k O[k]$, where twiddle symmetry $\\omega_N^{N/2} = -1$ "
            "halves work recursively: $T(N) = 2 T(N/2) + \\Theta(N) = \\Theta(N \\log_2 N)$.\n"
            "For upper-triangular Cholesky block elimination partitioned as $U = \\begin{pmatrix} U_{11} & U_{12} \\\\ 0 & U_{22} \\end{pmatrix}$, "
            "updating remaining blocks via outer products $W_{:, b_1:} \\leftarrow W_{:, b_1:} - E_{\\text{blk}} U_{12}$ amortizes GEMM bandwidth across BLAS tiles.\n<|im_end|>"
        ),
        (
            "<|im_start|>user\nState the Max-Flow Min-Cut theorem and prove the runtime bound of Dinic's blocking flow algorithm on unit networks.<|im_end|>\n"
            "<|im_start|>assistant\nBy building BFS level graphs $d(v)$ from source $s$ and saturating shortest augmenting paths via DFS "
            "with current-arc pointer advancement, each blocking flow phase takes $O(V E)$ in general and guarantees strictly increasing "
            "sink distance $d_t^{(k+1)} \\ge d_t^{(k)} + 1$. On unit-capacity networks, edge count bounds length by $O(\\sqrt{E})$ and "
            "total work by $O(E \\min(V^{2/3}, E^{1/2}))$. By flow conservation across the reachable set $S$ in the terminal residual graph, "
            "$|f| = c(S, V \\setminus S)$, establishing strong combinatorial duality.\n<|im_end|>"
        ),
    ]


def _agentic_tool_texts() -> list[str]:
    return [
        (
            "<|im_start|>system\nYou are an SRE debugging Kubernetes control-plane incidents using diagnostic tools.<|im_end|>\n"
            "<|im_start|>user\nPods in namespace `payments-prod` are stuck in CrashLoopBackOff since 03:12 UTC. Check events and pod logs.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nLet's inspect recent Kubernetes events and fetch logs from a crashing pod in `payments-prod`.\n</think>\n"
            "<tool_call>\n{\"name\": \"kubectl_get_events\", \"arguments\": {\"namespace\": \"payments-prod\", \"limit\": 30}}\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n{\"events\": [{\"reason\": \"OOMKilled\", \"exit_code\": 137, \"pod\": \"ledger-worker-7c9f8b\", \"limit_mib\": 512, \"rss_peak_mib\": 514}]}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\nThe deployment `ledger-worker` is hitting its 512 MiB memory cgroup limit (`OOMKilled`, exit 137) during batch settlement peak at 514 MiB RSS. Raising `resources.limits.memory` to 1024 MiB or lowering batch chunk size resolves the crash.\n<|im_end|>"
        ),
        (
            "<|im_start|>system\nYou have access to one tool: `restart_staging_vm(vm_id, force)`.\n"
            "User request: Look up today's Tokyo weather forecast without restarting any virtual machines.<|im_end|>\n"
            "<|im_start|>assistant\nI cannot retrieve live Tokyo weather forecasts because I only have a virtual machine restart tool (`restart_staging_vm`) and no weather or web lookup tool is available. No VM action was executed.<|im_end|>"
        ),
        (
            "<|im_start|>user\nCompute quarterly SKU margin metrics for `SKU-9941` from warehouse ledger API.<|im_end|>\n"
            "<|im_start|>assistant\n<tool_call>\n{\"name\": \"query_ledger_api\", \"arguments\": {\"sku\": \"SKU-9941\", \"quarter\": \"2026Q2\"}}\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n{\"sku\": \"SKU-9941\", \"revenue_usd\": 148920.50, \"cogs_usd\": 96410.00, \"returns_usd\": 3120.50}\n</tool_response><|im_end|>\n"
            "<|im_start|>assistant\nNet revenue is `\\$145,800.00` after returns, giving gross profit `\\$49,390.00` (33.88% net gross margin).\n<|im_end|>"
        ),
    ]


def build_diverse_calib_sequences(tok, workspace_root: str, n_seqs: int = 40, seq_len: int = 1536):
    """
    Assembles n_seqs independent sequences of exact length seq_len tokens across 6 diverse domains,
    verifies strict zero overlap against held-out CALIB_DOC and AGENTS P1..P6 sentinels.
    """
    eval_doc = sq.CALIB_DOC

    ru_math_tools_texts = _russian_tech_texts() + _math_algo_texts() + _agentic_tool_texts()
    weights = [
        ("swebench",      12, _gather_swebench_texts()),
        ("code",          12, _gather_code_texts(workspace_root)),
        ("docs",           8, _gather_docs_texts(workspace_root)),
        ("ru_math_tools",  8, ru_math_tools_texts),
    ]
    raw_counts = [max(1 if n_seqs >= 4 else 0, round(n_seqs * w / 40.0)) for _, w, _ in weights]
    # Adjust exactly to n_seqs
    while sum(raw_counts) > n_seqs:
        i = max(range(len(raw_counts)), key=lambda k: raw_counts[k])
        raw_counts[i] -= 1
    while sum(raw_counts) < n_seqs:
        i = 0 if len(weights[0][2]) > 0 else 1
        raw_counts[i] += 1

    seqs = []
    domain_summary = {}
    for (dom, _, texts), cnt in zip(weights, raw_counts):
        if cnt <= 0 or not texts:
            domain_summary[dom] = 0
            continue
        dom_seqs = _build_domain_chunks(tok, texts, cnt, seq_len, eval_doc, dom)
        seqs.extend(dom_seqs[:cnt])
        domain_summary[dom] = len(dom_seqs[:cnt])

    # Top up any remaining slots from distinct SWE-bench / docs (never repeating synthetic texts)
    if len(seqs) < n_seqs:
        extra_sw = _build_domain_chunks(tok, _gather_swebench_texts()[15:], n_seqs - len(seqs), seq_len, eval_doc, "swebench_extra")
        take_sw = extra_sw[: n_seqs - len(seqs)]
        seqs.extend(take_sw)
        if take_sw:
            domain_summary["swebench_extra"] = len(take_sw)
    if len(seqs) < n_seqs:
        extra_cd = _build_domain_chunks(tok, _gather_code_texts(workspace_root), n_seqs - len(seqs), seq_len, eval_doc, "code_extra")
        take_cd = extra_cd[: n_seqs - len(seqs)]
        seqs.extend(take_cd)
        if take_cd:
            domain_summary["code_extra"] = len(take_cd)

    assert len(seqs) == n_seqs, f"Expected {n_seqs} seqs, built {len(seqs)}"
    ids_cal = torch.tensor(seqs, dtype=torch.long)
    sha = hashlib.sha256(ids_cal.numpy().tobytes()).hexdigest()[:16]
    return ids_cal, domain_summary, sha


# ---------------------------------------------------------------------------
# 2. Batched Forward Helpers & True AWQ Scale Search
# ---------------------------------------------------------------------------

def hc_mix_down_up_batched(X_b, w_down_f, w_inj_f, w_up, hc_norm, inject=True):
    """Exact token-wise HC mix over [B, L, 4, 2560]. Returns u [B, L, 2560], w_in [B, L, 4]."""
    B, L, H, d = X_b.shape
    u, win = sq.hc_mix_down_up(X_b.reshape(B * L, H, d), w_down_f, w_inj_f, w_up, hc_norm, inject=inject)
    return u.view(B, L, d), (win.view(B, L, H) if win is not None else None)


def gdn_forward_batched(u_b, w_in, w_out, conv_w, A_log, dt_bias, gdn_norm, return_o_gated=False):
    """
    Batched causal GDN mixer over [B, L, 2560]. Each sequence b in [0..B) keeps independent
    conv1d padding and independent Gated Delta Rule recurrence state S_b by reshaping B into heads!
    """
    B, L, _ = u_b.shape
    proj = F.linear(u_b, w_in)  # [B, L, 16480]
    raw_qkv = proj[..., :10240]
    z = proj[..., 10240:16384].view(B, L, 48, 128)
    b = proj[..., 16384:16432].float()
    a = proj[..., 16432:16480].float()

    # Causal depthwise conv1d independently along time L for each batch element B
    zt = raw_qkv.transpose(1, 2).float()  # [B, 10240, L]
    cv = F.conv1d(F.pad(zt, (3, 0)), conv_w.squeeze(1).unsqueeze(1).float(), groups=10240)
    qkv = F.silu(cv.transpose(1, 2))  # [B, L, 10240]

    q = F.normalize(qkv[..., :2048].view(B, L, 16, 128).float(), p=2, dim=-1, eps=1e-6) * (128.0 ** -0.5)
    k = F.normalize(qkv[..., 2048:4096].view(B, L, 16, 128).float(), p=2, dim=-1, eps=1e-6)
    v = qkv[..., 4096:10240].view(B, L, 48, 128).float()

    q = q.repeat_interleave(3, dim=2)  # [B, L, 48, 128]
    k = k.repeat_interleave(3, dim=2)

    beta = torch.sigmoid(b)  # [B, L, 48]
    g = -torch.exp(A_log.float()).view(1, 1, 48) * F.softplus(a + dt_bias.float().view(1, 1, 48))

    # Flatten B into head dimension -> [L, B*48, 128]
    q_h = q.transpose(0, 1).reshape(L, B * 48, 128)
    k_h = k.transpose(0, 1).reshape(L, B * 48, 128)
    v_h = v.transpose(0, 1).reshape(L, B * 48, 128)
    b_h = beta.transpose(0, 1).reshape(L, B * 48)
    g_h = g.transpose(0, 1).reshape(L, B * 48)

    o_h = sq.chunked_delta_rule_gdn(q_h, k_h, v_h, b_h, g_h, chunk_size=64)  # [L, B*48, 128]
    o = o_h.view(L, B, 48, 128).transpose(0, 1)  # [B, L, 48, 128]

    o_norm = sq.rmsnorm_raw(o.to(u_b.dtype), gdn_norm, 1e-6)
    o_gated = (o_norm.float() * torch.sigmoid(z.float())).to(u_b.dtype).reshape(B, L, 6144)
    y = F.linear(o_gated, w_out)
    return (y, o_gated) if return_o_gated else y


def full_attn_forward_batched(u_b, w_qkv, w_o, q_norm, k_norm, cos, sin):
    """Batched GQA full attention over [B, L, 2560], L <= 2051."""
    B, L, _ = u_b.shape
    qkv = F.linear(u_b, w_qkv)
    qg = qkv[..., :12288].view(B, L, 24, 512)
    q, gate = qg[..., :256], qg[..., 256:]
    k = qkv[..., 12288:12800].view(B, L, 2, 256)
    v = qkv[..., 12800:13312].view(B, L, 2, 256)

    q = sq.rope_rotate_half(sq.rmsnorm_zc(q, q_norm, 1e-6).transpose(1, 2), cos, sin)  # [B, 24, L, 256]
    k = sq.rope_rotate_half(sq.rmsnorm_zc(k, k_norm, 1e-6).transpose(1, 2), cos, sin).repeat_interleave(12, dim=1)
    v = v.transpose(1, 2).repeat_interleave(12, dim=1)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=256.0 ** -0.5).transpose(1, 2)
    out = (attn.float() * torch.sigmoid(gate.float())).to(u_b.dtype).reshape(B, L, 6144)
    return F.linear(out, w_o)


def make_awq_scales(act_abs: torch.Tensor, alpha: float, s_min: float = 0.25, s_max: float = 4.0) -> torch.Tensor:
    """AWQ per-channel smoothing scales s = clamp(normalize(mean(|x|)^alpha), s_min, s_max)."""
    if alpha == 0.0:
        return torch.ones_like(act_abs, dtype=torch.float32)
    x = act_abs.float().clamp(min=1e-6)
    s = x.pow(alpha)
    norm = torch.sqrt(s.max(dim=-1, keepdim=True).values * s.min(dim=-1, keepdim=True).values).clamp(min=1e-6)
    s = (s / norm).clamp(min=s_min, max=s_max)
    return s


def search_awq_2d(W: torch.Tensor, H_gpu_f32: torch.Tensor, act_abs: torch.Tensor, fmt: str = "q4g128", alphas=ALPHAS):
    """Exact output Hessian MSE grid search for 2D weight matrix W [N, K] (gdn_in)."""
    best_loss = float("inf")
    base_loss = None
    best_alpha = 0.0
    best_s = torch.ones(W.shape[1], device=W.device, dtype=torch.float32)

    for a in alphas:
        s = make_awq_scales(act_abs, a)
        W_q = sq.fake_quant(W * s.unsqueeze(0), fmt).float() / s.unsqueeze(0)
        E = W_q - W
        # tr(E H E^T) = sum((E @ H) * E)
        loss = float(((E @ H_gpu_f32) * E).sum().item())
        if a == 0.0:
            base_loss = loss
        if loss < best_loss:
            best_loss = loss
            best_alpha = a
            best_s = s
        del W_q, E
    ratio = best_loss / max(base_loss, 1e-12)
    return best_s, best_alpha, ratio


def search_awq_mlp(W_exp_chunk0: torch.Tensor, W_sh_gu: torch.Tensor, H_mlp_f32: torch.Tensor, act_abs: torch.Tensor, counts_chunk0: torch.Tensor, T_cal: int, alphas=ALPHAS):
    """
    Searches shared s_mlp [2560] for mlp_hc.hc_norm minimizing combined exact Hessian MSE
    across sh_gu (q6g64) + chunk0 active routed exp_gu (q4g128), weighted by routing frequencies.
    """
    best_loss = float("inf")
    base_loss = None
    best_alpha = 0.0
    best_s = torch.ones(W_sh_gu.shape[1], device=W_sh_gu.device, dtype=torch.float32)
    active = torch.nonzero(counts_chunk0 > 0, as_tuple=True)[0]
    w_exp = (counts_chunk0[active].float() / float(T_cal)).to(W_sh_gu.device)

    for a in alphas:
        s = make_awq_scales(act_abs, a)
        E_sh = sq.fake_quant(W_sh_gu * s.unsqueeze(0), "q6g64").float() / s.unsqueeze(0) - W_sh_gu
        loss = float(((E_sh @ H_mlp_f32) * E_sh).sum().item())
        del E_sh

        # Stream sub-slices of chunk0 experts to keep temp VRAM small
        if active.numel() > 0:
            exp_loss = 0.0
            for i0 in range(0, active.numel(), 16):
                idx_sub = active[i0 : i0 + 16]
                we_sub = w_exp[i0 : i0 + 16]
                W_sub = W_exp_chunk0[idx_sub]  # [m, 1280, 2560]
                E_sub = sq.fake_quant(W_sub * s.view(1, 1, -1), "q4g128").float() / s.view(1, 1, -1) - W_sub
                EH = torch.matmul(E_sub, H_mlp_f32)
                per_e = (EH * E_sub).sum(dim=(1, 2))
                exp_loss += float((per_e * we_sub).sum().item())
                del W_sub, E_sub, EH, per_e
            # Extrapolate 64/512 slice uniformly onto all 512 routed experts (512/64 = 8x) vs 1x sh_gu
            loss += exp_loss * (512.0 / float(W_exp_chunk0.shape[0]))

        if a == 0.0:
            base_loss = loss
        if loss < best_loss:
            best_loss = loss
            best_alpha = a
            best_s = s
    ratio = best_loss / max(base_loss, 1e-12)
    return best_s, best_alpha, ratio


def search_awq_exp_down_batched(W_dn: torch.Tensor, H_dn: torch.Tensor, a_eff: torch.Tensor, hits: torch.Tensor, fmt: str = "q4g128", alphas=ALPHAS):
    """
    Vectorized AWQ grid search across cnt experts [cnt, 2560, 640] simultaneously using exact
    per-expert Hessian H_dn [cnt, 640, 640]. Forces s=1 for rare experts (hits < 16).
    Returns s_best [cnt, 640], best_alphas [cnt], mean_mse_ratio.
    """
    cnt = W_dn.shape[0]
    best_loss = torch.full((cnt,), float("inf"), device=W_dn.device, dtype=torch.float32)
    base_loss = torch.ones((cnt,), device=W_dn.device, dtype=torch.float32)
    best_s = torch.ones((cnt, W_dn.shape[2]), device=W_dn.device, dtype=torch.float32)
    best_a = torch.zeros((cnt,), device=W_dn.device, dtype=torch.float32)

    for a in alphas:
        s = make_awq_scales(a_eff, a)  # [cnt, 640]
        W_q = sq.fake_quant(W_dn * s.unsqueeze(1), fmt).float() / s.unsqueeze(1)
        E = W_q - W_dn
        loss = (torch.bmm(E, H_dn) * E).sum(dim=(1, 2))
        if a == 0.0:
            base_loss = loss.clamp(min=1e-12)
        better = loss < best_loss
        best_loss = torch.where(better, loss, best_loss)
        best_s = torch.where(better.unsqueeze(-1), s, best_s)
        best_a = torch.where(better, torch.full_like(best_a, a), best_a)
        del W_q, E

    eligible = hits >= 64
    best_s = torch.where(eligible.unsqueeze(-1), best_s, torch.ones_like(best_s))
    best_a = torch.where(eligible, best_a, torch.zeros_like(best_a))
    ratios = (best_loss / base_loss)[eligible]
    mean_ratio = float(ratios.mean().item()) if ratios.numel() > 0 else 1.0
    return best_s, best_a, mean_ratio


# ---------------------------------------------------------------------------
# 3. Pure RTN-Grid GPTQ (Zero Range Clipping — Shared 2D/3D + Batched Per-Expert)
# ---------------------------------------------------------------------------

def _rtn_minmax_params(W_blk: torch.Tensor, levels: float):
    """Exact pure RTN min-max BF16 (scale, minv) matching formats/q4.cpp with zero clipping."""
    lo = W_blk.amin(dim=-1)
    hi = W_blk.amax(dim=-1)
    s_vec = ((hi - lo) / levels).to(torch.bfloat16).float()
    m_vec = lo.to(torch.bfloat16).float()
    safe_s = torch.where(s_vec > 0.0, s_vec, torch.ones_like(s_vec))
    return s_vec, m_vec, safe_s


def quant_gptq_rtn_grid(
    w_2d_or_3d: torch.Tensor,
    H_cpu_f64: torch.Tensor = None,
    U_gpu: torch.Tensor = None,
    fmt: str = "q4g128",
    damp: float = 0.01,
    row_subbatch: int = 40960,
) -> torch.Tensor:
    """
    Block-wise GPTQ with PURE RTN min-max (scale, minv) and zero range clipping.
    Processes rows in row_subbatch chunks to bound VRAM on large MoE stacks [64, 1280, 2560].
    Accepts either H_cpu_f64 or precomputed upper Cholesky factor U_gpu [K, K] FP32.
    """
    orig_shape = w_2d_or_3d.shape
    K = orig_shape[-1]
    G = int(fmt[3:])
    if K % G != 0:
        G = math.gcd(K, G)
    bits = int(fmt[1])
    levels = float((1 << bits) - 1)
    dev = w_2d_or_3d.device

    if U_gpu is None:
        assert H_cpu_f64 is not None
        U_gpu = sc.compute_hinv_chol_cpu(H_cpu_f64, damp=damp).to(dev)

    W_flat = w_2d_or_3d.reshape(-1, K)
    R = W_flat.shape[0]
    Q_out = torch.empty((R, K), device=dev, dtype=torch.bfloat16)

    for r0 in range(0, R, row_subbatch):
        r1 = min(r0 + row_subbatch, R)
        W = W_flat[r0:r1].float().clone()
        Q = torch.empty_like(W)

        for b0 in range(0, K, G):
            b1 = b0 + G
            W_blk = W[:, b0:b1].clone()
            s_vec, m_vec, safe_s = _rtn_minmax_params(W_blk, levels)
            Err_blk = torch.zeros_like(W_blk)
            Q_blk = torch.empty_like(W_blk)
            U_blk = U_gpu[b0:b1, b0:b1]

            for j in range(G):
                wj = W_blk[:, j]
                code = ((wj - m_vec) / safe_s).round().clamp(0.0, levels)
                code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
                qj = code * s_vec + m_vec
                Q_blk[:, j] = qj
                err = (wj - qj) / U_blk[j, j]
                Err_blk[:, j] = err
                if j + 1 < G:
                    W_blk[:, j + 1 :] -= err.unsqueeze(1) * U_blk[j, j + 1 :].unsqueeze(0)

            Q[:, b0:b1] = Q_blk
            if b1 < K:
                W[:, b1:] -= Err_blk @ U_gpu[b0:b1, b1:]

        Q_out[r0:r1] = Q.to(torch.bfloat16)
        del W, Q

    return Q_out.reshape(orig_shape)


def quant_gptq_batched_exp_down_rtn(
    W_dn_in: torch.Tensor,       # [cnt, 2560, 640] FP32 (raw or AWQ-smoothed)
    H_dn_raw: torch.Tensor,      # [cnt, 640, 640] FP32
    s_dn: torch.Tensor,          # [cnt, 640] FP32 or None
    hits: torch.Tensor,          # [cnt] int64
    fallback_bf16: torch.Tensor, # [cnt, 2560, 640] BF16 pure RTN fallback
    min_hits: int = 640,
    base_damp: float = 0.08,
    fmt: str = "q4g128",
) -> tuple[torch.Tensor, int]:
    """
    Batched full Cholesky GPTQ across experts with hits >= min_hits using PURE RTN min-max grids
    (zero range clipping). Uses exact H_sm = diag(1/s) H_raw diag(1/s) when s_dn is provided.
    Experts with hits < min_hits keep pure RTN fallback_bf16.
    """
    out = fallback_bf16.clone()
    elig = torch.nonzero(hits >= min_hits, as_tuple=True)[0]
    n_elig = int(elig.numel())
    if n_elig == 0:
        return out, 0

    K = W_dn_in.shape[2]
    G = int(fmt[3:])
    bits = int(fmt[1])
    levels = float((1 << bits) - 1)
    dev = W_dn_in.device

    H_sel = H_dn_raw[elig].double().cpu()
    s_sel = s_dn[elig].double().cpu() if s_dn is not None else None
    hits_sel = hits[elig].tolist()
    U_list = []
    idx_k = torch.arange(K)
    for k_i, h_cnt in enumerate(hits_sel):
        H_sm = H_sel[k_i] if s_sel is None else (H_sel[k_i] / s_sel[k_i].unsqueeze(1)) / s_sel[k_i].unsqueeze(0)
        d_k = max(base_damp, base_damp * (1280.0 / float(h_cnt)))
        diag = torch.diag(H_sm)
        mean_d = float(diag.mean().item())
        U_ok = None
        if mean_d > 0.0:
            for mult in (1.0, 2.5, 6.0):
                H_d = H_sm.clone()
                dead = diag <= 0.0
                H_d[dead, dead] = 1.0
                H_d[idx_k, idx_k] += (d_k * mult) * mean_d
                try:
                    L = torch.linalg.cholesky(H_d)
                    H_inv = torch.cholesky_inverse(L)
                    U_ok = torch.linalg.cholesky(H_inv, upper=True).float()
                    break
                except Exception:
                    continue
        U_list.append(U_ok if U_ok is not None else torch.eye(K, dtype=torch.float32))

    U_b = torch.stack(U_list, dim=0).to(dev)  # [n_elig, K, K]
    W_b = W_dn_in[elig].float().clone()       # [n_elig, N, K]
    Q_b = torch.empty_like(W_b)

    for b0 in range(0, K, G):
        b1 = b0 + G
        W_blk = W_b[:, :, b0:b1].clone()  # [n_elig, N, G]
        s_vec, m_vec, safe_s = _rtn_minmax_params(W_blk, levels)
        Err_blk = torch.zeros_like(W_blk)
        Q_blk = torch.empty_like(W_blk)
        U_blk = U_b[:, b0:b1, b0:b1]

        for j in range(G):
            wj = W_blk[:, :, j]
            code = ((wj - m_vec) / safe_s).round().clamp(0.0, levels)
            code = torch.where(s_vec > 0.0, code, torch.zeros_like(code))
            qj = code * s_vec + m_vec
            Q_blk[:, :, j] = qj
            err = (wj - qj) / U_blk[:, j, j].unsqueeze(1)
            Err_blk[:, :, j] = err
            if j + 1 < G:
                W_blk[:, :, j + 1 :] -= err.unsqueeze(2) * U_blk[:, j, j + 1 :].unsqueeze(1)

        Q_b[:, :, b0:b1] = Q_blk
        if b1 < K:
            W_b[:, :, b1:] -= torch.bmm(Err_blk, U_b[:, b0:b1, b1:])

    out[elig] = Q_b.to(torch.bfloat16)
    return out, n_elig


# ---------------------------------------------------------------------------
# 4. Pre-Run Algebraic & Numerical Self-Tests
# ---------------------------------------------------------------------------

def run_self_tests(dev: torch.device) -> None:
    torch.manual_seed(42)

    # Test 1: hc_mix_down_up AWQ folding algebraic equivalence
    T, H, d = 16, 4, 2560
    X = torch.randn(T, H, d, device=dev, dtype=torch.bfloat16)
    w_down = torch.randn(320, H * d, device=dev, dtype=torch.float32) * 0.02
    w_inj = torch.randn(H, H * d, device=dev, dtype=torch.float32) * 0.02
    w_up = torch.randn(H * d, 320, device=dev, dtype=torch.float32) * 0.02
    hc_norm = torch.randn(H * d, device=dev, dtype=torch.float32) * 0.2
    W_lin = torch.randn(512, d, device=dev, dtype=torch.float32) * 0.02

    w_df = (w_down * (1.0 + hc_norm.unsqueeze(0))).to(torch.bfloat16)
    w_if = (w_inj * (1.0 + hc_norm.unsqueeze(0))).to(torch.bfloat16)
    w_ub = w_up.to(torch.bfloat16)

    u_ref, win_ref = sq.hc_mix_down_up(X, w_df, w_if, w_ub, hc_norm, inject=True)
    y_ref = F.linear(u_ref, W_lin.to(torch.bfloat16))

    s = torch.exp(torch.linspace(-1.0, 1.0, d, device=dev, dtype=torch.float32)).clamp(0.25, 4.0)
    hc_norm_awq = (1.0 + hc_norm) / s.repeat(H) - 1.0
    u_awq, win_awq = sq.hc_mix_down_up(X, w_df, w_if, w_ub, hc_norm_awq, inject=True)
    y_awq = F.linear(u_awq, (W_lin * s.unsqueeze(0)).to(torch.bfloat16))

    win_diff = float((win_ref - win_awq).abs().max().item())
    y_rel = float(((y_ref.float() - y_awq.float()).norm() / y_ref.float().norm()).item())
    assert win_diff == 0.0, f"Test 1 failed: w_in changed by {win_diff}"
    assert y_rel < 5e-3, f"Test 1 failed: BF16 HC AWQ y rel_err={y_rel}"
    print(f"[PASS] Test 1 (HC mix AWQ folding): w_in_diff={win_diff:.1e}, y_bf16_rel_err={y_rel:.5f}")

    # Test 2: W_up -> SwiGLU -> W_down AWQ folding + W_up q4g128 losslessness
    W_gu = torch.randn(1280, d, device=dev, dtype=torch.float32) * 0.04
    W_dn = torch.randn(d, 640, device=dev, dtype=torch.float32) * 0.04
    u_tok = torch.randn(32, d, device=dev, dtype=torch.float32) * 0.5
    s_dn = torch.exp(torch.linspace(-1.2, 1.2, 640, device=dev, dtype=torch.float32)).clamp(0.25, 4.0)

    gu0 = F.linear(u_tok, W_gu)
    h0 = F.silu(gu0[:, :640]) * gu0[:, 640:]
    out0 = F.linear(h0, W_dn)

    W_gu_f = W_gu.clone()
    W_gu_f[640:, :] /= s_dn.unsqueeze(-1)
    W_dn_f = W_dn * s_dn.unsqueeze(0)
    gu1 = F.linear(u_tok, W_gu_f)
    h1 = F.silu(gu1[:, :640]) * gu1[:, 640:]
    out1 = F.linear(h1, W_dn_f)
    rel_fp32 = float(((out0 - out1).norm() / out0.norm()).item())
    # Check W_up relative q4g128 error before vs after per-row scaling (only BF16 scale rounding differs)
    q_up0 = sq.fake_quant(W_gu[640:], "q4g128").float()
    q_up1 = (sq.fake_quant(W_gu_f[640:], "q4g128").float() * s_dn.unsqueeze(-1))
    up_rel_diff = float(((q_up0 - W_gu[640:]).norm() - (q_up1 - W_gu[640:]).norm()).abs().item() / W_gu[640:].norm().item())
    assert rel_fp32 < 1e-5 and up_rel_diff < 5e-4, f"Test 2 failed: rel_fp32={rel_fp32}, up_rel_diff={up_rel_diff}"
    print(f"[PASS] Test 2 (W_up -> SwiGLU -> W_down AWQ folding): fp32_rel={rel_fp32:.2e}, W_up q4g128 row-scale invariance={up_rel_diff:.2e}")

    # Test 3: gdn_forward_batched vs sq.gdn_forward AND quant_gptq_batched vs sc.quant_gptq
    B_t, L_t = 3, 128
    u_g = torch.randn(B_t, L_t, 2560, device=dev, dtype=torch.bfloat16)
    wg_in = (torch.randn(16480, 2560, device=dev) * 0.02).to(torch.bfloat16)
    wg_out = (torch.randn(2560, 6144, device=dev) * 0.02).to(torch.bfloat16)
    cw = torch.randn(10240, 1, 4, device=dev, dtype=torch.float32) * 0.1
    A_log = torch.randn(48, device=dev, dtype=torch.float32) * 0.5 - 2.0
    dt_b = torch.randn(48, device=dev, dtype=torch.float32) * 0.2
    gnorm = torch.ones(128, device=dev, dtype=torch.float32)

    yb_f32 = gdn_forward_batched(u_g.float(), wg_in.float(), wg_out.float(), cw, A_log, dt_b, gnorm)
    y_s_f32 = torch.stack([sq.gdn_forward(u_g[k].float(), wg_in.float(), wg_out.float(), cw, A_log, dt_b, gnorm) for k in range(B_t)], dim=0)
    gdn_diff_f32 = float((yb_f32 - y_s_f32).abs().max().item())
    yb = gdn_forward_batched(u_g, wg_in, wg_out, cw, A_log, dt_b, gnorm)
    y_singles = torch.stack([sq.gdn_forward(u_g[k], wg_in, wg_out, cw, A_log, dt_b, gnorm) for k in range(B_t)], dim=0)
    gdn_rel_bf16 = float(((yb.float() - y_singles.float()).norm() / y_singles.float().norm()).item())

    # Test 3b: Pure RTN-grid GPTQ vs sq.fake_quant (>= 40% MSE cut) & exact match with batched GPTQ
    E_t, N_t, K_t = 2, 16, 256
    W_3d = torch.randn(E_t, N_t, K_t, device=dev, dtype=torch.float32) * 0.05
    X_c0 = torch.randn(512, K_t, dtype=torch.float64)
    corr_t = torch.randn(K_t, K_t, dtype=torch.float64) * 0.35 + torch.eye(K_t, dtype=torch.float64)
    X_corr = X_c0 @ corr_t
    H0 = (X_corr.T @ X_corr) / 512.0
    H1 = H0 * 1.3
    Q_rtn_3d = sq.fake_quant(W_3d[0], "q4g128")
    Q_s0 = quant_gptq_rtn_grid(W_3d[0], H_cpu_f64=H0, fmt="q4g128", damp=0.05, row_subbatch=8)
    Xg = X_corr.float().to(dev)
    mse_rtn_g = float((((W_3d[0] - Q_rtn_3d.float()) @ Xg.T).pow(2).sum()).item())
    mse_gptq_g = float((((W_3d[0] - Q_s0.float()) @ Xg.T).pow(2).sum()).item())
    hits_t = torch.tensor([1280, 1280], device=dev, dtype=torch.long)
    Q_b, n_el = quant_gptq_batched_exp_down_rtn(
        W_3d,
        torch.stack([H0.float(), H1.float()], dim=0).to(dev),
        None,
        hits_t,
        sq.fake_quant(W_3d, "q4g128"),
        min_hits=640,
        base_damp=0.05,
        fmt="q4g128",
    )
    gptq_diff = float((Q_b[0].float() - Q_s0.float()).abs().max().item())
    assert n_el == 2 and gdn_diff_f32 < 2e-5 and gdn_rel_bf16 < 5e-3 and gptq_diff < 1e-5 and mse_gptq_g <= 0.60 * mse_rtn_g, (
        f"Test 3 failed: gdn_diff_f32={gdn_diff_f32}, gdn_rel_bf16={gdn_rel_bf16}, gptq_diff={gptq_diff}, mse_gptq={mse_gptq_g} vs {mse_rtn_g}"
    )
    print(f"[PASS] Test 3 (Batched GDN & Pure RTN-Grid GPTQ): gdn_f32={gdn_diff_f32:.2e}, gdn_bf16_rel={gdn_rel_bf16:.5f}, gptq_diff={gptq_diff:.2e}, GPTQ_vs_RTN={(mse_gptq_g/mse_rtn_g - 1)*100:+.1f}%")

    # Test 4: Synthetic outlier channel AWQ test (>= 50% output MSE reduction vs plain RTN)
    K_o, T_o = 512, 512
    X_o = torch.randn(T_o, K_o, device=dev, dtype=torch.float32)
    for col_o in (7, 63, 193, 311, 444):
        X_o[:, col_o] *= 22.0
    W_o = torch.randn(64, K_o, device=dev, dtype=torch.float32) * 0.03
    H_o = (X_o.T @ X_o) / float(T_o)
    act_o = X_o.abs().mean(dim=0)
    Q_rtn_o = sq.fake_quant(W_o, "q4g128").float()
    err_rtn = float((((W_o - Q_rtn_o) @ X_o.T).pow(2).sum()).item())
    s_o, a_o, _ = search_awq_2d(W_o, H_o, act_o, fmt="q4g128")
    Q_awq_o = sq.fake_quant(W_o * s_o.unsqueeze(0), "q4g128").float() / s_o.unsqueeze(0)
    err_awq = float((((W_o - Q_awq_o) @ X_o.T).pow(2).sum()).item())
    assert err_awq <= 0.50 * err_rtn, f"Test 4 failed: err_awq={err_awq} vs err_rtn={err_rtn}"
    print(f"[PASS] Test 4 (Synthetic outlier AWQ): alpha={a_o:.2f}, output_MSE cut {(err_awq/err_rtn - 1)*100:+.1f}%")


# ---------------------------------------------------------------------------
# 5. Full 48-Layer Calibration & Held-Out Simulator
# ---------------------------------------------------------------------------

VARIANTS = [
    "0_RTN_Prod",
    "1_AWQ_ExpDn_Only_RTN",
    "2_GPTQ_RTNGrid_GdnIn",
    "3_GPTQ_RTNGrid_Gdn_ExpGU",
    "4_GPTQ_All3_RTNGrid",
    "5_GPTQ_All3_plus_AWQ_ExpDn",
]


def run_awq_gptq_sim(
    model_dir: str,
    n_seqs: int = 40,
    seq_len: int = 1536,
    cal_mb: int = 4,
    expert_chunk: int = 64,
    gdn_damp: float = 0.01,
    exp_gu_damp: float = 0.05,
    dn_damp: float = 0.08,
    dn_min_hits: int = 640,
):
    dev = torch.device("cuda:0")
    _ = torch.zeros(1, device=dev)
    torch.cuda.synchronize()
    cpu = torch.device("cpu")
    torch.cuda.reset_peak_memory_stats(dev)

    sc.verify_gptq_math()
    run_self_tests(dev)

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tok = sq.QwenTokenizer(os.path.join(model_dir, "tokenizer.json"))
    ids_eval = tok.encode(sq.CALIB_DOC)
    T_eval = len(ids_eval)

    ids_cal_2d, dom_summary, cal_sha = build_diverse_calib_sequences(tok, workspace_root, n_seqs=n_seqs, seq_len=seq_len)
    T_cal = n_seqs * seq_len
    print(f"Held-out Eval: T_eval={T_eval} | Disjoint Calib: {n_seqs}x{seq_len} = {T_cal} tokens (sha={cal_sha}) | domains={dom_summary}")

    reader = sq.SafetensorsReader(model_dir)
    lm = "model.language_model."
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg_root = json.load(f)
    tc = cfg_root.get("text_config", cfg_root)
    n_layers = tc["num_hidden_layers"]
    layer_types = tc.get("layer_types")

    target_eval = torch.tensor(ids_eval[1:], device=dev, dtype=torch.long)
    cos_e, sin_e = sq.build_qwen_rope(T_eval, dev, torch.bfloat16)
    cos_c, sin_c = sq.build_qwen_rope(seq_len, dev, torch.bfloat16)

    state_names = ["BF16"] + VARIANTS
    prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t0 = time.time()
    layer_telemetry = []

    with torch.no_grad():
        emb_cpu = reader.read_tensor(lm + "embed_tokens.weight", cpu, torch.bfloat16)
        x_eval0 = F.embedding(torch.tensor(ids_eval, dtype=torch.long), emb_cpu).to(dev)
        cal_states_cpu = F.embedding(ids_cal_2d, emb_cpu).unsqueeze(2).expand(n_seqs, seq_len, 4, 2560).contiguous()
        del emb_cpu
        torch.cuda.empty_cache()

        states = {n: x_eval0.unsqueeze(1).expand(T_eval, 4, 2560).contiguous().clone() for n in state_names}
        del x_eval0

        for L in range(n_layers):
            tl = time.time()
            P = f"{lm}layers.{L}."
            is_attn = (layer_types[L] == "full_attention") if layer_types is not None else ((L + 1) % 4 == 0)
            lt = {"layer": L, "type": "attn" if is_attn else "gdn"}

            # ---- Optional PLE Injection (Layer 1, pure RTN q6g64) ----
            if (P + "ple.key_proj.weight") in reader.wm:
                ple_p = P + "ple."
                mult = reader.read_tensor(ple_p + "ple_embedding.layer_multipliers", cpu).tolist()
                vsz = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_vocab_sizes", cpu).tolist()
                offs = reader.read_tensor(ple_p + "ple_embedding.ngram_heads_offsets", cpu).tolist()

                ur_e, mi_e = sq.ple_ngram_ids_py(mult, vsz, offs, ids_eval, eos=248044)
                ple_e = reader.gather_ple_rows(ple_p, ur_e, dev)[torch.tensor(mi_e, device=dev, dtype=torch.long)].reshape(T_eval, 2560)

                w_kp = reader.read_tensor(ple_p + "key_proj.weight", dev, torch.float32)
                w_vp = reader.read_tensor(ple_p + "value_proj.weight", dev, torch.float32)
                w_ple = torch.cat([w_kp, w_vp], dim=0)
                del w_kp, w_vp
                ple_cw = reader.read_tensor(ple_p + "conv1d.weight", dev, torch.float32)
                ple_nk = reader.read_tensor(ple_p + "norm_key.weight", dev, torch.float32)
                ple_nq = reader.read_tensor(ple_p + "norm_query.weight", dev, torch.float32)
                ple_nc = reader.read_tensor(ple_p + "norm_conv.weight", dev, torch.float32)
                w_ple_bf16 = w_ple.to(torch.bfloat16)
                w_ple_rtn = sq.fake_quant(w_ple, "q6g64")
                del w_ple

                # Advance calibration states cleanly per sequence so n-gram history & conv1d never leak
                for b_idx in range(n_seqs):
                    seq_b = ids_cal_2d[b_idx].tolist()
                    ur_c, mi_c = sq.ple_ngram_ids_py(mult, vsz, offs, seq_b, eos=248044)
                    ple_c = reader.gather_ple_rows(ple_p, ur_c, dev)[torch.tensor(mi_c, device=dev, dtype=torch.long)].reshape(seq_len, 2560)
                    xc_b = cal_states_cpu[b_idx].to(dev)
                    cal_states_cpu[b_idx] = sq.ple_forward(xc_b, ple_c, w_ple_bf16, ple_cw, ple_nk, ple_nq, ple_nc).cpu()
                    del ple_c, xc_b

                for name in state_names:
                    wp = w_ple_bf16 if name == "BF16" else w_ple_rtn
                    states[name] = sq.ple_forward(states[name], ple_e, wp, ple_cw, ple_nk, ple_nq, ple_nc)
                del ple_e, w_ple_bf16, w_ple_rtn, ple_cw, ple_nk, ple_nq, ple_nc

            # ---- Attention / GDN Hyper-Connection Mix ----
            ahc = P + "attn_hyper_connection."
            a_wd = reader.read_tensor(ahc + "input_mix_weight_down.weight", dev, torch.float32)
            a_wi = reader.read_tensor(ahc + "block_inject_weight.weight", dev, torch.float32)
            a_wu = reader.read_tensor(ahc + "input_mix_weight_up.weight", dev, torch.float32)
            a_hn = reader.read_tensor(ahc + "hc_norm.weight", dev, torch.float32)
            a_wdf = a_wd * (1.0 + a_hn.unsqueeze(0))
            a_wif = a_wi * (1.0 + a_hn.unsqueeze(0))
            del a_wd, a_wi

            ahcd_bf16 = (a_wdf.to(torch.bfloat16), a_wif.to(torch.bfloat16))
            ahcu_bf16 = a_wu.to(torch.bfloat16)
            ahcd_q8 = (sq.fake_quant(a_wdf, "q8g64"), sq.fake_quant(a_wif, "q8g64"))
            ahcu_q8 = sq.fake_quant(a_wu, "q8g64")

            if not is_attn:
                gp = P + "linear_attn."
                w_qkv = reader.read_tensor(gp + "in_proj_qkv.weight", dev, torch.float32)
                w_z = reader.read_tensor(gp + "in_proj_z.weight", dev, torch.float32)
                w_b = reader.read_tensor(gp + "in_proj_b.weight", dev, torch.float32)
                w_a = reader.read_tensor(gp + "in_proj_a.weight", dev, torch.float32)
                w_gdn_in = torch.cat([w_qkv, w_z, w_b, w_a], dim=0)
                del w_qkv, w_z, w_b, w_a
                w_gdn_out = reader.read_tensor(gp + "out_proj.weight", dev, torch.float32)
                cw = reader.read_tensor(gp + "conv1d.weight", dev, torch.float32)
                A_log = reader.read_tensor(gp + "A_log", dev, torch.float32)
                dt_bias = reader.read_tensor(gp + "dt_bias", dev, torch.float32)
                gnorm = reader.read_tensor(gp + "norm.weight", dev, torch.float32)
                gin_bf16 = w_gdn_in.to(torch.bfloat16)
                gout_bf16 = w_gdn_out.to(torch.bfloat16)
                gout_rtn = sq.fake_quant(w_gdn_out, "q6g64")
                del w_gdn_out

                # Stream all B_cal sequences in micro-batches through attn_hc + BF16 GDN; collect H_gdn + act_abs
                H_gdn_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
                abs_gdn_sum = torch.zeros((2560,), device=dev, dtype=torch.float32)
                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = hc_mix_down_up_batched(xb, *ahcd_bf16, ahcu_bf16, a_hn, inject=True)
                    u_flat = ub.reshape(-1, 2560)
                    uf64 = u_flat.double()
                    H_gdn_cpu.add_((uf64.T @ uf64).cpu())
                    abs_gdn_sum.add_(u_flat.float().abs().sum(dim=0))
                    ym = gdn_forward_batched(ub, gin_bf16, gout_bf16, cw, A_log, dt_bias, gnorm)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, u_flat, uf64, ym

                H_gdn_cpu.div_(float(T_cal))

                gin_rtn = sq.fake_quant(w_gdn_in, "q4g128")
                gin_gptq = quant_gptq_rtn_grid(w_gdn_in, H_cpu_f64=H_gdn_cpu, fmt="q4g128", damp=gdn_damp, row_subbatch=40960)
                gin_variants = {
                    "BF16": gin_bf16,
                    "0_RTN_Prod": gin_rtn,
                    "1_AWQ_ExpDn_Only_RTN": gin_rtn,
                    "2_GPTQ_RTNGrid_GdnIn": gin_gptq,
                    "3_GPTQ_RTNGrid_Gdn_ExpGU": gin_gptq,
                    "4_GPTQ_All3_RTNGrid": gin_gptq,
                    "5_GPTQ_All3_plus_AWQ_ExpDn": gin_gptq,
                }

                for name in state_names:
                    hd = ahcd_bf16 if name == "BF16" else ahcd_q8
                    hu = ahcu_bf16 if name == "BF16" else ahcu_q8
                    u_e, win_e = sq.hc_mix_down_up(states[name], *hd, hu, a_hn, inject=True)
                    go = gout_bf16 if name == "BF16" else gout_rtn
                    ym = sq.gdn_forward(u_e, gin_variants[name], go, cw, A_log, dt_bias, gnorm)
                    states[name] = (states[name].float() + win_e.unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)
                    del u_e, win_e, ym

                del w_gdn_in, gin_bf16, gin_rtn, gin_gptq, gout_bf16, gout_rtn, cw, A_log, dt_bias, gnorm
                del H_gdn_cpu, abs_gdn_sum, gin_variants
            else:
                ap = P + "self_attn."
                wq = reader.read_tensor(ap + "q_proj.weight", dev, torch.float32)
                wk = reader.read_tensor(ap + "k_proj.weight", dev, torch.float32)
                wv = reader.read_tensor(ap + "v_proj.weight", dev, torch.float32)
                w_qkv = torch.cat([wq, wk, wv], dim=0)
                del wq, wk, wv
                w_ao = reader.read_tensor(ap + "o_proj.weight", dev, torch.float32)
                qn = reader.read_tensor(ap + "q_norm.weight", dev, torch.float32)
                kn = reader.read_tensor(ap + "k_norm.weight", dev, torch.float32)

                qkv_bf16, ao_bf16 = w_qkv.to(torch.bfloat16), w_ao.to(torch.bfloat16)
                qkv_rtn, ao_rtn = sq.fake_quant(w_qkv, "q6g64"), sq.fake_quant(w_ao, "q6g64")
                del w_qkv, w_ao

                for b0 in range(0, n_seqs, cal_mb):
                    b1 = min(b0 + cal_mb, n_seqs)
                    xb = cal_states_cpu[b0:b1].to(dev)
                    ub, win_b = hc_mix_down_up_batched(xb, *ahcd_bf16, ahcu_bf16, a_hn, inject=True)
                    ym = full_attn_forward_batched(ub, qkv_bf16, ao_bf16, qn, kn, cos_c, sin_c)
                    cal_states_cpu[b0:b1] = (xb.float() + win_b.unsqueeze(-1) * ym.float().unsqueeze(2)).to(torch.bfloat16).cpu()
                    del xb, ub, win_b, ym

                for name in state_names:
                    hd = ahcd_bf16 if name == "BF16" else ahcd_q8
                    hu = ahcu_bf16 if name == "BF16" else ahcu_q8
                    u_e, win_e = sq.hc_mix_down_up(states[name], *hd, hu, a_hn, inject=True)
                    ym = sq.full_attn_forward(
                        u_e,
                        qkv_bf16 if name == "BF16" else qkv_rtn,
                        ao_bf16 if name == "BF16" else ao_rtn,
                        qn, kn, cos_e, sin_e,
                    )
                    states[name] = (states[name].float() + win_e.unsqueeze(-1) * ym.float().unsqueeze(1)).to(torch.bfloat16)
                    del u_e, win_e, ym
                del qkv_bf16, ao_bf16, qkv_rtn, ao_rtn, qn, kn

            del a_wdf, a_wif, a_wu, a_hn, ahcd_bf16, ahcu_bf16, ahcd_q8, ahcu_q8

            # ---- MLP Hyper-Connection Mix ----
            mhc = P + "mlp_hyper_connection."
            m_wd = reader.read_tensor(mhc + "input_mix_weight_down.weight", dev, torch.float32)
            m_wi = reader.read_tensor(mhc + "block_inject_weight.weight", dev, torch.float32)
            m_wu = reader.read_tensor(mhc + "input_mix_weight_up.weight", dev, torch.float32)
            m_hn = reader.read_tensor(mhc + "hc_norm.weight", dev, torch.float32)
            m_wdf = m_wd * (1.0 + m_hn.unsqueeze(0))
            m_wif = m_wi * (1.0 + m_hn.unsqueeze(0))
            del m_wd, m_wi

            mhcd_bf16 = (m_wdf.to(torch.bfloat16), m_wif.to(torch.bfloat16))
            mhcu_bf16 = m_wu.to(torch.bfloat16)
            mhcd_q8 = (sq.fake_quant(m_wdf, "q8g64"), sq.fake_quant(m_wif, "q8g64"))
            mhcu_q8 = sq.fake_quant(m_wu, "q8g64")

            # Stream cal_states through mlp_hc to produce u_mlp_cal [T_cal, 2560], win_mlp_cal [T_cal, 4], H_mlp_cpu
            u_mlp_cal = torch.empty((T_cal, 2560), device=dev, dtype=torch.bfloat16)
            win_mlp_cal = torch.empty((T_cal, 4), device=dev, dtype=torch.float32)
            H_mlp_cpu = torch.zeros((2560, 2560), dtype=torch.float64)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                xb = cal_states_cpu[b0:b1].to(dev)
                ub, win_b = hc_mix_down_up_batched(xb, *mhcd_bf16, mhcu_bf16, m_hn, inject=True)
                u_flat = ub.reshape(-1, 2560)
                s0, s1 = b0 * seq_len, b1 * seq_len
                u_mlp_cal[s0:s1] = u_flat
                win_mlp_cal[s0:s1] = win_b.reshape(-1, 4)
                uf64 = u_flat.double()
                H_mlp_cpu.add_((uf64.T @ uf64).cpu())
                del xb, ub, win_b, u_flat, uf64

            H_mlp_cpu.div_(float(T_cal))
            U_mlp_gpu = sc.compute_hinv_chol_cpu(H_mlp_cpu, damp=exp_gu_damp).to(dev)
            del H_mlp_cpu

            # ---- Router & Shared Expert ----
            mp = P + "mlp."
            wg_r = reader.read_tensor(mp + "gate.weight", dev, torch.float32)
            wsg_r = reader.read_tensor(mp + "shared_expert_gate.weight", dev, torch.float32)
            w_router_all = torch.cat([wg_r, wsg_r], dim=0)
            del wg_r, wsg_r

            cal_route = sq.moe_route_qwen(u_mlp_cal, w_router_all.to(torch.bfloat16))
            # Pre-sort all T_cal * 10 routed expert assignments once for fast chunk slicing!
            tidx_flat = cal_route[0].reshape(-1)
            tw_flat = cal_route[1].reshape(-1)
            ord_all = torch.argsort(tidx_flat, stable=True)
            sorted_tok_all = ord_all // 10
            sorted_slot_all = ord_all % 10
            sorted_w_all = tw_flat[ord_all].unsqueeze(-1)
            counts_all = torch.bincount(tidx_flat, minlength=512)
            offsets_all = torch.zeros(513, device=dev, dtype=torch.long)
            torch.cumsum(counts_all, dim=0, out=offsets_all[1:])
            del tidx_flat, tw_flat, ord_all

            w_sh_g = reader.read_tensor(mp + "shared_expert.gate_proj.weight", dev, torch.float32)
            w_sh_u = reader.read_tensor(mp + "shared_expert.up_proj.weight", dev, torch.float32)
            w_sh_gu = torch.cat([w_sh_g, w_sh_u], dim=0)
            del w_sh_g, w_sh_u
            w_sh_d = reader.read_tensor(mp + "shared_expert.down_proj.weight", dev, torch.float32)

            # Compute BF16 calibration shared_out across all T_cal tokens
            sh_gu_c = F.linear(u_mlp_cal, w_sh_gu.to(torch.bfloat16))
            sh_h_c = F.silu(sh_gu_c[:, :640]) * sh_gu_c[:, 640:]
            sh_out_cal = F.linear(sh_h_c, w_sh_d.to(torch.bfloat16)).float() * cal_route[2].unsqueeze(-1)
            del sh_gu_c, sh_h_c

            gu_name = mp + "experts.gate_up_proj"
            dn_name = mp + "experts.down_proj"
            reader.info(gu_name)
            reader.info(dn_name)

            w_router_bf16 = w_router_all.to(torch.bfloat16)
            shgu_bf16, shdn_bf16 = w_sh_gu.to(torch.bfloat16), w_sh_d.to(torch.bfloat16)
            shgu_rtn, shdn_rtn = sq.fake_quant(w_sh_gu, "q6g64"), sq.fake_quant(w_sh_d, "q6g64")
            del w_router_all, w_sh_gu, w_sh_d

            # Compute honest held-out u_mlp, win_mlp, router_outputs, shared_outs per variant (NO AWQ router distortion!)
            u_mlp, win_mlp, routes, sh_outs = {}, {}, {}, {}
            for name in state_names:
                hd = mhcd_bf16 if name == "BF16" else mhcd_q8
                hu = mhcu_bf16 if name == "BF16" else mhcu_q8
                u_m, wm = sq.hc_mix_down_up(states[name], *hd, hu, m_hn, inject=True)
                u_mlp[name] = u_m
                win_mlp[name] = wm
                routes[name] = sq.moe_route_qwen(u_m, w_router_bf16)
                wgu_s = shgu_bf16 if name == "BF16" else shgu_rtn
                wdn_s = shdn_bf16 if name == "BF16" else shdn_rtn
                g_s = F.linear(u_m, wgu_s)
                h_s = F.silu(g_s[:, :640]) * g_s[:, 640:]
                sh_outs[name] = F.linear(h_s, wdn_s).float() * routes[name][2].unsqueeze(-1)
                del g_s, h_s

            del m_wdf, m_wif, m_wu, m_hn, mhcd_bf16, mhcu_bf16, mhcd_q8, mhcu_q8
            del w_router_bf16, shgu_bf16, shdn_bf16, shgu_rtn, shdn_rtn

            # ---- Stream Routed Experts in 64-Expert Chunks ----
            cal_routed = torch.zeros((T_cal, 2560), device=dev, dtype=torch.float32)
            routed_acc = {name: torch.zeros((T_eval, 2560), device=dev, dtype=torch.float32) for name in state_names}
            slot_buf = torch.zeros((T_eval, 10, 2560), device=dev, dtype=torch.float32)

            def _load_chunk_cpu(c_start, c_cnt):
                return (
                    reader.read_slice_dim0(gu_name, c_start, c_cnt, cpu, torch.bfloat16),
                    reader.read_slice_dim0(dn_name, c_start, c_cnt, cpu, torch.bfloat16),
                )

            CHUNK = expert_chunk
            next_fut = prefetch_pool.submit(_load_chunk_cpu, 0, min(CHUNK, 512))

            dn_alphas_all = []
            dn_ratios_all = []
            gptq_dn_elig_tot = 0
            max_h_sm_layer = 0.0

            def gu_key(name):
                if name == "BF16": return "bf16"
                if name in ("0_RTN_Prod", "2_GPTQ_RTNGrid_GdnIn"): return "rtn"
                if name == "1_AWQ_ExpDn_Only_RTN": return "awq_rtn"
                if name in ("3_GPTQ_RTNGrid_Gdn_ExpGU", "4_GPTQ_All3_RTNGrid"): return "gptq"
                return "gptq_awq"

            def dn_key(name):
                if name == "BF16": return "bf16"
                if name in ("0_RTN_Prod", "2_GPTQ_RTNGrid_GdnIn", "3_GPTQ_RTNGrid_Gdn_ExpGU"): return "rtn"
                if name == "1_AWQ_ExpDn_Only_RTN": return "awq_rtn"
                if name == "4_GPTQ_All3_RTNGrid": return "gptq"
                return "awq_gptq"

            for c0 in range(0, 512, CHUNK):
                cnt = min(CHUNK, 512 - c0)
                c1 = c0 + cnt
                gu_cpu, dn_cpu = next_fut.result()
                c_next = c1
                if c_next < 512:
                    next_fut = prefetch_pool.submit(_load_chunk_cpu, c_next, min(CHUNK, 512 - c_next))

                gu_bf16 = gu_cpu.to(dev)
                dn_bf16 = dn_cpu.to(dev)
                del gu_cpu, dn_cpu

                # Step A: BF16 cal forward through [c0:c1] + collect per-expert H_dn [cnt, 640, 640] & a_dn [cnt, 640]
                H_dn_chunk = torch.zeros((cnt, 640, 640), device=dev, dtype=torch.float32)
                a_dn_chunk = torch.zeros((cnt, 640), device=dev, dtype=torch.float32)
                hits_chunk = counts_all[c0:c1]

                h_max_chunk = torch.zeros((cnt, 640), device=dev, dtype=torch.float32)
                for le_i in range(cnt):
                    ge = c0 + le_i
                    p0 = int(offsets_all[ge].item())
                    p1 = int(offsets_all[ge + 1].item())
                    c_n = p1 - p0
                    if c_n <= 0:
                        continue
                    xs = u_mlp_cal[sorted_tok_all[p0:p1]]
                    gue = F.linear(xs, gu_bf16[le_i])
                    hs = F.silu(gue[:, :640]) * gue[:, 640:]
                    hs_f = hs.float()
                    a_dn_chunk[le_i] = hs_f.abs().mean(dim=0)
                    h_max_chunk[le_i] = hs_f.abs().amax(dim=0)
                    H_dn_chunk[le_i] = (hs_f.T @ hs_f) / float(c_n)

                    ye = F.linear(hs, dn_bf16[le_i]).float() * sorted_w_all[p0:p1]
                    cal_routed.index_add_(0, sorted_tok_all[p0:p1], ye)
                    del xs, gue, hs, hs_f, ye

                # Empirical shrinkage of act_abs toward chunk active-expert mean (tau=64.0)
                act_mask = hits_chunk > 0
                if act_mask.any():
                    a_prior = a_dn_chunk[act_mask].mean(dim=0, keepdim=True)
                else:
                    a_prior = torch.ones((1, 640), device=dev, dtype=torch.float32)
                hf_c = hits_chunk.float().unsqueeze(-1)
                a_eff = (hf_c * a_dn_chunk + 64.0 * a_prior) / (hf_c + 64.0)

                dn_f = dn_bf16.float()
                s_dn, a_dn_sel, mr_dn = search_awq_exp_down_batched(dn_f, H_dn_chunk, a_eff, hits_chunk, fmt="q4g128", alphas=ALPHAS)
                dn_alphas_all.extend(a_dn_sel[hits_chunk >= 64].tolist())
                dn_ratios_all.append(mr_dn)
                max_h_sm_chunk = float((h_max_chunk / s_dn).amax().item()) if act_mask.any() else 0.0
                if max_h_sm_chunk > max_h_sm_layer:
                    max_h_sm_layer = max_h_sm_chunk
                del a_dn_chunk, h_max_chunk, a_prior, a_eff, hf_c

                # Pure RTN-grid exp_down variants for [c0:c1] (NO MSEClip!)
                dn_rtn = sq.fake_quant(dn_bf16, "q4g128")
                dn_awq = dn_f * s_dn.unsqueeze(1)
                dn_awq_rtn = sq.fake_quant(dn_awq, "q4g128")
                dn_gptq, n_el = quant_gptq_batched_exp_down_rtn(
                    dn_f, H_dn_chunk, None, hits_chunk, dn_rtn,
                    min_hits=dn_min_hits, base_damp=dn_damp, fmt="q4g128",
                )
                dn_awq_gptq, _ = quant_gptq_batched_exp_down_rtn(
                    dn_awq, H_dn_chunk, s_dn, hits_chunk, dn_awq_rtn,
                    min_hits=dn_min_hits, base_damp=dn_damp, fmt="q4g128",
                )
                gptq_dn_elig_tot += n_el
                q_dn = {
                    "bf16": dn_bf16,
                    "rtn": dn_rtn,
                    "awq_rtn": dn_awq_rtn,
                    "gptq": dn_gptq,
                    "awq_gptq": dn_awq_gptq,
                }
                del dn_bf16, dn_f, dn_awq, dn_rtn, dn_awq_rtn, dn_gptq, dn_awq_gptq, H_dn_chunk

                # Pure RTN-grid + shared-H_mlp GPTQ exp_gu variants (with optional 1/s_dn W_up row fold)
                gu_awq_dn = gu_bf16.float()
                gu_awq_dn[:, 640:, :] /= s_dn.unsqueeze(-1)
                q_gu = {
                    "bf16": gu_bf16,
                    "rtn": sq.fake_quant(gu_bf16, "q4g128"),
                    "awq_rtn": sq.fake_quant(gu_awq_dn, "q4g128"),
                    "gptq": quant_gptq_rtn_grid(gu_bf16, U_gpu=U_mlp_gpu, fmt="q4g128", row_subbatch=40960),
                    "gptq_awq": quant_gptq_rtn_grid(gu_awq_dn, U_gpu=U_mlp_gpu, fmt="q4g128", row_subbatch=40960),
                }
                del gu_bf16, gu_awq_dn, s_dn

                # Step B: Held-out eval forward across all variants on chunk [c0:c1]
                for name in state_names:
                    tidx, tw, _ = routes[name]
                    in_chunk = (tidx >= c0) & (tidx < c1)
                    if not in_chunk.any():
                        continue
                    tok_pos, slot_pos = torch.nonzero(in_chunk, as_tuple=True)
                    loc_e = tidx[tok_pos, slot_pos] - c0
                    order = torch.argsort(loc_e, stable=True)
                    tok_sorted = tok_pos[order]
                    slot_sorted = slot_pos[order]
                    w_sorted = tw[tok_pos, slot_pos][order].unsqueeze(-1)
                    uniq_e, counts_e = torch.unique_consecutive(loc_e[order], return_counts=True)

                    wgu_c = q_gu[gu_key(name)]
                    wdn_c = q_dn[dn_key(name)]
                    x_hits = u_mlp[name][tok_sorted]
                    outs = torch.empty_like(x_hits)
                    pos = 0
                    for le_i, cnt_i in zip(uniq_e.tolist(), counts_e.tolist()):
                        xs = x_hits[pos : pos + cnt_i]
                        gue = F.linear(xs, wgu_c[le_i])
                        hs = F.silu(gue[:, :640]) * gue[:, 640:]
                        outs[pos : pos + cnt_i] = F.linear(hs, wdn_c[le_i])
                        pos += cnt_i

                    slot_buf.zero_()
                    slot_buf[tok_sorted, slot_sorted] = outs.float() * w_sorted
                    routed_acc[name].add_(slot_buf.sum(dim=1))

                del q_gu, q_dn

            del slot_buf, sorted_tok_all, sorted_slot_all, sorted_w_all, offsets_all

            # Advance calibration states cleanly back into CPU buffer
            cal_moe = (cal_routed + sh_out_cal).to(torch.bfloat16)
            for b0 in range(0, n_seqs, cal_mb):
                b1 = min(b0 + cal_mb, n_seqs)
                s0, s1 = b0 * seq_len, b1 * seq_len
                xb = cal_states_cpu[b0:b1].to(dev)
                ym = cal_moe[s0:s1].view(b1 - b0, seq_len, 1, 2560).float()
                wb = win_mlp_cal[s0:s1].view(b1 - b0, seq_len, 4, 1)
                cal_states_cpu[b0:b1] = (xb.float() + wb * ym).to(torch.bfloat16).cpu()
                del xb, ym, wb

            for name in state_names:
                y_moe = (routed_acc[name] + sh_outs[name]).to(torch.bfloat16)
                states[name] = (states[name].float() + win_mlp[name].unsqueeze(-1) * y_moe.float().unsqueeze(1)).to(torch.bfloat16)

            hits_list = counts_all.tolist()
            hits_sorted = sorted(hits_list)
            lt["alpha_dn_mean"] = round(statistics.mean(dn_alphas_all), 3) if dn_alphas_all else 0.0
            lt["dn_awq_mse_ratio"] = round(statistics.mean(dn_ratios_all), 4) if dn_ratios_all else 1.0
            lt["hits_min"] = hits_sorted[0]
            lt["hits_med"] = hits_sorted[256]
            lt["hits_max"] = hits_sorted[-1]
            lt["dn_gptq_elig"] = gptq_dn_elig_tot
            lt["max_h_smooth"] = round(max_h_sm_layer, 2)
            layer_telemetry.append(lt)

            del u_mlp_cal, win_mlp_cal, cal_route, counts_all, sh_out_cal, cal_routed, cal_moe, U_mlp_gpu
            del u_mlp, win_mlp, routes, routed_acc, sh_outs

            if L in (0, 1, 2, 3, 15, 31, 47):
                peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                print(
                    f"  [Layer {L:2d}/47] {'Attn' if is_attn else 'GDN '} t={time.time()-tl:4.1f}s ({time.time()-t0:6.1f}s) | "
                    f"a_dn={lt['alpha_dn_mean']:.2f}({lt['dn_awq_mse_ratio']:.2f}x) | "
                    f"hits[{lt['hits_min']}/{lt['hits_med']}/{lt['hits_max']}] elig={gptq_dn_elig_tot}/512 max|h|={lt['max_h_smooth']:.1f} VRAM={peak_gb:.2f}G",
                    flush=True,
                )

        # ---- Final HC Mixer (q8g64) + LM Head (q8g64) Evaluation on Held-Out CALIB_DOC ----
        del cal_states_cpu
        fhc = lm + "hyper_connection_mixer."
        f_wd = reader.read_tensor(fhc + "input_mix_weight_down.weight", dev, torch.float32)
        f_wu = reader.read_tensor(fhc + "input_mix_weight_up.weight", dev, torch.float32)
        f_hn = reader.read_tensor(fhc + "hc_norm.weight", dev, torch.float32)
        f_wdf = f_wd * (1.0 + f_hn.unsqueeze(0))
        del f_wd
        dummy_inj = torch.zeros((4, 10240), device=dev, dtype=torch.float32)

        lm_w = reader.read_tensor("lm_head.weight", dev, torch.bfloat16)
        f_bf16_u, _ = sq.hc_mix_down_up(states["BF16"], f_wdf.to(torch.bfloat16), dummy_inj, f_wu.to(torch.bfloat16), f_hn, inject=False)

        n_eval = T_eval - 1
        TOK_CHUNK = 256
        bf16_nll_sum = 0.0
        for s0 in range(0, n_eval, TOK_CHUNK):
            s1 = min(s0 + TOK_CHUNK, n_eval)
            lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
            bf16_nll_sum += F.nll_loss(F.log_softmax(lgt, dim=-1), target_eval[s0:s1], reduction="sum").item()
            del lgt
        bf16_ppl = math.exp(bf16_nll_sum / float(n_eval))

        qlm_q8 = sq.fake_quant_rows(lm_w, "q8g64")
        fhcd_q8 = sq.fake_quant(f_wdf, "q8g64")
        fhcu_q8 = sq.fake_quant(f_wu, "q8g64")

        def eval_u(u_cand):
            kld_parts = []
            nll_sum = 0.0
            t1_hits = 0
            t5_hits = 0
            for s0 in range(0, n_eval, TOK_CHUNK):
                s1 = min(s0 + TOK_CHUNK, n_eval)
                ref_lgt = F.linear(f_bf16_u[s0:s1], lm_w).float()
                ref_lp = F.log_softmax(ref_lgt, dim=-1)
                ref_p = ref_lp.exp()
                q_lgt = F.linear(u_cand[s0:s1], qlm_q8).float()
                q_lp = F.log_softmax(q_lgt, dim=-1)
                kld_parts.append((ref_p * (ref_lp - q_lp)).sum(dim=-1))
                nll_sum += F.nll_loss(q_lp, target_eval[s0:s1], reduction="sum").item()
                ref_t1 = ref_lgt.argmax(dim=-1)
                t1_hits += (q_lgt.argmax(dim=-1) == ref_t1).sum().item()
                t5_hits += (q_lgt.topk(5, dim=-1).indices == ref_t1.unsqueeze(-1)).any(dim=-1).sum().item()
                del ref_lgt, ref_lp, ref_p, q_lgt, q_lp
            kld_v = torch.cat(kld_parts, dim=0)
            return (
                kld_v.mean().item(),
                torch.quantile(kld_v, 0.50).item(),
                torch.quantile(kld_v, 0.95).item(),
                kld_v.max().item(),
                math.exp(nll_sum / float(n_eval)),
                100.0 * t1_hits / float(n_eval),
                100.0 * t5_hits / float(n_eval),
            )

        print(f"\n=== HELD-OUT EVALUATION (Thinking-Cap, T_eval={T_eval}, T_cal={T_cal}, BF16 PPL={bf16_ppl:.3f}) ===")
        print(f"{'Variant':28s} | {'Mean KLD':9s} | {'vs RTN':8s} | {'Med KLD':9s} | {'p95 KLD':9s} | {'Max KLD':9s} | {'PPL':7s} | {'dPPL':7s} | {'Top1%':6s} | {'Top5%':6s}")
        print("-" * 116)
        print(f"{'BF16 Baseline':28s} | {'0.000000':9s} | {'---':8s} | {'0.000000':9s} | {'0.000000':9s} | {'0.000000':9s} | {bf16_ppl:7.3f} | {'+0.000':7s} | {'100.0%':6s} | {'100.0%':6s}")

        out_rows = []
        rtn_kld = None
        for name in VARIANTS:
            u_fin, _ = sq.hc_mix_down_up(states[name], fhcd_q8, dummy_inj, fhcu_q8, f_hn, inject=False)
            mk, medk, p95k, maxk, ppl, t1, t5 = eval_u(u_fin)
            del u_fin
            if name == "0_RTN_Prod":
                rtn_kld = mk
            rel_pct = (mk / rtn_kld - 1.0) * 100.0
            rel_s = f"{rel_pct:+6.1f}%"
            dp = ppl - bf16_ppl
            print(f"{name:28s} | {mk:9.6f} | {rel_s:8s} | {medk:9.6f} | {p95k:9.6f} | {maxk:9.6f} | {ppl:7.3f} | {dp:+7.3f} | {t1:5.1f}% | {t5:5.1f}%")
            out_rows.append({
                "variant": name,
                "mean_kld": mk,
                "delta_vs_rtn_pct": rel_pct,
                "median_kld": medk,
                "p95_kld": p95k,
                "max_kld": maxk,
                "ppl": ppl,
                "dppl": dp,
                "top1_pct": t1,
                "top5_pct": t5,
            })

        peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        elapsed = time.time() - t0
        max_h_all = max(lt["max_h_smooth"] for lt in layer_telemetry)
        print(f"\nElapsed: {elapsed:.1f}s | Peak VRAM: {peak_gb:.2f} GiB | Global max(|h_smooth|): {max_h_all:.2f}")

    prefetch_pool.shutdown()
    reader.close()
    return {
        "model": model_dir,
        "n_seqs": n_seqs,
        "seq_len": seq_len,
        "t_cal": T_cal,
        "calib_sha256_16": cal_sha,
        "domains": dom_summary,
        "t_eval": T_eval,
        "bf16_ppl": bf16_ppl,
        "peak_vram_gib": peak_gb,
        "elapsed_s": elapsed,
        "gdn_damp": gdn_damp,
        "exp_gu_damp": exp_gu_damp,
        "dn_damp": dn_damp,
        "dn_min_hits": dn_min_hits,
        "max_h_smooth_global": max_h_all,
        "variants": out_rows,
        "layer_telemetry": layer_telemetry,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/nvme2/Thinking-Cap-Qwen-3.8-FN")
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=1536)
    ap.add_argument("--cal-mb", type=int, default=4)
    ap.add_argument("--expert-chunk", type=int, default=64)
    ap.add_argument("--gdn-damp", type=float, default=0.01)
    ap.add_argument("--exp-gu-damp", type=float, default=0.05)
    ap.add_argument("--dn-damp", type=float, default=0.08)
    ap.add_argument("--dn-min-hits", type=int, default=640)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = run_awq_gptq_sim(
        args.model,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        cal_mb=args.cal_mb,
        expert_chunk=args.expert_chunk,
        gdn_damp=args.gdn_damp,
        exp_gu_damp=args.exp_gu_damp,
        dn_damp=args.dn_damp,
        dn_min_hits=args.dn_min_hits,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
