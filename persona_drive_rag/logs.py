"""
logs.py - Real-time SLURM log viewer with SSE streaming
=============================================================
Usage:
    pip install flask
    python logs.py --log-dir slurm_logs --port 8787

Then on your laptop:
    ssh -L 8787:localhost:8787 <user>@<cluster-login-host>

Open browser: http://localhost:8787
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, Response, jsonify, render_template_string, request

app     = Flask(__name__)
LOG_DIR = Path("slurm_logs")

# ---------------------------------------------------------------------------
# HTML + JS
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RAG Pipeline — Live Logs</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:     #0b0d12;
  --bg2:    #11141c;
  --bg3:    #181c27;
  --border: #1e2333;
  --text:   #c9d1e0;
  --muted:  #4a546a;
  --accent: #3d7eff;
  --done:   #00e5a0;
  --error:  #ff4f6a;
  --warn:   #ffb547;
  --purple: #a78bfa;
  --cyan:   #22d3ee;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; }
body { display: grid; grid-template-columns: 300px 1fr; grid-template-rows: 48px 1fr; }

/* ── Top bar ── */
#topbar {
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 16px;
  padding: 0 20px; border-bottom: 1px solid var(--border); background: var(--bg2);
}
#topbar h1 { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 700; letter-spacing: 0.08em; white-space: nowrap; }
#topbar h1 span { color: var(--accent); }
.pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--done); animation: blink 2s infinite; flex-shrink: 0; }
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.7)} }
#summary-bar { display: flex; gap: 16px; margin-left: auto; font-size: 11px; color: var(--muted); }
.sum-item { white-space: nowrap; }
.sum-item strong { font-weight: 700; }
.sum-done    strong { color: var(--done); }
.sum-running strong { color: var(--accent); }
.sum-error   strong { color: var(--error); }
.sum-records strong { color: var(--purple); }

/* ── Sidebar ── */
#sidebar { border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--bg2); overflow: hidden; }

/* Search */
#search-wrap { padding: 8px 12px; border-bottom: 1px solid var(--border); }
#search {
  width: 100%; background: var(--bg3); border: 1px solid var(--border);
  color: var(--text); padding: 5px 10px; border-radius: 4px;
  font-size: 11px; font-family: inherit; outline: none;
}
#search:focus { border-color: var(--accent); }

/* Filter tabs */
#filter-bar {
  display: flex; border-bottom: 1px solid var(--border);
  overflow-x: auto; flex-shrink: 0;
}
#filter-bar::-webkit-scrollbar { display: none; }
.filter-btn {
  flex: 1; padding: 6px 4px; font-size: 9px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.05em;
  background: transparent; border: none; border-bottom: 2px solid transparent;
  color: var(--muted); cursor: pointer; font-family: inherit;
  transition: all .12s; white-space: nowrap;
}
.filter-btn:hover { color: var(--text); }
.filter-btn.active          { color: var(--text);   border-bottom-color: var(--text); }
.filter-btn.active.f-all     { color: var(--text);   border-bottom-color: var(--muted); }
.filter-btn.active.f-running { color: var(--accent); border-bottom-color: var(--accent); }
.filter-btn.active.f-done    { color: var(--done);   border-bottom-color: var(--done); }
.filter-btn.active.f-error   { color: var(--error);  border-bottom-color: var(--error); }
.filter-btn.active.f-pending { color: var(--muted);  border-bottom-color: var(--muted); }
.filter-count { opacity: .6; margin-left: 3px; }

/* Sort bar */
#sort-bar {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-bottom: 1px solid var(--border);
  font-size: 9px; color: var(--muted);
}
#sort-bar span { flex: 1; }
.sort-btn {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  padding: 2px 8px; border-radius: 3px; font-size: 9px; cursor: pointer;
  font-family: inherit; transition: all .1s;
}
.sort-btn:hover { border-color: var(--text); color: var(--text); }
.sort-btn.active { border-color: var(--accent); color: var(--accent); }

/* Job list */
#job-list { overflow-y: auto; flex: 1; }
#job-list::-webkit-scrollbar { width: 4px; }
#job-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.job-item {
  padding: 8px 12px; cursor: pointer;
  border-bottom: 1px solid var(--border);
  border-left: 3px solid transparent;
  transition: background .12s, border-color .12s;
}
.job-item:hover { background: var(--bg3); }
.job-item.active { background: var(--bg3); border-left-color: var(--accent); }
.job-name { font-size: 10.5px; color: var(--text); word-break: break-all; line-height: 1.4; }
.job-time { font-size: 9px; color: var(--muted); margin-top: 2px; }
.job-meta { display: flex; gap: 5px; align-items: center; margin-top: 4px; flex-wrap: wrap; }

.badge {
  padding: 1px 6px; border-radius: 3px;
  font-size: 9px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.04em; cursor: default;
}
.badge-done    { background: rgba(0,229,160,.15);  color: var(--done); }
.badge-running { background: rgba(61,126,255,.15); color: var(--accent); }
.badge-error   { background: rgba(255,79,106,.15); color: var(--error); }
.badge-pending { background: rgba(74,84,106,.2);   color: var(--muted); }

.badge-err-btn {
  background: rgba(255,79,106,.15); color: var(--error);
  padding: 1px 6px; border-radius: 3px;
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  cursor: pointer; border: 1px solid rgba(255,79,106,.3);
  font-family: inherit; transition: background .1s;
}
.badge-err-btn:hover { background: rgba(255,79,106,.3); }
.job-rec  { font-size: 9px; color: var(--purple); }
.job-size { font-size: 9px; color: var(--muted); }

#no-results {
  padding: 24px 16px; text-align: center;
  font-size: 11px; color: var(--muted); display: none;
}

/* ── Main pane ── */
#main { display: flex; flex-direction: column; overflow: hidden; }
#tab-stripe { height: 2px; background: var(--border); transition: background .2s; }
#tab-stripe.out-active { background: var(--accent); }
#tab-stripe.err-active { background: var(--error); }

#log-header {
  padding: 0 16px; border-bottom: 1px solid var(--border);
  background: var(--bg2); display: flex; align-items: center; gap: 8px;
  height: 40px; min-height: 40px;
}
#log-title { font-size: 11px; color: var(--muted); flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#line-count { font-size: 9px; color: var(--muted); white-space: nowrap; }

.tab-btn {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  padding: 3px 12px; border-radius: 3px; font-size: 10px; cursor: pointer;
  font-family: inherit; transition: all .12s; white-space: nowrap;
}
.tab-btn:hover { border-color: var(--text); color: var(--text); }
.tab-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(61,126,255,.08); }
.tab-btn.has-err { border-color: rgba(255,79,106,.4); }
.tab-btn.has-err.active { border-color: var(--error); color: var(--error); background: rgba(255,79,106,.08); }

.ctrl-btn {
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  padding: 3px 10px; border-radius: 3px; font-size: 10px; cursor: pointer;
  font-family: inherit; transition: all .12s;
}
.ctrl-btn:hover { border-color: var(--text); color: var(--text); }
.ctrl-btn.on { border-color: var(--done); color: var(--done); }

#stream-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); flex-shrink: 0; transition: background .3s; }
#stream-dot.live { background: var(--done); animation: blink 1.5s infinite; }
#stream-dot.err  { background: var(--error); animation: blink 1.5s infinite; }

/* ── Log content ── */
#log-wrap { flex: 1; overflow-y: auto; padding: 12px 16px; }
#log-wrap::-webkit-scrollbar { width: 4px; }
#log-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
#log-content { font-size: 11.5px; line-height: 1.7; }

.ll { white-space: pre-wrap; word-break: break-all; }
.ll-err  { color: var(--error); }
.ll-warn { color: var(--warn); }
.ll-done { color: var(--done); font-weight: 700; }
.ll-prog { color: var(--cyan); }
.ll-sub  { color: var(--purple); }
.ll-norm { color: var(--text); opacity: .75; }

#empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; height: 100%; color: var(--muted); gap: 8px;
}
#empty-state .arrow { font-size: 24px; color: var(--border); }
#empty-state p { font-size: 12px; }
</style>
</head>
<body>

<!-- Top bar -->
<div id="topbar">
  <div class="pulse"></div>
  <h1>RAG Pipeline <span>//</span> Live Logs</h1>
  <div id="summary-bar">
    <div class="sum-item">total <strong id="s-total">—</strong></div>
    <div class="sum-item sum-done">done <strong id="s-done">—</strong></div>
    <div class="sum-item sum-running">running <strong id="s-running">—</strong></div>
    <div class="sum-item sum-error">errors <strong id="s-error">—</strong></div>
    <div class="sum-item sum-records">records <strong id="s-records">—</strong></div>
  </div>
</div>

<!-- Sidebar -->
<div id="sidebar">

  <!-- Search -->
  <div id="search-wrap">
    <input id="search" type="text" placeholder="search jobs..." oninput="renderList()">
  </div>

  <!-- Filter tabs -->
  <div id="filter-bar">
    <button class="filter-btn f-all active"     onclick="setFilter('all')">All<span class="filter-count" id="fc-all"></span></button>
    <button class="filter-btn f-running"         onclick="setFilter('running')">Running<span class="filter-count" id="fc-running"></span></button>
    <button class="filter-btn f-done"            onclick="setFilter('done')">Done<span class="filter-count" id="fc-done"></span></button>
    <button class="filter-btn f-error"           onclick="setFilter('error')">Error<span class="filter-count" id="fc-error"></span></button>
    <button class="filter-btn f-pending"         onclick="setFilter('pending')">Pending<span class="filter-count" id="fc-pending"></span></button>
  </div>

  <!-- Sort bar -->
  <div id="sort-bar">
    <span>sort by</span>
    <button class="sort-btn active" id="sort-time" onclick="setSort('time')">time ↓</button>
    <button class="sort-btn"        id="sort-name" onclick="setSort('name')">name</button>
    <button class="sort-btn"        id="sort-rec"  onclick="setSort('records')">records</button>
  </div>

  <div id="job-list">
    <div id="no-results">No jobs match filter</div>
  </div>
</div>

<!-- Main -->
<div id="main">
  <div id="tab-stripe"></div>
  <div id="log-header">
    <div id="stream-dot"></div>
    <div id="log-title">no job selected</div>
    <span id="line-count"></span>
    <button class="tab-btn active" id="btn-out" onclick="switchTab('out')">stdout</button>
    <button class="tab-btn"        id="btn-err" onclick="switchTab('err')">stderr</button>
    <button class="ctrl-btn on"    id="btn-scroll" onclick="toggleScroll()">scroll ↓</button>
    <button class="ctrl-btn"       onclick="clearLog()">clear</button>
  </div>
  <div id="log-wrap">
    <div id="log-content">
      <div id="empty-state">
        <div class="arrow">←</div>
        <p>select a job to stream logs</p>
      </div>
    </div>
  </div>
</div>

<script>
let currentJob = null;
let currentTab = 'out';
let autoScroll = true;
let lineCount  = 0;
let activeES   = null;
let allJobs    = [];   // full list from server
let jobMeta    = {};   // name → job info
let activeFilter = 'all';
let activeSort   = 'time';

// ── Line colouring ────────────────────────────────────────────────────────
function cls(line) {
  if (/error|Error|exception|Exception|Traceback|FAILED/i.test(line) && !/0 errors/i.test(line)) return 'll-err';
  if (/warning|Warning/i.test(line)) return 'll-warn';
  if (/DONE\s*\|/i.test(line))       return 'll-done';
  if (/routes\s*\|/i.test(line))     return 'll-prog';
  if (/Submitting/i.test(line))      return 'll-sub';
  return 'll-norm';
}

function appendLine(line) {
  const el = document.getElementById('log-content');
  const es = document.getElementById('empty-state');
  if (es) es.remove();
  const div = document.createElement('div');
  div.className = `ll ${cls(line)}`;
  div.textContent = line;
  el.appendChild(div);
  lineCount++;
  document.getElementById('line-count').textContent = lineCount + ' lines';
  if (autoScroll) document.getElementById('log-wrap').scrollTop = 999999;
}

// ── SSE ───────────────────────────────────────────────────────────────────
function startStream(name, tab) {
  if (activeES) { activeES.close(); activeES = null; }
  const dot = document.getElementById('stream-dot');
  dot.className = tab === 'err' ? 'err' : 'live';
  const es = new EventSource(`/stream?name=${encodeURIComponent(name)}&tab=${tab}&tail=2000`);
  activeES = es;
  es.onmessage = e => {
    const data = JSON.parse(e.data);
    if (data.line !== undefined) appendLine(data.line);
  };
  es.onerror = () => {
    dot.className = '';
    setTimeout(() => { if (currentJob === name && currentTab === tab) startStream(name, tab); }, 3000);
  };
}

// ── Tab switching ─────────────────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  document.getElementById('btn-out').classList.toggle('active', tab === 'out');
  document.getElementById('btn-err').classList.toggle('active', tab === 'err');
  document.getElementById('tab-stripe').className = tab === 'out' ? 'out-active' : 'err-active';
  if (currentJob) {
    lineCount = 0;
    document.getElementById('log-content').innerHTML = '';
    document.getElementById('line-count').textContent = '';
    startStream(currentJob, tab);
  }
}

// ── Job selection ─────────────────────────────────────────────────────────
function selectJob(name, forceTab) {
  currentJob = name;
  lineCount  = 0;
  document.getElementById('log-content').innerHTML = '';
  document.getElementById('line-count').textContent = '';
  document.getElementById('log-title').textContent = name;
  document.querySelectorAll('.job-item').forEach(el =>
    el.classList.toggle('active', el.dataset.name === name));

  const meta   = jobMeta[name] || {};
  const btnErr = document.getElementById('btn-err');
  btnErr.textContent = meta.has_err ? `stderr (${meta.err_size})` : 'stderr';
  btnErr.classList.toggle('has-err', !!meta.has_err);

  if (forceTab) {
    currentTab = forceTab;
    document.getElementById('btn-out').classList.toggle('active', forceTab === 'out');
    btnErr.classList.toggle('active', forceTab === 'err');
    document.getElementById('tab-stripe').className = forceTab === 'out' ? 'out-active' : 'err-active';
  }
  startStream(name, currentTab);
}

// ── Filter & sort ─────────────────────────────────────────────────────────
function setFilter(f) {
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.querySelector(`.f-${f}`).classList.add('active');
  renderList();
}

function setSort(s) {
  activeSort = s;
  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`sort-${s}`).classList.add('active');
  renderList();
}

function renderList() {
  const q = document.getElementById('search').value.toLowerCase();

  // Filter
  let jobs = allJobs.filter(j => {
    const matchFilter = activeFilter === 'all' || j.status === activeFilter;
    const matchSearch = j.name.toLowerCase().includes(q);
    return matchFilter && matchSearch;
  });

  // Sort
  jobs.sort((a, b) => {
    if (activeSort === 'time')    return b.submitted_ts - a.submitted_ts;  // newest first
    if (activeSort === 'name')    return a.name.localeCompare(b.name);
    if (activeSort === 'records') return b.records - a.records;
    return 0;
  });

  const list = document.getElementById('job-list');
  const noResults = document.getElementById('no-results');

  if (jobs.length === 0) {
    list.innerHTML = '';
    list.appendChild(noResults);
    noResults.style.display = 'block';
    return;
  }
  noResults.style.display = 'none';

  list.innerHTML = jobs.map(j => `
    <div class="job-item${j.name === currentJob ? ' active' : ''}"
         data-name="${j.name}"
         onclick="selectJob('${j.name}')">
      <div class="job-name">${j.name}</div>
      ${j.submitted ? `<div class="job-time">${j.submitted}</div>` : ''}
      <div class="job-meta">
        <span class="badge badge-${j.status}">${j.status}</span>
        ${j.records > 0 ? `<span class="job-rec">${j.records} rec</span>` : ''}
        <span class="job-size">out: ${j.size}</span>
        ${j.has_err
          ? `<button class="badge-err-btn"
               title="Click to view stderr (${j.err_size})"
               onclick="event.stopPropagation(); selectJob('${j.name}', 'err')">
               err ${j.err_size}
             </button>`
          : ''}
      </div>
    </div>`).join('') + '<div id="no-results" style="display:none"></div>';
}

function updateCounts(data) {
  const counts = { all: data.total, running: 0, done: 0, error: 0, pending: 0 };
  data.jobs.forEach(j => { if (counts[j.status] !== undefined) counts[j.status]++; });
  ['all','running','done','error','pending'].forEach(k => {
    const el = document.getElementById(`fc-${k}`);
    if (el) el.textContent = counts[k] > 0 ? ` ${counts[k]}` : '';
  });
}

// ── Sidebar polling ───────────────────────────────────────────────────────
function refreshJobs() {
  fetch('/jobs').then(r => r.json()).then(data => {
    allJobs = data.jobs;
    data.jobs.forEach(j => { jobMeta[j.name] = j; });

    updateCounts(data);
    renderList();

    document.getElementById('s-total').textContent   = data.total;
    document.getElementById('s-done').textContent    = data.done;
    document.getElementById('s-running').textContent = data.running;
    document.getElementById('s-error').textContent   = data.errors;
    document.getElementById('s-records').textContent =
      data.total_records > 9999
        ? (data.total_records / 1000).toFixed(1) + 'k'
        : data.total_records;

    if (currentJob && jobMeta[currentJob]) {
      const meta   = jobMeta[currentJob];
      const btnErr = document.getElementById('btn-err');
      btnErr.textContent = meta.has_err ? `stderr (${meta.err_size})` : 'stderr';
      btnErr.classList.toggle('has-err', !!meta.has_err);
    }
  });
}

// ── Scroll & clear controls ───────────────────────────────────────────────
function toggleScroll() {
  autoScroll = !autoScroll;
  document.getElementById('btn-scroll').classList.toggle('on', autoScroll);
  document.getElementById('btn-scroll').textContent = autoScroll ? 'scroll ↓' : 'scroll ↓';
}

function clearLog() {
  document.getElementById('log-content').innerHTML = '';
  lineCount = 0;
  document.getElementById('line-count').textContent = '';
}

refreshJobs();
setInterval(refreshJobs, 5000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Matches SLURM job header line: "Start Time: Mon Apr 13 02:00:22 PM PDT 2026"
_START_TIME_RE = re.compile(r"Start Time:\s+(.+)")
_DONE_RE = re.compile(r"DONE\s*\|\s*records:\s*(\d+)")


def _fmt_size(n: int) -> str:
    return f"{n // 1024}KB" if n >= 1024 else f"{n}B"


def _read_head_text(path: Path, max_bytes: int = 4096) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_bytes)
    except Exception:
        return ""


def _tail_lines(path: Path, max_lines: int, block_size: int = 8192) -> str:
    if max_lines <= 0:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            data = b""
            pos = end
            while pos > 0 and data.count(b"\n") <= max_lines:
                read_size = min(block_size, pos)
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
            lines = data.splitlines()[-max_lines:]
            return "\n".join(line.decode("utf-8", errors="replace") for line in lines)
    except Exception:
        return ""


@dataclass
class JobCache:
    size: int = 0
    mtime: float = 0.0
    last_scan_pos: int = 0
    records: int = 0
    status: str = "pending"
    submitted: str = ""
    submitted_ts: float = 0.0


_JOB_CACHE: Dict[Path, JobCache] = {}


def parse_job_info(out_path: Path, err_path: Optional[Path]) -> dict:
    name = out_path.stem
    status = "pending"
    out_size = 0
    err_size = 0
    submitted = ""
    submitted_ts = 0.0
    records = 0

    if out_path.exists():
        stat = out_path.stat()
        out_size = stat.st_size
        cache = _JOB_CACHE.get(out_path)
        if cache is None:
            cache = JobCache()
            _JOB_CACHE[out_path] = cache

        if cache.submitted_ts <= 0:
            head = _read_head_text(out_path)
            m = _START_TIME_RE.search(head)
            if m:
                submitted = m.group(1).strip()
                try:
                    import email.utils
                    submitted_ts = email.utils.parsedate_to_datetime(submitted).timestamp()
                except Exception:
                    submitted_ts = stat.st_mtime
            else:
                submitted_ts = stat.st_mtime
            cache.submitted = submitted
            cache.submitted_ts = submitted_ts
        else:
            submitted = cache.submitted
            submitted_ts = cache.submitted_ts

        status = cache.status
        records = cache.records

        if stat.st_size != cache.size or stat.st_mtime != cache.mtime:
            if stat.st_size < cache.last_scan_pos:
                cache.last_scan_pos = 0
                records = 0
                status = "pending"
            try:
                with open(out_path, "r", errors="replace") as f:
                    f.seek(cache.last_scan_pos)
                    new_data = f.read()
                    cache.last_scan_pos = f.tell()
                if new_data:
                    for m2 in _DONE_RE.finditer(new_data):
                        records += int(m2.group(1))
                    if status != "done":
                        if "DONE |" in new_data:
                            status = "done"
                        elif "routes |" in new_data or "Submitting" in new_data:
                            status = "running"
            except Exception:
                pass

            cache.size = stat.st_size
            cache.mtime = stat.st_mtime
            cache.records = records
            cache.status = status

    if err_path and err_path.exists():
        err_size = err_path.stat().st_size
        if err_size > 0 and status == "pending":
            status = "error"

    return {
        "name": name,
        "status": status,
        "records": records,
        "size": _fmt_size(out_size),
        "has_err": err_size > 0,
        "err_size": _fmt_size(err_size),
        "submitted": submitted,
        "submitted_ts": submitted_ts,
    }


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/jobs")
def jobs():
    out_files     = sorted(LOG_DIR.glob("rag_*.out"))
    job_list      = []
    total_records = done = running = errors = 0

    for out_path in out_files:
        err_path = out_path.with_suffix(".err")
        info     = parse_job_info(out_path, err_path)
        job_list.append(info)
        total_records += info["records"]
        if info["status"] == "done":    done    += 1
        if info["status"] == "running": running += 1
        if info["status"] == "error":   errors  += 1

    # Default server-side sort: newest first
    job_list.sort(key=lambda j: j["submitted_ts"], reverse=True)

    return jsonify({
        "jobs": job_list, "total": len(job_list),
        "done": done, "running": running,
        "errors": errors, "total_records": total_records,
    })


@app.route("/stream")
def stream():
    name = request.args.get("name", "")
    tab  = request.args.get("tab", "out")
    try:
        tail = int(request.args.get("tail", "0"))
    except ValueError:
        tail = 0
    ext  = ".out" if tab == "out" else ".err"
    path = LOG_DIR / f"{name}{ext}"

    def generate():
        pos = 0
        if path.exists():
            if tail > 0:
                tail_text = _tail_lines(path, tail)
                if tail_text:
                    for line in tail_text.splitlines():
                        yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                pos = path.stat().st_size
            else:
                with open(path, errors="replace") as f:
                    for line in f:
                        yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                pos = path.stat().st_size

        while True:
            if not path.exists():
                yield ": heartbeat\n\n"
                time.sleep(2)
                continue
            current_size = path.stat().st_size
            if current_size > pos:
                with open(path, errors="replace") as f:
                    f.seek(pos)
                    for line in f:
                        yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
                pos = current_size
            else:
                yield ": heartbeat\n\n"
                time.sleep(2)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="slurm_logs")
    parser.add_argument("--port",    type=int, default=8787)
    args = parser.parse_args()

    LOG_DIR = Path(args.log_dir)
    if not LOG_DIR.exists():
        print(f"Warning: log dir not found: {LOG_DIR}")

    print(f"\nLog server (SSE) — http://localhost:{args.port}")
    print(f"Watching: {LOG_DIR.resolve()}")
    print(f"\nSSH tunnel from your laptop:")
    print(f"  ssh -L {args.port}:localhost:{args.port} <user>@<cluster-login-host>")
    print(f"  Open: http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)