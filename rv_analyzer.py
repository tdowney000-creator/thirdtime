#!/usr/bin/env python3
"""
rv_analyzer.py — Standalone Relative Volume (RV) Analyzer

Scans a universe of stocks and reports which ones are trading at an unusual
volume relative to their recent average ("relative volume" / RV), while
filtering out noise like ETFs, ETNs, and holding companies based on their
listed name.

STANDALONE: this script has no dependency on any other project files. It only
needs the `yfinance` and `pandas` packages.

--------------------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------------------
Every knob below can be changed either by editing the DEFAULT_CONFIG dict,
or (recommended) by passing command-line flags, which always win over the
dict. See `python rv_analyzer.py --help` for the full flag list.

    dollar_volume_min   Minimum today's-dollar-volume (price * volume) a
                         stock must have to be included. Filters out illiquid
                         / low-dollar names. Default: 5,000,000 ($5M)

    rv_min               Minimum relative volume (current volume / average
                         volume) required to show up in results.
                         Default: 2.0  (i.e. trading at 2x normal volume)

    rv_max               Maximum relative volume allowed. Use this to screen
                         OUT extreme outliers (halts, data errors, etc).
                         Default: None (no cap)

    avg_volume_days      Number of trading days used to compute the average
                         "normal" volume baseline. Default: 20

    exclude_name_keywords
                         Case-insensitive substrings; any security whose
                         name contains one of these is dropped BEFORE any
                         network calls are made (cheap filter, applied early).
                         Default: ["etf", "holdings"]
--------------------------------------------------------------------------
"""

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import time
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency: pandas. Install with: pip install pandas --break-system-packages")

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency: yfinance. Install with: pip install yfinance --break-system-packages")


# ==========================================================================
# DEFAULT CONFIG — edit these, or override via CLI flags (see --help)
# ==========================================================================
DEFAULT_CONFIG = {
    "dollar_volume_min": 2,              # $ minimum today's dollar volume (plain number input in the HTML report)
    "rv_min": 2.0,                      # minimum relative volume
    "rv_max": 20.0,                     # maximum relative volume (None = no cap)
    "avg_volume_days": 20,              # lookback window for "average" volume
    "exclude_name_keywords": ["etf", "holdings"],
    "exclude_dollar_prefix": True,       # drop symbols starting with "$" (e.g. index/composite tickers)
    "exclude_dotted_symbols": True,      # drop symbols with a "." in the middle (e.g. "ABC.WS", "ABC.PRA",
                                          # preferred/warrant/unit/rights share classes and dual-class tickers
                                          # like "BRK.B" — set False to keep these)
    "max_tickers": None,                # cap universe size for a quick test run (None = all)
    "batch_size": 200,                  # yfinance download batch size
    "output_format": "both",            # "csv", "html", or "both"
    "output_csv": None,                 # path to write CSV results (None = auto-named)
    "output_html": None,                # path to write HTML report (None = auto-named)
    "auto_open_html": True,             # open the HTML report in your default browser when done
}

NASDAQ_TRADER_URLS = {
    # Official NASDAQ Trader symbol directory files (pipe-delimited).
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}


@dataclass
class Config:
    dollar_volume_min: float = DEFAULT_CONFIG["dollar_volume_min"]
    rv_min: float = DEFAULT_CONFIG["rv_min"]
    rv_max: Optional[float] = DEFAULT_CONFIG["rv_max"]
    avg_volume_days: int = DEFAULT_CONFIG["avg_volume_days"]
    exclude_name_keywords: List[str] = field(
        default_factory=lambda: list(DEFAULT_CONFIG["exclude_name_keywords"])
    )
    exclude_dollar_prefix: bool = DEFAULT_CONFIG["exclude_dollar_prefix"]
    exclude_dotted_symbols: bool = DEFAULT_CONFIG["exclude_dotted_symbols"]
    max_tickers: Optional[int] = DEFAULT_CONFIG["max_tickers"]
    batch_size: int = DEFAULT_CONFIG["batch_size"]
    output_format: str = DEFAULT_CONFIG["output_format"]
    output_csv: Optional[str] = DEFAULT_CONFIG["output_csv"]
    output_html: Optional[str] = DEFAULT_CONFIG["output_html"]
    auto_open_html: bool = DEFAULT_CONFIG["auto_open_html"]
    tickers_file: Optional[str] = None


# ==========================================================================
# UNIVERSE LOADING
# ==========================================================================
def _fetch_symbol_directory(url: str) -> List[dict]:
    """Download and parse one NASDAQ Trader pipe-delimited symbol file."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")

    # Last line is a file-creation-time footer; drop it.
    lines = [ln for ln in raw.splitlines() if ln and not ln.startswith("File Creation Time")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="|")
    return list(reader)


def load_universe_from_nasdaqtrader() -> pd.DataFrame:
    """
    Pulls the full list of NASDAQ- and NYSE/AMEX-listed securities along with
    their official security name, so we can filter by name BEFORE making any
    per-ticker network calls (fast + avoids wasted API hits).
    """
    rows = []

    nasdaq_rows = _fetch_symbol_directory(NASDAQ_TRADER_URLS["nasdaq"])
    for r in nasdaq_rows:
        symbol = r.get("Symbol", "").strip()
        name = r.get("Security Name", "").strip()
        etf_flag = r.get("ETF", "N").strip().upper()
        test_issue = r.get("Test Issue", "N").strip().upper()
        if not symbol or test_issue == "Y":
            continue
        rows.append({"symbol": symbol, "name": name, "etf_flag": etf_flag})

    other_rows = _fetch_symbol_directory(NASDAQ_TRADER_URLS["other"])
    for r in other_rows:
        symbol = r.get("ACT Symbol", "").strip()
        name = r.get("Security Name", "").strip()
        etf_flag = r.get("ETF", "N").strip().upper()
        test_issue = r.get("Test Issue", "N").strip().upper()
        if not symbol or test_issue == "Y":
            continue
        rows.append({"symbol": symbol, "name": name, "etf_flag": etf_flag})

    df = pd.DataFrame(rows).drop_duplicates(subset="symbol")
    return df


def load_universe_from_file(path: str) -> pd.DataFrame:
    """
    Load a custom ticker list. Accepts either:
      - one ticker per line, or
      - a CSV with columns: symbol,name
    If no name column is present, name filtering is skipped for that file.
    """
    with open(path, "r") as f:
        first_line = f.readline()

    if "," in first_line:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "symbol" not in df.columns:
            sys.exit(f"tickers file '{path}' must have a 'symbol' column")
        if "name" not in df.columns:
            df["name"] = ""
        df["etf_flag"] = "N"
        return df[["symbol", "name", "etf_flag"]]
    else:
        with open(path, "r") as f:
            symbols = [ln.strip().upper() for ln in f if ln.strip()]
        return pd.DataFrame({"symbol": symbols, "name": "", "etf_flag": "N"})


def apply_symbol_pattern_filter(
    df: pd.DataFrame, exclude_dollar_prefix: bool, exclude_dotted_symbols: bool
) -> pd.DataFrame:
    """
    Drops symbols by raw formatting rather than name:
      - exclude_dollar_prefix: symbols starting with "$" (index/composite-style tickers)
      - exclude_dotted_symbols: symbols with a "." with characters on both sides
        (e.g. "ABC.WS" warrants, "ABC.PRA" preferred, "ABC.U" units — and also
        catches legitimate dual-class tickers like "BRK.B"; set False to keep those)
    Must run on the RAW symbol, before any "." -> "-" conversion for yfinance.
    """
    if df.empty:
        return df
    mask = pd.Series(False, index=df.index)
    if exclude_dollar_prefix:
        mask |= df["symbol"].str.startswith("$")
    if exclude_dotted_symbols:
        mask |= df["symbol"].str.contains(r"^.+\..+$", regex=True, na=False)
    return df[~mask].copy()


def apply_name_filter(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Drop any row whose name (or NASDAQ ETF flag) matches an excluded keyword."""
    if df.empty:
        return df
    name_lower = df["name"].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for kw in keywords:
        mask |= name_lower.str.contains(kw.lower(), na=False)
    # Also honor NASDAQ's own ETF flag when available, since "etf" keyword
    # alone won't catch every ETF whose official name doesn't say "ETF".
    if "etf_flag" in df.columns:
        mask |= df["etf_flag"].fillna("N").str.upper().eq("Y")
    return df[~mask].copy()


# ==========================================================================
# VOLUME / RV CALCULATION
# ==========================================================================
def compute_rv_for_batch(symbols: List[str], avg_days: int) -> pd.DataFrame:
    """
    Downloads recent daily OHLCV for a batch of symbols and computes:
      - last_close, last_volume (most recent trading day)
      - avg_volume (mean volume over the avg_days days BEFORE the last day)
      - dollar_volume = last_close * last_volume
      - rv = last_volume / avg_volume
    Returns one row per symbol (symbols with insufficient/bad data are skipped).
    """
    period_days = avg_days + 10  # pad for weekends/holidays
    data = yf.download(
        tickers=symbols,
        period=f"{period_days}d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    results = []
    single = len(symbols) == 1
    for sym in symbols:
        try:
            df = data if single else data[sym]
            df = df.dropna(subset=["Volume", "Close"])
            if len(df) < avg_days + 1:
                continue

            last_row = df.iloc[-1]
            baseline = df.iloc[-(avg_days + 1):-1]  # the N days before the last one

            last_close = float(last_row["Close"])
            last_volume = float(last_row["Volume"])
            avg_volume = float(baseline["Volume"].mean())

            if avg_volume <= 0 or last_close <= 0:
                continue

            rv = last_volume / avg_volume
            dollar_volume = last_close * last_volume

            results.append(
                {
                    "symbol": sym,
                    "last_close": round(last_close, 2),
                    "last_volume": int(last_volume),
                    "avg_volume": int(avg_volume),
                    "rv": round(rv, 2),
                    "dollar_volume": round(dollar_volume, 0),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(results)


def chunk(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def write_html_report(df: pd.DataFrame, path: str, cfg: "Config") -> None:
    """
    Writes a self-contained, interactive HTML report (no server, no external
    assets — just open the file in a browser). Embeds the full computed
    dataset (already restricted by name/symbol-pattern filters, but NOT yet
    by dollar-volume/RV) and lets you re-filter live with three sliders:

        Min $ Volume   — slider bound scales to the largest dollar volume
                         in this run's data; starts at cfg.dollar_volume_min
        Min RV         — 0x to 20x; starts at cfg.rv_min
        Max RV         — 0x to 20x (hard cap, per request); starts at
                         min(cfg.rv_max, 20) if set, else 20

    Rows outside the current slider band are hidden client-side; nothing is
    re-downloaded, so this is instant.
    """
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Embed the raw numeric rows for client-side filtering/formatting.
    records = df[["symbol", "name", "last_close", "last_volume", "avg_volume", "rv", "dollar_volume"]].to_dict(orient="records")
    data_json = json.dumps(records)

    RV_SLIDER_CAP = 20  # hard cap per request
    min_rv_default = min(cfg.rv_min, RV_SLIDER_CAP)
    max_rv_default = min(cfg.rv_max, RV_SLIDER_CAP) if cfg.rv_max is not None else RV_SLIDER_CAP

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Relative Volume Report — {generated}</title>
<style>
  :root {{
    --bg: #0f1117; --panel: #171a23; --border: #2a2e3a; --text: #e6e8ee;
    --muted: #9aa1b2; --accent: #4f8cff; --row-alt: #1b1f2a; --hover: #232838;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .wrap {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 20px; }}
  .controls {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 20px; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px 28px;
  }}
  .control label {{
    display: flex; justify-content: space-between; font-size: 12px; color: var(--muted);
    text-transform: uppercase; letter-spacing: .04em; margin-bottom: 8px;
  }}
  .control label b {{ color: var(--text); font-size: 13px; text-transform: none; letter-spacing: 0; }}
  .control input[type=range] {{ width: 100%; accent-color: var(--accent); }}
  .control input[type=number] {{
    width: 100%; margin-top: 6px; background: #10131c; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 5px 8px; font-size: 13px;
  }}
  .config-static {{ color: var(--muted); font-size: 12px; margin-bottom: 16px; }}
  table.rv-table {{
    width: 100%; border-collapse: collapse; background: var(--panel);
    border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
    font-size: 14px;
  }}
  table.rv-table th {{
    text-align: left; padding: 10px 14px; background: #1d2130; color: var(--muted);
    font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
    border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
  }}
  table.rv-table td {{
    padding: 9px 14px; border-bottom: 1px solid var(--border); white-space: nowrap;
  }}
  table.rv-table tr.alt {{ background: var(--row-alt); }}
  table.rv-table tr:hover {{ background: var(--hover); }}
  table.rv-table td:first-child {{ font-weight: 700; color: var(--accent); }}
  .empty {{ color: var(--muted); padding: 24px; text-align: center; }}
  #resultCount {{ color: var(--text); font-weight: 600; }}
</style>
</head>
<body>
  <div class="wrap">
    <h1>Relative Volume Report</h1>
    <div class="meta">Generated {generated} &middot; <span id="resultCount"></span> of {len(records)} symbols shown</div>

    <div class="config-static">
      Avg Volume Window: <b>{cfg.avg_volume_days}d</b> &nbsp;|&nbsp;
      Excluded Keywords: <b>{', '.join(cfg.exclude_name_keywords) if cfg.exclude_name_keywords else 'none'}</b> &nbsp;|&nbsp;
      $-Prefix Excluded: <b>{cfg.exclude_dollar_prefix}</b> &nbsp;|&nbsp;
      Dotted Symbols Excluded: <b>{cfg.exclude_dotted_symbols}</b>
    </div>

    <div class="controls">
      <div class="control">
        <label><b>Min $ Volume</b> <span id="minDollarVal"></span></label>
        <input type="number" id="minDollarNum" min="0" step="1" value="{cfg.dollar_volume_min}">
      </div>
      <div class="control">
        <label><b>Min RV</b> <span id="minRvVal"></span></label>
        <input type="range" id="minRvSlider" min="0" max="{RV_SLIDER_CAP}" step="0.1" value="{min_rv_default}">
        <input type="number" id="minRvNum" min="0" max="{RV_SLIDER_CAP}" step="0.1" value="{min_rv_default}">
      </div>
      <div class="control">
        <label><b>Max RV</b> <span id="maxRvVal"></span></label>
        <input type="range" id="maxRvSlider" min="0" max="{RV_SLIDER_CAP}" step="0.5" value="{max_rv_default}">
        <input type="number" id="maxRvNum" min="0" max="{RV_SLIDER_CAP}" step="0.5" value="{max_rv_default}">
      </div>
    </div>

    <table class="rv-table" id="rvTable">
      <thead>
        <tr>
          <th data-key="symbol">Symbol</th>
          <th data-key="name">Name</th>
          <th data-key="last_close">Last Close</th>
          <th data-key="last_volume">Volume</th>
          <th data-key="avg_volume">Avg Volume ({cfg.avg_volume_days}d)</th>
          <th data-key="rv">RV</th>
          <th data-key="dollar_volume">Dollar Volume</th>
        </tr>
      </thead>
      <tbody id="rvBody"></tbody>
    </table>
    <div class="empty" id="emptyMsg" style="display:none;">No symbols match the current filters.</div>
  </div>

<script>
  const DATA = {data_json};
  let sortKey = "rv", sortDir = -1;

  const fmtMoney = v => "$" + Number(v).toLocaleString(undefined, {{maximumFractionDigits: 0}});
  const fmtPrice = v => "$" + Number(v).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
  const fmtNum = v => Number(v).toLocaleString();
  const fmtRv = v => Number(v).toFixed(2) + "x";

  function currentFilters() {{
    return {{
      minDollar: parseFloat(document.getElementById('minDollarNum').value) || 0,
      minRv: parseFloat(document.getElementById('minRvSlider').value),
      maxRv: parseFloat(document.getElementById('maxRvSlider').value),
    }};
  }}

  function render() {{
    const f = currentFilters();
    document.getElementById('minDollarVal').textContent = fmtMoney(f.minDollar);
    document.getElementById('minRvVal').textContent = f.minRv.toFixed(1) + "x";
    document.getElementById('maxRvVal').textContent = f.maxRv.toFixed(1) + "x";

    let rows = DATA.filter(r => r.dollar_volume >= f.minDollar && r.rv >= f.minRv && r.rv <= f.maxRv);
    rows.sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0) * sortDir);

    const body = document.getElementById('rvBody');
    body.innerHTML = "";
    rows.forEach((r, i) => {{
      const tr = document.createElement('tr');
      if (i % 2 === 1) tr.className = 'alt';
      tr.innerHTML = `<td>${{r.symbol}}</td><td>${{r.name}}</td><td>${{fmtPrice(r.last_close)}}</td>` +
        `<td>${{fmtNum(r.last_volume)}}</td><td>${{fmtNum(r.avg_volume)}}</td>` +
        `<td>${{fmtRv(r.rv)}}</td><td>${{fmtMoney(r.dollar_volume)}}</td>`;
      body.appendChild(tr);
    }});
    document.getElementById('resultCount').textContent = rows.length;
    document.getElementById('emptyMsg').style.display = rows.length ? 'none' : 'block';
  }}

  function link(sliderId, numId) {{
    const slider = document.getElementById(sliderId), num = document.getElementById(numId);
    slider.addEventListener('input', () => {{ num.value = slider.value; render(); }});
    num.addEventListener('input', () => {{ slider.value = num.value; render(); }});
  }}
  link('minRvSlider', 'minRvNum');
  link('maxRvSlider', 'maxRvNum');
  document.getElementById('minDollarNum').addEventListener('input', render);

  document.querySelectorAll('#rvTable th').forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.dataset.key;
      sortDir = (sortKey === key) ? -sortDir : -1;
      sortKey = key;
      render();
    }});
  }});

  render();
</script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================
def run(cfg: Config) -> pd.DataFrame:
    print("Loading ticker universe...")
    if cfg.tickers_file:
        universe = load_universe_from_file(cfg.tickers_file)
    else:
        universe = load_universe_from_nasdaqtrader()
    print(f"  {len(universe)} raw symbols loaded")

    universe = apply_symbol_pattern_filter(
        universe, cfg.exclude_dollar_prefix, cfg.exclude_dotted_symbols
    )
    print(f"  {len(universe)} symbols remain after symbol-pattern filtering "
          f"($ prefix={cfg.exclude_dollar_prefix}, dotted={cfg.exclude_dotted_symbols})")

    universe = apply_name_filter(universe, cfg.exclude_name_keywords)
    print(f"  {len(universe)} symbols remain after excluding {cfg.exclude_name_keywords}")

    # Yahoo Finance uses "-" instead of "." for any remaining edge-case symbols
    universe["symbol"] = universe["symbol"].str.replace(".", "-", regex=False)

    symbols = universe["symbol"].tolist()
    if cfg.max_tickers:
        symbols = symbols[: cfg.max_tickers]
        print(f"  limited to first {len(symbols)} symbols (max_tickers set)")

    name_lookup = dict(zip(universe["symbol"], universe["name"]))

    print(f"Pulling volume data in batches of {cfg.batch_size}...")
    all_results = []
    total_batches = (len(symbols) + cfg.batch_size - 1) // cfg.batch_size
    for i, batch in enumerate(chunk(symbols, cfg.batch_size), start=1):
        print(f"  batch {i}/{total_batches} ({len(batch)} symbols)...")
        batch_df = compute_rv_for_batch(batch, cfg.avg_volume_days)
        if not batch_df.empty:
            all_results.append(batch_df)
        time.sleep(0.5)  # be polite to the data source

    if not all_results:
        print("No data returned — nothing to report.")
        return pd.DataFrame()

    results = pd.concat(all_results, ignore_index=True)
    results["name"] = results["symbol"].map(name_lookup).fillna("")

    cols = ["symbol", "name", "last_close", "last_volume", "avg_volume", "rv", "dollar_volume"]
    results_full = results[cols].sort_values("rv", ascending=False).reset_index(drop=True)
    screened = results_full[
        (results_full["dollar_volume"] >= cfg.dollar_volume_min)
        & (results_full["rv"] >= cfg.rv_min)
        & (results_full["rv"] <= cfg.rv_max if cfg.rv_max is not None else True)
    ].reset_index(drop=True)

    out_paths = []
    if cfg.output_format in ("csv", "both"):
        csv_path = cfg.output_csv or f"rv_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
        screened.to_csv(csv_path, index=False)
        out_paths.append(csv_path)
    if cfg.output_format in ("html", "both"):
        html_path = cfg.output_html or f"rv_results_{dt.datetime.now():%Y%m%d_%H%M%S}.html"
        # HTML embeds the FULL results (name/symbol filters applied, but not the
        # dollar/RV thresholds) so the in-page sliders have data to explore;
        # the sliders just start positioned at cfg.dollar_volume_min / rv_min / rv_max.
        write_html_report(results_full, html_path, cfg)
        out_paths.append(html_path)
        if cfg.auto_open_html:
            try:
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception:
                pass  # auto-open is best-effort; the file is still on disk

    print(f"\nSaved {len(screened)} results (of {len(results_full)} total scanned) to: {', '.join(out_paths)}")

    return screened


# ==========================================================================
# CLI
# ==========================================================================
def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Standalone Relative Volume (RV) analyzer with ETF/holdings name filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dollar-amount", type=float, default=DEFAULT_CONFIG["dollar_volume_min"],
                    help="Minimum today's dollar volume (price * volume) to include a stock")
    p.add_argument("--min-rv", type=float, default=DEFAULT_CONFIG["rv_min"],
                    help="Minimum relative volume (current / average) to include a stock")
    p.add_argument("--max-rv", type=float, default=DEFAULT_CONFIG["rv_max"],
                    help="Maximum relative volume to include a stock (omit for no cap)")
    p.add_argument("--avg-days", type=int, default=DEFAULT_CONFIG["avg_volume_days"],
                    help="Number of trading days used for the average-volume baseline")
    p.add_argument("--exclude", nargs="*", default=DEFAULT_CONFIG["exclude_name_keywords"],
                    help="Case-insensitive keywords; names containing any of these are excluded")
    p.add_argument("--allow-dollar-symbols", action="store_true",
                    help="Keep symbols that start with '$' (excluded by default)")
    p.add_argument("--allow-dotted-symbols", action="store_true",
                    help="Keep symbols with a '.' in the middle, e.g. warrants/preferred/units and "
                         "dual-class tickers like 'BRK.B' (excluded by default)")
    p.add_argument("--tickers-file", type=str, default=None,
                    help="Path to a custom ticker list (one per line, or CSV with symbol,name columns). "
                         "If omitted, the full NASDAQ/NYSE/AMEX listing is downloaded automatically.")
    p.add_argument("--max-tickers", type=int, default=DEFAULT_CONFIG["max_tickers"],
                    help="Cap the universe size (useful for a quick test run)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                    help="Number of symbols per yfinance download batch")
    p.add_argument("--output-format", choices=["csv", "html", "both"], default=DEFAULT_CONFIG["output_format"],
                    help="Which report format(s) to write")
    p.add_argument("--output", type=str, default=DEFAULT_CONFIG["output_csv"],
                    help="Output CSV path (default: auto-named with a timestamp)")
    p.add_argument("--output-html", type=str, default=DEFAULT_CONFIG["output_html"],
                    help="Output HTML path (default: auto-named with a timestamp)")
    p.add_argument("--no-auto-open", action="store_true",
                    help="Don't automatically open the HTML report in your browser when done")
    args = p.parse_args()

    return Config(
        dollar_volume_min=args.dollar_amount,
        rv_min=args.min_rv,
        rv_max=args.max_rv,
        avg_volume_days=args.avg_days,
        exclude_name_keywords=args.exclude,
        exclude_dollar_prefix=not args.allow_dollar_symbols,
        exclude_dotted_symbols=not args.allow_dotted_symbols,
        tickers_file=args.tickers_file,
        max_tickers=args.max_tickers,
        batch_size=args.batch_size,
        output_format=args.output_format,
        output_csv=args.output,
        output_html=args.output_html,
        auto_open_html=not args.no_auto_open,
    )


if __name__ == "__main__":
    config = parse_args()
    results_df = run(config)
    if not results_df.empty:
        with pd.option_context("display.max_rows", 50, "display.width", 140):
            print("\nTop results:\n")
            print(results_df.head(50).to_string(index=False))
