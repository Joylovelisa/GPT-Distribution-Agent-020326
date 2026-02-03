import os
import re
import io
import json
import base64
import random
import datetime
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import streamlit as st
import pandas as pd
import yaml
from rapidfuzz import fuzz

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Deployment assumptions (Hugging Face Spaces + Streamlit)
# - Supports: OpenAI, Gemini, Anthropic, Grok (xAI) via API keys
# - agents.yaml optional (editable via UI); fallback embedded below
# ============================================================

CORAL = "#FF7F50"

KEY_ENV_CANDIDATES = {
    "OPENAI_API_KEY": ["OPENAI_API_KEY"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"],
    "XAI_API_KEY": ["XAI_API_KEY"],
}

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-3-pro-preview"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
XAI_MODELS = ["grok-4-fast-reasoning", "grok-3-mini"]

DEFAULT_AGENTS_YAML = """version: "1.0"
agents:
  - id: dataset_standardizer_advisor
    name: Dataset Standardizer Advisor
    description: Review schema mapping report; suggest fixes or additional synonyms.
    provider: openai
    model: gpt-4o-mini
    temperature: 0.1
    max_tokens: 2000
    system_prompt: |
      You are a meticulous data engineering assistant.
      You never fabricate data. You propose mapping and cleaning rules only.
      Output Markdown with clear steps.
    user_prompt: |
      Given the mapping report and sample records, suggest improvements to the standardization:
      - missing columns
      - ambiguous fields
      - date parsing
      - numeric coercion
      - dedup strategy

  - id: distribution_insights_analyst
    name: Distribution Insights Analyst
    description: Produce executive summary + anomalies + recommended visualizations.
    provider: openai
    model: gpt-4o-mini
    temperature: 0.2
    max_tokens: 3500
    system_prompt: |
      You are a senior distribution analytics consultant.
      You work only from provided data summaries; do not invent values.
      Output Markdown with sections and bullet points. Include risks and data quality notes.
    user_prompt: |
      Analyze the distribution dataset:
      - KPIs and trends
      - network structure insights (Supplier->License->Model->Customer)
      - outliers/anomalies
      - data quality issues and fixes
      - recommended next actions
"""

# ============================================================
# i18n (English + Traditional Chinese)
# ============================================================
STRINGS = {
    "en": {
        "app_title": "WOW Distribution Analysis Studio",
        "nav_data": "Data Studio",
        "nav_dashboard": "Interactive Dashboard",
        "nav_agents": "Agent Studio",
        "settings": "Settings",
        "theme": "Theme",
        "language": "Language",
        "style": "Painter Style",
        "jackpot": "Jackpot",
        "light": "Light",
        "dark": "Dark",

        "status": "WOW Status",
        "api_keys": "API Keys",
        "managed_by_env": "Authenticated via Environment",
        "missing_key": "Missing — enter on this page",
        "session_key": "Session",

        "data_source": "Dataset Source",
        "use_default": "Use default dataset",
        "paste": "Paste",
        "upload": "Upload",
        "parse_load": "Parse & Load",
        "auto_standardize": "Auto-standardize (recommended)",
        "standardization_report": "Standardization report",
        "preview_20": "Preview (first 20)",
        "download_csv": "Download CSV",
        "download_json": "Download JSON",

        "filters": "Filters",
        "supplier_id": "Supplier ID",
        "license_no": "License No",
        "model": "Model",
        "customer_id": "Customer ID",
        "date_range": "Date range",
        "search": "Search (Device/Category/UDI/Lot/Serial...)",
        "rows": "Rows",
        "quantity": "Quantity",

        "viz_instructions": "Optional instructions for visualization/analysis",
        "dashboard": "Dashboard",
        "summary": "Summary",
        "table": "Filtered Table",

        "viz_sankey": "1) Distribution Sankey (Supplier → License → Model → Customer)",
        "viz_network": "2) Distribution Network Graph (layered)",
        "viz_timeseries": "3) Delivery Trend Over Time",
        "viz_top_suppliers": "4) Top Suppliers",
        "viz_sunburst": "5) Sunburst (Supplier → License → Model → Customer)",
        "viz_heatmap": "6) Heatmap (Supplier × Model)",

        "agent_pipeline": "Agent Pipeline",
        "agent": "Agent",
        "provider": "Provider",
        "model_select": "Model",
        "max_tokens": "Max tokens",
        "temperature": "Temperature",
        "system_prompt": "System prompt",
        "user_prompt": "User prompt",
        "run_agent": "Run agent",
        "input_to_agent": "Input to agent",
        "output": "Output",
        "edit_for_next": "Edit output used as input to next agent",

        "data_quality": "Data quality notes",
        "clear_session": "Clear session",
    },
    "zh-TW": {
        "app_title": "WOW 配送/流向分析工作室",
        "nav_data": "資料工作室",
        "nav_dashboard": "互動儀表板",
        "nav_agents": "代理工作室",
        "settings": "設定",
        "theme": "主題",
        "language": "語言",
        "style": "畫家風格",
        "jackpot": "隨機開獎",
        "light": "亮色",
        "dark": "暗色",

        "status": "WOW 狀態",
        "api_keys": "API 金鑰",
        "managed_by_env": "由環境變數驗證",
        "missing_key": "未設定 — 請在網頁輸入",
        "session_key": "Session",

        "data_source": "資料來源",
        "use_default": "使用預設資料",
        "paste": "貼上",
        "upload": "上傳",
        "parse_load": "解析並載入",
        "auto_standardize": "自動標準化（建議）",
        "standardization_report": "標準化報告",
        "preview_20": "預覽（前 20 筆）",
        "download_csv": "下載 CSV",
        "download_json": "下載 JSON",

        "filters": "篩選條件",
        "supplier_id": "供應商代碼 SupplierID",
        "license_no": "許可證字號 LicenseNo",
        "model": "型號 Model",
        "customer_id": "客戶代碼 CustomerID",
        "date_range": "日期範圍",
        "search": "搜尋（品名/分類/UDI/批號/序號…）",
        "rows": "筆數",
        "quantity": "數量",

        "viz_instructions": "（可選）視覺化/分析指令",
        "dashboard": "儀表板",
        "summary": "摘要",
        "table": "篩選後表格",

        "viz_sankey": "1) 配送 Sankey（Supplier → License → Model → Customer）",
        "viz_network": "2) 配送網路圖（分層）",
        "viz_timeseries": "3) 配送趨勢（時間序列）",
        "viz_top_suppliers": "4) Top 供應商",
        "viz_sunburst": "5) Sunburst（Supplier → License → Model → Customer）",
        "viz_heatmap": "6) 熱圖（Supplier × Model）",

        "agent_pipeline": "代理流程",
        "agent": "代理",
        "provider": "供應商",
        "model_select": "模型",
        "max_tokens": "最大 tokens",
        "temperature": "溫度",
        "system_prompt": "系統提示詞",
        "user_prompt": "使用者提示詞",
        "run_agent": "執行代理",
        "input_to_agent": "代理輸入",
        "output": "輸出",
        "edit_for_next": "編修輸出（作為下一代理輸入）",

        "data_quality": "資料品質備註",
        "clear_session": "清除 session",
    }
}


def t(lang: str, key: str) -> str:
    return STRINGS.get(lang, STRINGS["en"]).get(key, key)


# ============================================================
# Painter styles (20)
# ============================================================
PAINTER_STYLES = [
    {"id": "monet", "name": "Claude Monet", "accent": "#7FB3D5"},
    {"id": "vangogh", "name": "Vincent van Gogh", "accent": "#F4D03F"},
    {"id": "picasso", "name": "Pablo Picasso", "accent": "#AF7AC5"},
    {"id": "rembrandt", "name": "Rembrandt", "accent": "#D4AC0D"},
    {"id": "vermeer", "name": "Johannes Vermeer", "accent": "#5DADE2"},
    {"id": "hokusai", "name": "Hokusai", "accent": "#48C9B0"},
    {"id": "klimt", "name": "Gustav Klimt", "accent": "#F5CBA7"},
    {"id": "kahlo", "name": "Frida Kahlo", "accent": "#EC7063"},
    {"id": "pollock", "name": "Jackson Pollock", "accent": "#58D68D"},
    {"id": "cezanne", "name": "Paul Cézanne", "accent": "#F0B27A"},
    {"id": "turner", "name": "J.M.W. Turner", "accent": "#F5B041"},
    {"id": "matisse", "name": "Henri Matisse", "accent": "#EB984E"},
    {"id": "dali", "name": "Salvador Dalí", "accent": "#85C1E9"},
    {"id": "warhol", "name": "Andy Warhol", "accent": "#FF5DA2"},
    {"id": "sargent", "name": "John Singer Sargent", "accent": "#AED6F1"},
    {"id": "rothko", "name": "Mark Rothko", "accent": "#CD6155"},
    {"id": "caravaggio", "name": "Caravaggio", "accent": "#A04000"},
    {"id": "okeeffe", "name": "Georgia O’Keeffe", "accent": "#F1948A"},
    {"id": "seurat", "name": "Georges Seurat", "accent": "#76D7C4"},
    {"id": "basquiat", "name": "Jean-Michel Basquiat", "accent": "#F7DC6F"},
]


def jackpot_style():
    return random.choice(PAINTER_STYLES)


# ============================================================
# WOW CSS (Light/Dark + Painter Accent + Glassmorphism)
# ============================================================
def inject_css(theme: str, painter_accent: str, coral: str = CORAL):
    if theme == "light":
        bg = "#F6F7FB"
        fg = "#0B1020"
        card = "rgba(10, 16, 32, 0.05)"
        border = "rgba(10, 16, 32, 0.12)"
        shadow = "rgba(10, 16, 32, 0.12)"
    else:
        bg = "#0B1020"
        fg = "#EAF0FF"
        card = "rgba(255,255,255,0.06)"
        border = "rgba(255,255,255,0.10)"
        shadow = "rgba(0,0,0,0.40)"

    return f"""
    <style>
      :root {{
        --bg: {bg};
        --fg: {fg};
        --card: {card};
        --border: {border};
        --accent: {painter_accent};
        --coral: {coral};
        --ok: #2ECC71;
        --warn: #F1C40F;
        --bad: #E74C3C;
        --shadow: {shadow};
      }}

      .stApp {{
        background:
          radial-gradient(1200px 600px at 20% 0%, rgba(255,127,80,0.14), transparent 60%),
          radial-gradient(900px 500px at 80% 10%, rgba(0,200,255,0.12), transparent 55%),
          var(--bg);
        color: var(--fg);
      }}

      .wow-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px;
        backdrop-filter: blur(12px);
        box-shadow: 0 18px 55px var(--shadow);
      }}

      .wow-mini {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 10px 12px;
        backdrop-filter: blur(12px);
      }}

      .chip {{
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding: 6px 10px;
        margin: 0 8px 8px 0;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--card);
        font-size: 12px;
        line-height: 1;
      }}
      .dot {{
        width: 9px; height: 9px; border-radius: 99px;
        background: var(--accent);
        box-shadow: 0 0 0 3px rgba(255,255,255,0.06);
      }}

      .coral {{
        color: var(--coral);
        font-weight: 900;
      }}

      .fab {{
        position: fixed;
        bottom: 20px;
        right: 22px;
        z-index: 9999;
        border-radius: 999px;
        padding: 12px 16px;
        background: linear-gradient(135deg, var(--accent), var(--coral));
        color: white;
        font-weight: 900;
        border: 0px;
        box-shadow: 0 22px 55px rgba(0,0,0,0.45);
        letter-spacing: 0.5px;
      }}
      .fab-sub {{
        position: fixed;
        bottom: 68px;
        right: 22px;
        z-index: 9999;
        font-size: 12px;
        padding: 8px 10px;
        border-radius: 12px;
        background: var(--card);
        border: 1px solid var(--border);
        color: var(--fg);
        backdrop-filter: blur(10px);
      }}
      a {{ color: var(--accent) !important; }}
    </style>
    """


# ============================================================
# API key management (hide key input if env exists)
# ============================================================
def provider_model_map():
    return {
        "openai": OPENAI_MODELS,
        "gemini": GEMINI_MODELS,
        "anthropic": ANTHROPIC_MODELS,
        "xai": XAI_MODELS,
    }


def _get_env_any(env_keys: List[str]) -> Optional[str]:
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    return None


def get_api_key(env_primary: str) -> Tuple[Optional[str], str]:
    env_val = _get_env_any(KEY_ENV_CANDIDATES.get(env_primary, [env_primary]))
    if env_val:
        return env_val, "env"
    sess = st.session_state.get("api_keys", {}).get(env_primary)
    if sess:
        return sess, "session"
    return None, "missing"


def call_llm_text(provider: str, model: str, api_key: str, system: str, user: str,
                  max_tokens: int = 12000, temperature: float = 0.2) -> str:
    provider = (provider or "").lower().strip()

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.output_text or ""

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        r = m.generate_content([system, user])
        return (r.text or "").strip()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = []
        for b in msg.content:
            if getattr(b, "type", "") == "text":
                parts.append(b.text)
        return "".join(parts).strip()

    if provider == "xai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.responses.create(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.output_text or ""

    raise ValueError(f"Unsupported provider: {provider}")


# ============================================================
# Default dataset (provided by user) + parsing
# ============================================================
DEFAULT_DISTRIBUTION_CSV = """SupplierID,Deliverdate,CustomerID,LicenseNo,Category,UDID,DeviceNAME,LotNO,SerNo,Model,Number
B00079,20251107,C05278,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576331,“波士頓科技”英吉尼心臟節律器,890057,,L111,1
B00079,20251106,C06030,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576331,“波士頓科技”英吉尼心臟節律器,872177,,L111,1
B00079,20251106,C00123,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576331,“波士頓科技”英吉尼心臟節律器,889490,,L111,1
B00079,20251105,C06034,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576331,“波士頓科技”英吉尼心臟節律器,889253,,L111,1
B00079,20251103,C05363,衛部醫器輸字第029100號,E.3610植入式心律器之脈搏產生器,00802526576461,“波士頓科技”艾科雷心臟節律器,869531,,L311,1
B00079,20251103,C06034,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576331,“波士頓科技”英吉尼心臟節律器,889230,,L111,1
B00079,20251103,C05278,衛部醫器輸字第029100號,E.3610植入式心律器之脈搏產生器,00802526576485,“波士頓科技”艾科雷心臟節律器,182310,,L331,1
B00051,20251030,C02822,衛部醫器輸字第028560號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606478,“尼奧麥迪克”舒兒莉芙特骨盆懸吊系統,CC250520,19,CPS02,1
B00079,20251030,C00123,衛部醫器輸字第033951號,E.3610植入式心律器之脈搏產生器,00802526576324,“波士頓科技”英吉尼心臟節律器,915900,,L110,1
B00051,20251030,C02822,衛部醫器輸字第028560號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606478,“尼奧麥迪克”舒兒莉芙特骨盆懸吊系統,CC250520,20,CPS02,1
B00051,20251029,C02082,衛部醫器輸字第028560號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606478,“尼奧麥迪克”舒兒莉芙特骨盆懸吊系統,CC250326,4,CPS02,1
B00051,20251029,C02082,衛部醫器輸字第028560號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606478,“尼奧麥迪克”舒兒莉芙特骨盆懸吊系統,CC250326,5,CPS02,1
B00209,20251028,C03210,衛部醫器輸字第026988號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,07798121803473,“博美敦”凱莉星脫垂修補系統,,00012150,Calistar S,1
B00051,20251028,C01774,衛部醫器輸字第030820號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606515,“尼奧麥迪克”蜜普思微創骨盆懸吊系統,MB241203,140,KITMIPS02,1
B00209,20251028,C03210,衛部醫器輸字第026988號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,07798121803473,“博美敦”凱莉星脫垂修補系統,,00012154,Calistar S,1
B00051,20251028,C01773,衛部醫器輸字第028560號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606478,“尼奧麥迪克”舒兒莉芙特骨盆懸吊系統,CC241128,85,CPS02,1
B00209,20251028,C03210,衛部醫器輸字第026988號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,07798121803473,“博美敦”凱莉星脫垂修補系統,,00012155,Calistar S,1
B00051,20251028,C01774,衛部醫器輸字第030820號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,08437007606515,“尼奧麥迪克”蜜普思微創骨盆懸吊系統,MB241203,142,KITMIPS02,1
B00209,20251028,C03210,衛部醫器輸字第026988號,L.5980經陰道骨盆腔器官脫垂治療用手術網片,07798121803473,“博美敦”凱莉星脫垂修補系統,,00012156,Calistar S,1
"""


def detect_format(text: str) -> str:
    t0 = (text or "").lstrip()
    if not t0:
        return "unknown"
    if t0.startswith("{") or t0.startswith("["):
        return "json"
    if "," in t0 and "\n" in t0:
        return "csv"
    return "text"


def parse_dataset_blob(blob: Union[str, bytes], filename: Optional[str] = None) -> pd.DataFrame:
    if isinstance(blob, bytes):
        text = blob.decode("utf-8", errors="ignore")
    else:
        text = blob

    fmt = None
    if filename:
        fn = filename.lower()
        if fn.endswith(".json"):
            fmt = "json"
        elif fn.endswith(".csv") or fn.endswith(".txt"):
            fmt = detect_format(text)
        else:
            fmt = detect_format(text)
    else:
        fmt = detect_format(text)

    if fmt == "json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ["data", "records", "items", "rows", "dataset"]:
                if k in obj and isinstance(obj[k], list):
                    obj = obj[k]
                    break
            if isinstance(obj, dict):
                obj = [obj]
        if not isinstance(obj, list):
            raise ValueError("JSON must be a list of objects (or a wrapper containing a list).")
        return pd.DataFrame(obj)

    # csv or text-like (attempt CSV)
    return pd.read_csv(io.StringIO(text))


# ============================================================
# Standardization for distribution datasets
# Canonical schema (internal):
# supplier_id, deliver_date, customer_id, license_no, category,
# udi_di, device_name, lot_no, serial_no, model, quantity
# ============================================================
CANON = [
    "supplier_id",
    "deliver_date",
    "customer_id",
    "license_no",
    "category",
    "udi_di",
    "device_name",
    "lot_no",
    "serial_no",
    "model",
    "quantity",
]

SYNONYMS = {
    "supplier_id": ["supplierid", "supplier_id", "supplier", "vendor", "供應商", "供應商代碼"],
    "deliver_date": ["deliverdate", "deliver_date", "date", "shipment_date", "delivery_date", "出貨日", "交貨日", "日期"],
    "customer_id": ["customerid", "customer_id", "customer", "client", "買方", "客戶", "客戶代碼"],
    "license_no": ["licenseno", "license_no", "license", "permit", "許可證", "許可證字號"],
    "category": ["category", "class", "product_category", "分類", "類別"],
    "udi_di": ["udid", "udi_di", "udi", "di", "主識別碼", "UDI", "UDID"],
    "device_name": ["devicename", "device_name", "device", "product_name", "品名", "裝置名稱", "DeviceNAME"],
    "lot_no": ["lotno", "lot_no", "lot", "batch", "批號", "LotNO"],
    "serial_no": ["serno", "serial_no", "serial", "sn", "序號", "SerNo"],
    "model": ["model", "model_no", "model_number", "型號"],
    "quantity": ["number", "qty", "quantity", "count", "數量", "Number"],
}


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _best_match_column(df_cols: List[str], candidates: List[str]) -> Optional[str]:
    norm_map = {_norm_col(c): c for c in df_cols}
    for cand in candidates:
        n = _norm_col(cand)
        if n in norm_map:
            return norm_map[n]
    best, best_score = None, 0
    for c in df_cols:
        for cand in candidates:
            sc = fuzz.ratio(_norm_col(c), _norm_col(cand))
            if sc > best_score:
                best_score, best = sc, c
    return best if best_score >= 85 else None


def _clean_quotes(s: Any) -> Any:
    if s is None:
        return None
    if isinstance(s, str):
        return s.replace("“", '"').replace("”", '"').strip()
    return s


def _parse_deliver_date(v: Any) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    # common format: YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        try:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except Exception:
            return None
    # fallback
    return pd.to_datetime(s, errors="coerce")


def standardize_distribution_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(columns=CANON), "No data to standardize."

    original_cols = list(df.columns)
    mapped: Dict[str, Optional[str]] = {}
    report_lines = ["### Standardization Mapping", "", "| Canonical field | Source column |", "|---|---|"]

    for cfield in CANON:
        src = _best_match_column(original_cols, SYNONYMS.get(cfield, [cfield]))
        mapped[cfield] = src
        report_lines.append(f"| `{cfield}` | `{src if src else '— (missing)'}` |")

    out = pd.DataFrame()
    for cfield in CANON:
        src = mapped[cfield]
        out[cfield] = df[src] if (src and src in df.columns) else None

    # clean strings
    for c in ["supplier_id", "customer_id", "license_no", "category", "udi_di", "device_name", "lot_no", "serial_no", "model"]:
        out[c] = out[c].apply(_clean_quotes)
        out[c] = out[c].astype("string").str.strip()

    # quantity to int
    def to_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        try:
            return int(float(str(x).replace(",", "").strip()))
        except Exception:
            return 0

    out["quantity"] = out["quantity"].apply(to_int)

    # parse dates
    out["deliver_date"] = out["deliver_date"].apply(_parse_deliver_date)

    # remove empty rows
    def has_signal(r):
        for c in CANON:
            v = r.get(c)
            if c == "quantity":
                if int(v or 0) != 0:
                    return True
                continue
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if str(v).strip() != "" and str(v).strip().lower() != "nan":
                return True
        return False

    out = out[out.apply(has_signal, axis=1)].reset_index(drop=True)

    # data quality notes
    missing = [c for c in CANON if out[c].isna().all()]
    report_lines += ["", f"**Rows:** {len(out)}", f"**Original columns:** {len(original_cols)}"]
    if missing:
        report_lines += ["", "### Missing Canonical Fields", "- " + "\n- ".join([f"`{m}`" for m in missing])]

    # sort by date if possible
    if "deliver_date" in out.columns:
        out = out.sort_values("deliver_date", na_position="last").reset_index(drop=True)

    return out, "\n".join(report_lines)


def df_to_json_records(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


# ============================================================
# Visualization builders (6 features)
# ============================================================
def build_sankey(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    # aggregate by path
    g = df.groupby(["supplier_id", "license_no", "model", "customer_id"], dropna=False)["quantity"].sum().reset_index()
    g = g.fillna("∅")

    # label namespaces so same code doesn't collide across levels
    def ns(level: str, v: Any) -> str:
        return f"{level}:{v}"

    suppliers = sorted(g["supplier_id"].unique().tolist())
    licenses = sorted(g["license_no"].unique().tolist())
    models = sorted(g["model"].unique().tolist())
    customers = sorted(g["customer_id"].unique().tolist())

    nodes = [ns("Supplier", s) for s in suppliers] + \
            [ns("License", s) for s in licenses] + \
            [ns("Model", s) for s in models] + \
            [ns("Customer", s) for s in customers]

    node_index = {n: i for i, n in enumerate(nodes)}

    # edges: Supplier->License, License->Model, Model->Customer
    e1 = g.groupby(["supplier_id", "license_no"])["quantity"].sum().reset_index()
    e2 = g.groupby(["license_no", "model"])["quantity"].sum().reset_index()
    e3 = g.groupby(["model", "customer_id"])["quantity"].sum().reset_index()

    src, tgt, val = [], [], []

    for _, r in e1.iterrows():
        src.append(node_index[ns("Supplier", r["supplier_id"])])
        tgt.append(node_index[ns("License", r["license_no"])])
        val.append(float(r["quantity"]))

    for _, r in e2.iterrows():
        src.append(node_index[ns("License", r["license_no"])])
        tgt.append(node_index[ns("Model", r["model"])])
        val.append(float(r["quantity"]))

    for _, r in e3.iterrows():
        src.append(node_index[ns("Model", r["model"])])
        tgt.append(node_index[ns("Customer", r["customer_id"])])
        val.append(float(r["quantity"]))

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=12,
            label=nodes,
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
        )
    )])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_layered_network(df: pd.DataFrame, max_nodes_per_layer: int = 40) -> go.Figure:
    """
    Simple deterministic layered "network graph" without external deps:
    x = layer (supplier/license/model/customer)
    y = evenly spaced per node
    edges drawn as line segments between layers
    """
    if df.empty:
        return go.Figure()

    g = df.groupby(["supplier_id", "license_no", "model", "customer_id"], dropna=False)["quantity"].sum().reset_index()
    g = g.fillna("∅")

    def top_values(col: str) -> List[str]:
        agg = df.groupby(col, dropna=False)["quantity"].sum().reset_index().fillna("∅")
        agg = agg.sort_values("quantity", ascending=False)
        return agg[col].astype(str).head(max_nodes_per_layer).tolist()

    suppliers = top_values("supplier_id")
    licenses = top_values("license_no")
    models = top_values("model")
    customers = top_values("customer_id")

    # filter edges to keep graph readable
    g2 = g[
        g["supplier_id"].astype(str).isin(suppliers) &
        g["license_no"].astype(str).isin(licenses) &
        g["model"].astype(str).isin(models) &
        g["customer_id"].astype(str).isin(customers)
    ].copy()

    layers = [
        ("Supplier", suppliers, 0.0),
        ("License", licenses, 1.0),
        ("Model", models, 2.0),
        ("Customer", customers, 3.0),
    ]

    pos = {}
    node_x, node_y, node_text, node_color = [], [], [], []
    for lname, nodes, x in layers:
        n = max(1, len(nodes))
        for i, v in enumerate(nodes):
            y = (i / (n - 1)) if n > 1 else 0.5
            key = f"{lname}:{v}"
            pos[key] = (x, y)
            node_x.append(x)
            node_y.append(y)
            node_text.append(key)
            node_color.append(lname)

    # edges (three hops)
    def add_edges(pairs: pd.DataFrame, a_name: str, b_name: str, a_col: str, b_col: str) -> Tuple[List[float], List[float]]:
        ex, ey = [], []
        for _, r in pairs.iterrows():
            a = f"{a_name}:{r[a_col]}"
            b = f"{b_name}:{r[b_col]}"
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        return ex, ey

    e1 = g2.groupby(["supplier_id", "license_no"])["quantity"].sum().reset_index()
    e2 = g2.groupby(["license_no", "model"])["quantity"].sum().reset_index()
    e3 = g2.groupby(["model", "customer_id"])["quantity"].sum().reset_index()

    ex1, ey1 = add_edges(e1, "Supplier", "License", "supplier_id", "license_no")
    ex2, ey2 = add_edges(e2, "License", "Model", "license_no", "model")
    ex3, ey3 = add_edges(e3, "Model", "Customer", "model", "customer_id")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex1 + ex2 + ex3,
        y=ey1 + ey2 + ey3,
        mode="lines",
        line=dict(width=1, color="rgba(180,180,200,0.35)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[s.split(":", 1)[1] for s in node_text],
        textposition="middle right",
        marker=dict(size=9, color=node_color),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["Supplier", "License", "Model", "Customer"],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def build_timeseries(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    tmp = df.copy()
    tmp = tmp.dropna(subset=["deliver_date"])
    if tmp.empty:
        return go.Figure()
    g = tmp.groupby(pd.Grouper(key="deliver_date", freq="D"))["quantity"].sum().reset_index()
    fig = px.line(g, x="deliver_date", y="quantity", markers=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_top_suppliers(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if df.empty:
        return go.Figure()
    g = df.groupby("supplier_id")["quantity"].sum().reset_index().sort_values("quantity", ascending=False).head(top_n)
    fig = px.bar(g, x="supplier_id", y="quantity")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_sunburst(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    tmp = df.fillna("∅").copy()
    fig = px.sunburst(
        tmp,
        path=["supplier_id", "license_no", "model", "customer_id"],
        values="quantity",
        color="supplier_id",
        maxdepth=4,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_heatmap(df: pd.DataFrame, top_suppliers: int = 20, top_models: int = 30) -> go.Figure:
    if df.empty:
        return go.Figure()

    s_top = df.groupby("supplier_id")["quantity"].sum().reset_index().sort_values("quantity", ascending=False)["supplier_id"].head(top_suppliers)
    m_top = df.groupby("model")["quantity"].sum().reset_index().sort_values("quantity", ascending=False)["model"].head(top_models)

    tmp = df[df["supplier_id"].isin(s_top) & df["model"].isin(m_top)].copy()
    if tmp.empty:
        return go.Figure()

    pivot = tmp.pivot_table(index="supplier_id", columns="model", values="quantity", aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ============================================================
# Summary + filtering
# ============================================================
def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"rows": 0}

    out = {
        "rows": int(len(df)),
        "total_quantity": int(df["quantity"].sum()) if "quantity" in df.columns else 0,
        "unique_suppliers": int(df["supplier_id"].nunique(dropna=True)),
        "unique_customers": int(df["customer_id"].nunique(dropna=True)),
        "unique_licenses": int(df["license_no"].nunique(dropna=True)),
        "unique_models": int(df["model"].nunique(dropna=True)),
    }

    # date range
    if "deliver_date" in df.columns:
        dmin = df["deliver_date"].min()
        dmax = df["deliver_date"].max()
        out["date_min"] = None if pd.isna(dmin) else str(pd.to_datetime(dmin).date())
        out["date_max"] = None if pd.isna(dmax) else str(pd.to_datetime(dmax).date())

    # tops
    def top_list(col: str, n=10):
        if col not in df.columns:
            return []
        g = df.groupby(col)["quantity"].sum().reset_index().sort_values("quantity", ascending=False).head(n)
        return [{"value": str(r[col]), "quantity": int(r["quantity"])} for _, r in g.iterrows()]

    out["top_suppliers"] = top_list("supplier_id", 10)
    out["top_customers"] = top_list("customer_id", 10)
    out["top_models"] = top_list("model", 10)
    out["top_licenses"] = top_list("license_no", 10)
    out["top_categories"] = top_list("category", 10)
    return out


def apply_filters(df: pd.DataFrame,
                  supplier_ids: List[str],
                  license_nos: List[str],
                  models: List[str],
                  customer_ids: List[str],
                  date_range: Optional[Tuple[datetime.date, datetime.date]],
                  query: str) -> pd.DataFrame:
    if df.empty:
        return df

    tmp = df.copy()

    if supplier_ids:
        tmp = tmp[tmp["supplier_id"].isin(supplier_ids)]
    if license_nos:
        tmp = tmp[tmp["license_no"].isin(license_nos)]
    if models:
        tmp = tmp[tmp["model"].isin(models)]
    if customer_ids:
        tmp = tmp[tmp["customer_id"].isin(customer_ids)]

    if date_range and "deliver_date" in tmp.columns:
        start, end = date_range
        d = tmp["deliver_date"]
        tmp = tmp[(d.notna()) & (d.dt.date >= start) & (d.dt.date <= end)]

    q = (query or "").strip().lower()
    if q:
        hay_cols = ["device_name", "category", "udi_di", "lot_no", "serial_no", "license_no", "model", "supplier_id", "customer_id"]
        existing = [c for c in hay_cols if c in tmp.columns]
        if existing:
            mask = False
            for c in existing:
                mask = mask | tmp[c].astype("string").str.lower().fillna("").str.contains(q, regex=False)
            tmp = tmp[mask]

    return tmp.reset_index(drop=True)


# ============================================================
# Agents YAML handling
# ============================================================
def standardize_agents_obj(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {"version": "1.0", "agents": []}
    if isinstance(obj, list):
        obj = {"version": "1.0", "agents": obj}
    if not isinstance(obj, dict):
        return {"version": "1.0", "agents": []}

    version = str(obj.get("version", "1.0"))
    agents = obj.get("agents", obj.get("items", obj.get("data", [])))
    if not isinstance(agents, list):
        agents = []

    fixed = []
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            continue
        fixed.append({
            "id": str(a.get("id") or f"agent_{i+1}"),
            "name": str(a.get("name") or a.get("id") or f"Agent {i+1}"),
            "description": str(a.get("description") or ""),
            "provider": str((a.get("provider") or "openai")).lower(),
            "model": str(a.get("model") or "gpt-4o-mini"),
            "temperature": float(a.get("temperature", 0.2)),
            "max_tokens": int(a.get("max_tokens", 2500)),
            "system_prompt": str(a.get("system_prompt") or ""),
            "user_prompt": str(a.get("user_prompt") or "Analyze the dataset context."),
        })
    return {"version": version, "agents": fixed}


def load_agents_yaml(raw_text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        obj = yaml.safe_load(raw_text) if raw_text.strip() else {"version": "1.0", "agents": []}
        cfg = standardize_agents_obj(obj)
        return cfg, None
    except Exception as e:
        return {"version": "1.0", "agents": []}, str(e)


def dump_agents_yaml(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


# ============================================================
# Streamlit App Shell
# ============================================================
st.set_page_config(page_title="WOW Distribution Analysis", layout="wide")


def ss_init():
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("style", PAINTER_STYLES[0])

    st.session_state.setdefault("api_keys", {})

    st.session_state.setdefault("nav", "Dashboard")

    st.session_state.setdefault("raw_df", pd.DataFrame())
    st.session_state.setdefault("std_df", pd.DataFrame(columns=CANON))
    st.session_state.setdefault("std_report", "")

    st.session_state.setdefault("source_mode", "default")
    st.session_state.setdefault("paste_text", "")
    st.session_state.setdefault("viz_instructions", "")

    st.session_state.setdefault("agents_yaml_text", DEFAULT_AGENTS_YAML)
    st.session_state.setdefault("agents_cfg", {"version": "1.0", "agents": []})
    st.session_state.setdefault("agent_runs", [])  # list of dicts with output + edited_output
    st.session_state.setdefault("agent_input_override", "")


ss_init()

lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]
st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)
st.markdown("<div class='fab'>WOW</div><div class='fab-sub'>Distribution Studio</div>", unsafe_allow_html=True)


# ============================================================
# WOW Status chips
# ============================================================
def status_chip(label: str, env_primary: str) -> str:
    key, src = get_api_key(env_primary)
    if src == "env":
        dot = "var(--ok)"; stt = t(lang, "managed_by_env")
    elif src == "session":
        dot = "var(--warn)"; stt = t(lang, "session_key")
    else:
        dot = "var(--bad)"; stt = t(lang, "missing_key")
    return f"<span class='chip'><span class='dot' style='background:{dot}'></span>{label}: {stt}</span>"


def dataset_chip(df: pd.DataFrame) -> str:
    rows = len(df) if df is not None else 0
    return f"<span class='chip'><span class='dot'></span>{t(lang,'rows')}: <span class='coral'>{rows}</span></span>"


# ============================================================
# Top bar
# ============================================================
top = st.container()
with top:
    c1, c2, c3 = st.columns([2.2, 3.8, 1.4], vertical_alignment="center")
    with c1:
        st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'app_title')}</h3></div>", unsafe_allow_html=True)

    with c2:
        chips = ""
        chips += status_chip("OpenAI", "OPENAI_API_KEY")
        chips += status_chip("Gemini", "GEMINI_API_KEY")
        chips += status_chip("Anthropic", "ANTHROPIC_API_KEY")
        chips += status_chip("xAI", "XAI_API_KEY")
        chips += dataset_chip(st.session_state["std_df"])
        st.markdown(f"<div class='wow-card'>{chips}</div>", unsafe_allow_html=True)

    with c3:
        with st.popover(t(lang, "settings")):
            st.session_state["theme"] = st.radio(
                t(lang, "theme"), ["dark", "light"],
                index=0 if st.session_state["theme"] == "dark" else 1,
                key="set_theme",
            )
            st.session_state["lang"] = st.radio(
                t(lang, "language"), ["en", "zh-TW"],
                index=0 if st.session_state["lang"] == "en" else 1,
                key="set_lang",
            )

            style_names = [s["name"] for s in PAINTER_STYLES]
            curr = st.session_state["style"]["name"]
            ix = style_names.index(curr) if curr in style_names else 0
            pick = st.selectbox(t(lang, "style"), style_names, index=ix, key="set_style")
            st.session_state["style"] = next(s for s in PAINTER_STYLES if s["name"] == pick)

            if st.button(t(lang, "jackpot"), use_container_width=True, key="style_jackpot"):
                st.session_state["style"] = jackpot_style()
                st.rerun()

# re-inject after setting changes
lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]
st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)


# ============================================================
# Sidebar: API keys (rule: show input only if NOT in env)
# ============================================================
with st.sidebar:
    st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'api_keys')}</h4></div>", unsafe_allow_html=True)

    def api_key_block(label: str, env_primary: str):
        key, src = get_api_key(env_primary)
        if src == "env":
            st.markdown(f"<div class='wow-mini'><b>{label}</b><br/>{t(lang,'managed_by_env')}</div>", unsafe_allow_html=True)
            return
        val = st.text_input(f"{label} key", value=st.session_state["api_keys"].get(env_primary, ""), type="password", key=f"key_{env_primary}")
        if val:
            st.session_state["api_keys"][env_primary] = val

    api_key_block("OpenAI", "OPENAI_API_KEY")
    api_key_block("Gemini", "GEMINI_API_KEY")
    api_key_block("Anthropic", "ANTHROPIC_API_KEY")
    api_key_block("xAI", "XAI_API_KEY")

    st.divider()
    if st.button(t(lang, "clear_session"), use_container_width=True, key="clear_session_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ============================================================
# Navigation
# ============================================================
nav = st.columns([1.2, 1.2, 2.6], vertical_alignment="center")
with nav[0]:
    page = st.selectbox(
        "Navigation",
        [t(lang, "nav_data"), t(lang, "nav_dashboard"), t(lang, "nav_agents")],
        index=1,
        key="nav_select",
    )
with nav[1]:
    st.session_state["source_mode"] = st.selectbox(
        t(lang, "data_source"),
        ["default", "paste", "upload"],
        format_func=lambda x: t(lang, "use_default") if x == "default" else t(lang, x),
        key="source_mode_sel",
    )
with nav[2]:
    st.session_state["viz_instructions"] = st.text_input(
        t(lang, "viz_instructions"),
        value=st.session_state["viz_instructions"],
        placeholder="e.g., focus on anomalies, show monthly trend, highlight top 5 suppliers, compare license patterns...",
        key="viz_inst",
    )


# ============================================================
# Data loading panel (shared)
# ============================================================
def load_data_ui():
    source_mode = st.session_state["source_mode"]
    auto_std = st.checkbox(t(lang, "auto_standardize"), value=True, key="auto_std_cb")

    raw_df = None
    if source_mode == "default":
        raw_df = parse_dataset_blob(DEFAULT_DISTRIBUTION_CSV)
        st.caption("Loaded default dataset embedded in app.py")

    elif source_mode == "paste":
        st.session_state["paste_text"] = st.text_area(
            f"{t(lang,'paste')} dataset (CSV/JSON)",
            value=st.session_state["paste_text"],
            height=160,
            key="paste_area",
        )
        if st.button(t(lang, "parse_load"), use_container_width=True, key="parse_paste_btn"):
            try:
                raw_df = parse_dataset_blob(st.session_state["paste_text"])
                st.session_state["raw_df"] = raw_df
            except Exception as e:
                st.error(f"Parse failed: {e}")
        raw_df = st.session_state["raw_df"]

    else:
        up = st.file_uploader(f"{t(lang,'upload')} dataset file (txt/csv/json)", type=["txt", "csv", "json"], key="upload_file")
        if up:
            try:
                raw_df = parse_dataset_blob(up.read(), filename=up.name)
                st.session_state["raw_df"] = raw_df
            except Exception as e:
                st.error(f"Parse failed: {e}")
        raw_df = st.session_state["raw_df"]

    if raw_df is None or raw_df.empty:
        st.warning("No dataset loaded yet.")
        return

    st.session_state["raw_df"] = raw_df

    # preview raw
    with st.expander(t(lang, "preview_20") + " (raw)", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True, height=240)

    if auto_std:
        std_df, rep = standardize_distribution_df(raw_df)
        st.session_state["std_df"] = std_df
        st.session_state["std_report"] = rep


# ============================================================
# Pages
# ============================================================
def page_data_studio():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'nav_data')}</h3></div>", unsafe_allow_html=True)

    load_data_ui()

    std_df = st.session_state["std_df"]
    st.divider()

    st.markdown(f"<div class='wow-mini'><b>{t(lang,'standardization_report')}</b></div>", unsafe_allow_html=True)
    st.markdown(st.session_state.get("std_report", ""))

    st.markdown(f"<div class='wow-mini'><b>{t(lang,'preview_20')} (standardized)</b></div>", unsafe_allow_html=True)
    st.dataframe(std_df.head(20), use_container_width=True, height=280)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            t(lang, "download_csv"),
            data=std_df.to_csv(index=False).encode("utf-8"),
            file_name="distribution_standardized.csv",
            use_container_width=True,
            key="dl_csv",
        )
    with c2:
        st.download_button(
            t(lang, "download_json"),
            data=df_to_json_records(std_df).encode("utf-8"),
            file_name="distribution_standardized.json",
            use_container_width=True,
            key="dl_json",
        )


def page_dashboard():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'dashboard')}</h3></div>", unsafe_allow_html=True)

    load_data_ui()
    df = st.session_state["std_df"]
    if df is None or df.empty:
        return

    # Filters
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'filters')}</b></div>", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        supplier_ids = st.multiselect(
            t(lang, "supplier_id"),
            options=sorted(df["supplier_id"].dropna().unique().tolist()),
            default=[],
            key="flt_supplier",
        )
    with f2:
        license_nos = st.multiselect(
            t(lang, "license_no"),
            options=sorted(df["license_no"].dropna().unique().tolist()),
            default=[],
            key="flt_license",
        )
    with f3:
        models = st.multiselect(
            t(lang, "model"),
            options=sorted(df["model"].dropna().unique().tolist()),
            default=[],
            key="flt_model",
        )
    with f4:
        customer_ids = st.multiselect(
            t(lang, "customer_id"),
            options=sorted(df["customer_id"].dropna().unique().tolist()),
            default=[],
            key="flt_customer",
        )

    # date range + search
    cA, cB = st.columns([1.2, 2.8])
    with cA:
        dmin = df["deliver_date"].min()
        dmax = df["deliver_date"].max()
        if pd.isna(dmin) or pd.isna(dmax):
            date_rng = None
            st.caption(t(lang, "date_range") + ": (no valid dates)")
        else:
            date_rng = st.date_input(
                t(lang, "date_range"),
                value=(pd.to_datetime(dmin).date(), pd.to_datetime(dmax).date()),
                key="flt_date",
            )
            if isinstance(date_rng, tuple) and len(date_rng) == 2:
                date_rng = (date_rng[0], date_rng[1])
            else:
                date_rng = None
    with cB:
        q = st.text_input(t(lang, "search"), value="", key="flt_query")

    df_f = apply_filters(df, supplier_ids, license_nos, models, customer_ids, date_rng, q)

    # Summary
    st.divider()
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'summary')}</b></div>", unsafe_allow_html=True)
    s = compute_summary(df_f)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t(lang, "rows"), f"{s.get('rows', 0)}")
    k2.metric(t(lang, "quantity"), f"{s.get('total_quantity', 0)}")
    k3.metric("Suppliers", f"{s.get('unique_suppliers', 0)}")
    k4.metric("Customers", f"{s.get('unique_customers', 0)}")
    k5.metric("Models", f"{s.get('unique_models', 0)}")

    if s.get("date_min") and s.get("date_max"):
        st.caption(f"Date range: {s['date_min']} → {s['date_max']}")

    # Visualizations (6)
    st.divider()
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'dashboard')}</b></div>", unsafe_allow_html=True)

    st.caption(t(lang, "viz_sankey"))
    st.plotly_chart(build_sankey(df_f), use_container_width=True, key="viz_sankey")

    st.caption(t(lang, "viz_network"))
    st.plotly_chart(build_layered_network(df_f), use_container_width=True, key="viz_network")

    c1, c2 = st.columns(2)
    with c1:
        st.caption(t(lang, "viz_timeseries"))
        st.plotly_chart(build_timeseries(df_f), use_container_width=True, key="viz_ts")
    with c2:
        st.caption(t(lang, "viz_top_suppliers"))
        st.plotly_chart(build_top_suppliers(df_f), use_container_width=True, key="viz_top_sup")

    st.caption(t(lang, "viz_sunburst"))
    st.plotly_chart(build_sunburst(df_f), use_container_width=True, key="viz_sunburst")

    st.caption(t(lang, "viz_heatmap"))
    st.plotly_chart(build_heatmap(df_f), use_container_width=True, key="viz_heatmap")

    # Filtered table + download
    st.divider()
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'table')}</b></div>", unsafe_allow_html=True)
    st.dataframe(df_f, use_container_width=True, height=420)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            t(lang, "download_csv"),
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="distribution_filtered.csv",
            use_container_width=True,
            key="dl_csv_filtered",
        )
    with d2:
        st.download_button(
            t(lang, "download_json"),
            data=df_to_json_records(df_f).encode("utf-8"),
            file_name="distribution_filtered.json",
            use_container_width=True,
            key="dl_json_filtered",
        )


def page_agents():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'nav_agents')}</h3></div>", unsafe_allow_html=True)

    load_data_ui()
    df = st.session_state["std_df"]
    if df is None or df.empty:
        return

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown(f"<div class='wow-mini'><b>agents.yaml</b></div>", unsafe_allow_html=True)

        st.session_state["agents_yaml_text"] = st.text_area(
            "agents.yaml",
            value=st.session_state["agents_yaml_text"],
            height=360,
            key="agents_yaml_editor",
        )
        cfg, err = load_agents_yaml(st.session_state["agents_yaml_text"])
        if err:
            st.error(f"YAML invalid: {err}")
            return
        st.session_state["agents_cfg"] = cfg

        st.download_button(
            "Download agents.yaml",
            data=dump_agents_yaml(cfg),
            file_name="agents.yaml",
            use_container_width=True,
            key="dl_agents_yaml",
        )

        st.divider()
        st.markdown(f"<div class='wow-mini'><b>{t(lang,'input_to_agent')}</b></div>", unsafe_allow_html=True)

        # input context to agent: dataset summary + sample + optional viz instructions
        summary = compute_summary(df)
        sample = df.head(20).to_csv(index=False)
        base_context = f"""DATASET SUMMARY (JSON):
{json.dumps(summary, ensure_ascii=False, indent=2)}

USER INSTRUCTIONS (optional):
{st.session_state.get('viz_instructions','')}

SAMPLE RECORDS (CSV, first 20):
{sample}
"""
        st.session_state["agent_input_override"] = st.text_area(
            t(lang, "input_to_agent"),
            value=st.session_state.get("agent_input_override") or base_context,
            height=240,
            key="agent_input_override_area",
        )

    with right:
        st.markdown(f"<div class='wow-mini'><b>{t(lang,'agent_pipeline')}</b></div>", unsafe_allow_html=True)

        agents = st.session_state["agents_cfg"].get("agents", [])
        if not agents:
            st.warning("No agents in config.")
            return

        agent_names = [f"{a.get('name')} ({a.get('id')})" for a in agents]
        pick = st.selectbox(t(lang, "agent"), agent_names, index=0, key="agent_pick")
        agent = agents[agent_names.index(pick)]

        pmap = provider_model_map()
        provider = st.selectbox(
            t(lang, "provider"),
            list(pmap.keys()),
            index=list(pmap.keys()).index(agent.get("provider", "openai")) if agent.get("provider", "openai") in pmap else 0,
            key="agent_provider",
        )
        model = st.selectbox(t(lang, "model_select"), pmap[provider], index=0, key="agent_model")
        max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=3500, step=256, key="agent_max_tokens")
        temperature = st.slider(t(lang, "temperature"), 0.0, 1.0, float(agent.get("temperature", 0.2)), 0.05, key="agent_temp")

        system_prompt = st.text_area(t(lang, "system_prompt"), value=str(agent.get("system_prompt", "")), height=150, key="agent_system_prompt")
        user_prompt = st.text_area(t(lang, "user_prompt"), value=str(agent.get("user_prompt", "")), height=150, key="agent_user_prompt")

        if st.button(t(lang, "run_agent"), use_container_width=True, key="run_agent_btn"):
            env_primary = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[provider]
            api_key, src = get_api_key(env_primary)
            if not api_key:
                st.error(f"{env_primary} missing.")
            else:
                try:
                    full_user = f"{user_prompt}\n\n---\nINPUT:\n{st.session_state['agent_input_override']}"
                    with st.spinner("Running agent..."):
                        out = call_llm_text(
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            system=system_prompt,
                            user=full_user,
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                        )
                    st.session_state["agent_runs"].append({
                        "ts": datetime.datetime.utcnow().isoformat(),
                        "agent_id": agent.get("id", ""),
                        "agent_name": agent.get("name", ""),
                        "provider": provider,
                        "model": model,
                        "max_tokens": int(max_tokens),
                        "temperature": float(temperature),
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "input": st.session_state["agent_input_override"],
                        "output": out,
                        "edited_output": out,
                    })
                    st.success("Agent completed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent run failed: {e}")
                    st.code(traceback.format_exc())

        st.divider()
        if st.session_state["agent_runs"]:
            for idx in range(len(st.session_state["agent_runs"]) - 1, -1, -1):
                run = st.session_state["agent_runs"][idx]
                st.markdown(
                    f"<div class='wow-mini'><b>Run {idx+1}</b> — {run['agent_name']} "
                    f"(<span class='coral'>{run['provider']}/{run['model']}</span>)</div>",
                    unsafe_allow_html=True,
                )
                tabs = st.tabs([t(lang, "output"), t(lang, "edit_for_next")])
                with tabs[0]:
                    st.markdown(run["output"] if run["output"] else "—")
                with tabs[1]:
                    st.session_state["agent_runs"][idx]["edited_output"] = st.text_area(
                        t(lang, "edit_for_next"),
                        value=run["edited_output"],
                        height=220,
                        key=f"edit_out_{idx}",
                    )
                    if st.button("Use this edited output as next agent input", use_container_width=True, key=f"use_next_{idx}"):
                        st.session_state["agent_input_override"] = run["edited_output"]
                        st.success("Set as next agent input.")
                        st.rerun()


# ============================================================
# Router
# ============================================================
if page == t(lang, "nav_data"):
    page_data_studio()
elif page == t(lang, "nav_agents"):
    page_agents()
else:
    page_dashboard()
