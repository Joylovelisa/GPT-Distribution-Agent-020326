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

# Optional: enables Plotly node click -> details panel
# If not installed, the app still runs (fallback: hover + manual filters)
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False


# ============================================================
# Constants
# ============================================================
CORAL = "#FF7F50"

KEY_ENV_CANDIDATES = {
    "OPENAI_API_KEY": ["OPENAI_API_KEY"],
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"],
    "XAI_API_KEY": ["XAI_API_KEY"],
}

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini"]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
XAI_MODELS = ["grok-4-fast-reasoning", "grok-3-mini"]


# ============================================================
# Default dataset (user provided)
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


# ============================================================
# Default SKILL.md + agents.yaml (fallback)
# ============================================================
DEFAULT_SKILL_MD = """# SKILL — WOW 配送/流向分析代理共用系統提示詞（System Prompt）

你是「WOW 配送/流向分析工作室」的資深資料分析與資料視覺化顧問（偏資料工程 + 分析 + 視覺化敘事）。
你將被提供：資料摘要、樣本、與使用者可選指令。

## 最高原則（不可違反）
1. 禁止捏造：不得虛構數值、趨勢、結論；缺資料請標示 Gap。
2. 可追溯：所有結論要說明依據（輸入摘要/樣本/明確規則）。
3. 隱私最小揭露：避免大量列出識別碼；必要時遮蔽。
4. 產出可落地：提供清楚的 KPI 定義、圖表規格、互動與降載策略。

## 預設輸出格式
- 繁體中文 Markdown
- 建議章節：目標 / 假設與限制 / 方法 / 結果(若可) / 視覺化建議 / 資料品質風險 / 下一步
"""

DEFAULT_AGENTS_YAML = """version: "1.0"
agents:
  - id: distribution_insights_analyst
    name: 配送洞察分析師
    description: 主管摘要 + 洞察 + 異常假說 + 視覺化建議。
    provider: openai
    model: gpt-4o-mini
    temperature: 0.2
    max_tokens: 3500
    system_prompt: |
      你是資深配送/流向分析顧問。輸出繁體中文 Markdown。不得捏造。
    user_prompt: |
      請分析資料摘要與樣本，輸出：
      - KPI 概覽（若輸入提供）
      - 供應商/客戶/型號/許可證重點
      - 可能異常與風險（以需檢核方式表達）
      - 建議的儀表板視覺化與篩選器
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
        "nav_config": "Config Studio (agents.yaml / SKILL.md)",
        "nav_compare": "Compare Two Datasets",
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
        "clear_session": "Clear session",
        "apply": "Apply",
        "run": "Run",
        "reset": "Reset to defaults",
        "upload_yaml": "Upload YAML",
        "upload_md": "Upload MD",
        "download_yaml": "Download agents.yaml",
        "download_md": "Download SKILL.md",
        "paste_yaml": "Paste agents.yaml",
        "paste_md": "Paste SKILL.md",
        "standardize_now": "Standardize now",
        "diff": "Diff",
        "invalid_yaml": "YAML invalid",
        "dataset_a": "Dataset A",
        "dataset_b": "Dataset B",
        "compare_summary": "Comparison Summary (Markdown)",
        "ai_prompt": "AI Prompt (keep with dataset)",
        "ai_run": "Run AI Summary",
        "click_hint": "Click a node to show detailed records (requires streamlit-plotly-events).",
    },
    "zh-TW": {
        "app_title": "WOW 配送/流向分析工作室",
        "nav_data": "資料工作室",
        "nav_dashboard": "互動儀表板",
        "nav_agents": "代理工作室",
        "nav_config": "設定工作室（agents.yaml / SKILL.md）",
        "nav_compare": "兩份資料集比較",
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
        "clear_session": "清除 session",
        "apply": "套用",
        "run": "執行",
        "reset": "重置為預設",
        "upload_yaml": "上傳 YAML",
        "upload_md": "上傳 MD",
        "download_yaml": "下載 agents.yaml",
        "download_md": "下載 SKILL.md",
        "paste_yaml": "貼上 agents.yaml",
        "paste_md": "貼上 SKILL.md",
        "standardize_now": "立即標準化",
        "diff": "差異（Diff）",
        "invalid_yaml": "YAML 無效",
        "dataset_a": "資料集 A",
        "dataset_b": "資料集 B",
        "compare_summary": "比較摘要（Markdown）",
        "ai_prompt": "AI 提示詞（綁定資料集）",
        "ai_run": "執行 AI 摘要",
        "click_hint": "點擊網路圖節點可顯示詳細資料（需要 streamlit-plotly-events）。",
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
# WOW CSS
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
      .coral {{ color: var(--coral); font-weight: 900; }}
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
      div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
      }}
    </style>
    """


# ============================================================
# API keys + LLM routing
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
# Parsing + Standardization (Distribution dataset)
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
    "supplier_id": ["supplierid", "supplier_id", "supplier", "vendor", "供應商", "供應商代碼", "SupplierID"],
    "deliver_date": ["deliverdate", "deliver_date", "date", "shipment_date", "delivery_date", "出貨日", "交貨日", "日期", "Deliverdate"],
    "customer_id": ["customerid", "customer_id", "customer", "client", "買方", "客戶", "客戶代碼", "CustomerID"],
    "license_no": ["licenseno", "license_no", "license", "permit", "許可證", "許可證字號", "LicenseNo"],
    "category": ["category", "class", "product_category", "分類", "類別", "Category"],
    "udi_di": ["udid", "udi_di", "udi", "di", "主識別碼", "UDI", "UDID"],
    "device_name": ["devicename", "device_name", "device", "product_name", "品名", "裝置名稱", "DeviceNAME"],
    "lot_no": ["lotno", "lot_no", "lot", "batch", "批號", "LotNO"],
    "serial_no": ["serno", "serial_no", "serial", "sn", "序號", "SerNo"],
    "model": ["model", "model_no", "model_number", "型號", "Model"],
    "quantity": ["number", "qty", "quantity", "count", "數量", "Number"],
}


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
            for k in ["data", "records", "items", "rows", "dataset", "datasets"]:
                if k in obj and isinstance(obj[k], list):
                    obj = obj[k]
                    break
            if isinstance(obj, dict):
                obj = [obj]
        if not isinstance(obj, list):
            raise ValueError("JSON must be a list of objects (or a wrapper containing a list).")
        return pd.DataFrame(obj)

    return pd.read_csv(io.StringIO(text))


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
    if re.fullmatch(r"\d{8}", s):
        try:
            return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        except Exception:
            return None
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

    for c in ["supplier_id", "customer_id", "license_no", "category", "udi_di", "device_name", "lot_no", "serial_no", "model"]:
        out[c] = out[c].apply(_clean_quotes).astype("string").str.strip()

    def to_int(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return 0
        try:
            return int(float(str(x).replace(",", "").strip()))
        except Exception:
            return 0

    out["quantity"] = out["quantity"].apply(to_int)
    out["deliver_date"] = out["deliver_date"].apply(_parse_deliver_date)

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
            if str(v).strip() and str(v).strip().lower() != "nan":
                return True
        return False

    out = out[out.apply(has_signal, axis=1)].reset_index(drop=True)

    missing = [c for c in CANON if out[c].isna().all()]
    report_lines += ["", f"**Rows:** {len(out)}", f"**Original columns:** {len(original_cols)}"]
    if missing:
        report_lines += ["", "### Missing Canonical Fields", "- " + "\n- ".join([f"`{m}`" for m in missing])]

    if "deliver_date" in out.columns:
        out = out.sort_values("deliver_date", na_position="last").reset_index(drop=True)

    return out, "\n".join(report_lines)


def df_to_json_records(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


# ============================================================
# Summary + filtering
# ============================================================
def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"rows": 0}

    out = {
        "rows": int(len(df)),
        "total_quantity": int(df["quantity"].sum()) if "quantity" in df.columns else 0,
        "unique_suppliers": int(df["supplier_id"].nunique(dropna=True)),
        "unique_customers": int(df["customer_id"].nunique(dropna=True)),
        "unique_licenses": int(df["license_no"].nunique(dropna=True)),
        "unique_models": int(df["model"].nunique(dropna=True)),
    }

    if "deliver_date" in df.columns:
        dmin = df["deliver_date"].min()
        dmax = df["deliver_date"].max()
        out["date_min"] = None if pd.isna(dmin) else str(pd.to_datetime(dmin).date())
        out["date_max"] = None if pd.isna(dmax) else str(pd.to_datetime(dmax).date())

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
    if df is None or df.empty:
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
# Visualizations (Original 6 + WOW 5 features)
# ============================================================
def build_sankey(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()

    g = df.groupby(["supplier_id", "license_no", "model", "customer_id"], dropna=False)["quantity"].sum().reset_index()
    g = g.fillna("∅")

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
        node=dict(pad=12, thickness=12, label=nodes),
        link=dict(source=src, target=tgt, value=val),
    )])
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_layered_network(df: pd.DataFrame, max_nodes_per_layer: int = 40) -> go.Figure:
    """
    BUGFIX: Plotly marker.color cannot accept arbitrary strings in graph_objects Scatter.
            We map layer names to HEX colors explicitly.
    """
    if df is None or df.empty:
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

    layer_colors = {
        "Supplier": "#FF7F50",
        "License": "#5DADE2",
        "Model": "#58D68D",
        "Customer": "#AF7AC5",
    }

    pos = {}
    node_x, node_y, node_text, node_color_hex = [], [], [], []
    for lname, nodes, x in layers:
        n = max(1, len(nodes))
        for i, v in enumerate(nodes):
            y = (i / (n - 1)) if n > 1 else 0.5
            key = f"{lname}:{v}"
            pos[key] = (x, y)
            node_x.append(x)
            node_y.append(y)
            node_text.append(key)
            node_color_hex.append(layer_colors.get(lname, "#CCCCCC"))

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

    display_labels = []
    for s in node_text:
        v = s.split(":", 1)[1]
        display_labels.append(v if len(v) <= 14 else (v[:12] + "…"))

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=display_labels,
        textposition="middle right",
        marker=dict(size=10, color=node_color_hex, line=dict(width=1, color="rgba(255,255,255,0.25)")),
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
    if df is None or df.empty:
        return go.Figure()
    tmp = df.dropna(subset=["deliver_date"]).copy()
    if tmp.empty:
        return go.Figure()
    g = tmp.groupby(pd.Grouper(key="deliver_date", freq="D"))["quantity"].sum().reset_index()
    fig = px.line(g, x="deliver_date", y="quantity", markers=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_top_suppliers(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    g = df.groupby("supplier_id")["quantity"].sum().reset_index().sort_values("quantity", ascending=False).head(top_n)
    fig = px.bar(g, x="supplier_id", y="quantity")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_sunburst(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
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
    if df is None or df.empty:
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


# WOW feature 7
def build_pareto(df: pd.DataFrame, col: str, top_n: int = 20) -> go.Figure:
    if df is None or df.empty or col not in df.columns:
        return go.Figure()
    g = df.groupby(col)["quantity"].sum().reset_index().sort_values("quantity", ascending=False)
    g = g.head(top_n).copy()
    total = float(g["quantity"].sum()) if len(g) else 0.0
    if total <= 0:
        return go.Figure()
    g["cum_pct"] = (g["quantity"].cumsum() / total) * 100.0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g[col].astype(str), y=g["quantity"], name="Quantity"))
    fig.add_trace(go.Scatter(x=g[col].astype(str), y=g["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Quantity"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h"),
    )
    return fig


# WOW feature 8
def build_supplier_customer_matrix(df: pd.DataFrame, top_suppliers: int = 15, top_customers: int = 25) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    s_top = df.groupby("supplier_id")["quantity"].sum().sort_values(ascending=False).head(top_suppliers).index
    c_top = df.groupby("customer_id")["quantity"].sum().sort_values(ascending=False).head(top_customers).index
    tmp = df[df["supplier_id"].isin(s_top) & df["customer_id"].isin(c_top)]
    if tmp.empty:
        return go.Figure()
    pivot = tmp.pivot_table(index="supplier_id", columns="customer_id", values="quantity", aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Viridis")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# WOW feature 9
def build_weekday_week_heatmap(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    tmp = df.dropna(subset=["deliver_date"]).copy()
    if tmp.empty:
        return go.Figure()
    iso = tmp["deliver_date"].dt.isocalendar()
    tmp["iso_week"] = iso.week.astype(int)
    tmp["weekday"] = tmp["deliver_date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tmp["weekday"] = pd.Categorical(tmp["weekday"], categories=weekday_order, ordered=True)
    pivot = tmp.pivot_table(index="weekday", columns="iso_week", values="quantity", aggfunc="sum", fill_value=0).sort_index()
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Plasma")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# WOW feature 10
def build_treemap(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    tmp = df.fillna("∅").copy()
    fig = px.treemap(tmp, path=["supplier_id", "license_no", "model"], values="quantity", color="supplier_id")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# WOW feature 11
def build_quantity_box(df: pd.DataFrame, by_col: str = "supplier_id", top_n: int = 15) -> go.Figure:
    if df is None or df.empty or by_col not in df.columns:
        return go.Figure()
    top = df.groupby(by_col)["quantity"].sum().sort_values(ascending=False).head(top_n).index
    tmp = df[df[by_col].isin(top)].copy()
    if tmp.empty:
        return go.Figure()
    fig = px.box(tmp, x=by_col, y="quantity", points="all")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ============================================================
# Agents YAML standardization + optional LLM standardizer
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

    pmap = provider_model_map()
    valid_providers = set(pmap.keys())

    fixed = []
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            continue

        aid = a.get("id") or a.get("agent_id") or a.get("key") or f"agent_{i+1}"
        name = a.get("name") or a.get("title") or str(aid)
        desc = a.get("description") or a.get("desc") or ""

        provider = (a.get("provider") or a.get("vendor") or "openai")
        provider = str(provider).lower().strip()
        if provider not in valid_providers:
            provider = "openai"

        model = a.get("model") or a.get("llm") or "gpt-4o-mini"
        model = str(model).strip()
        if model not in pmap.get(provider, []):
            model = pmap.get(provider, ["gpt-4o-mini"])[0]

        temp = a.get("temperature", 0.2)
        mx = a.get("max_tokens", a.get("max_output_tokens", 2500))
        system_prompt = a.get("system_prompt") or a.get("system") or a.get("instructions") or a.get("prompt") or ""
        user_prompt = a.get("user_prompt") or a.get("user") or a.get("task") or "請分析提供的內容。"

        def to_float(x, default=0.2):
            try:
                return float(x)
            except Exception:
                return default

        def to_int(x, default=2500):
            try:
                return int(x)
            except Exception:
                return default

        fixed.append({
            "id": str(aid),
            "name": str(name),
            "description": str(desc),
            "provider": provider,
            "model": model,
            "temperature": to_float(temp, 0.2),
            "max_tokens": to_int(mx, 2500),
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
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


def unified_diff(a: str, b: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        a.splitlines(keepends=True),
        b.splitlines(keepends=True),
        fromfile="before.yaml",
        tofile="after.yaml",
    ))


def llm_standardize_agents_yaml(raw_yaml: str, provider: str, model: str, api_key: str, lang: str) -> str:
    schema = {
        "version": "1.0",
        "agents": [{
            "id": "string (unique)",
            "name": "string",
            "description": "string",
            "provider": "enum(openai, gemini, anthropic, xai)",
            "model": "string",
            "temperature": 0.2,
            "max_tokens": 2500,
            "system_prompt": "string",
            "user_prompt": "string",
        }]
    }
    sys = "You convert arbitrary YAML agent configs into a strict standard YAML schema. Output YAML only."
    if lang == "zh-TW":
        sys = "你負責把任意 agents YAML 轉成嚴格標準 schema。只輸出 YAML。"
    user = f"""Convert the following YAML into this strict schema. Fill missing fields conservatively.

SCHEMA (example):
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

INPUT YAML:
```yaml
{raw_yaml}
```
"""
    out = call_llm_text(provider, model, api_key, sys, user, max_tokens=5000, temperature=0.0)
    return out.strip()


# ============================================================
# Compare Two Datasets: helpers
# ============================================================
def compute_timeseries(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["deliver_date", "quantity"])
    tmp = df.dropna(subset=["deliver_date"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["deliver_date", "quantity"])
    g = tmp.groupby(pd.Grouper(key="deliver_date", freq=freq))["quantity"].sum().reset_index()
    g = g.sort_values("deliver_date")
    return g


def build_compare_kpi_bar(sA: Dict[str, Any], sB: Dict[str, Any], labelA: str, labelB: str) -> go.Figure:
    keys = [
        ("rows", "Rows"),
        ("total_quantity", "Total Qty"),
        ("unique_suppliers", "Suppliers"),
        ("unique_licenses", "Licenses"),
        ("unique_models", "Models"),
        ("unique_customers", "Customers"),
    ]
    data = []
    for k, name in keys:
        data.append({"metric": name, "dataset": labelA, "value": float(sA.get(k, 0))})
        data.append({"metric": name, "dataset": labelB, "value": float(sB.get(k, 0))})
    dfm = pd.DataFrame(data)
    fig = px.bar(dfm, x="metric", y="value", color="dataset", barmode="group")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_compare_timeseries(tsA: pd.DataFrame, tsB: pd.DataFrame, labelA: str, labelB: str) -> go.Figure:
    if tsA.empty and tsB.empty:
        return go.Figure()
    out = []
    if not tsA.empty:
        a = tsA.copy()
        a["dataset"] = labelA
        out.append(a)
    if not tsB.empty:
        b = tsB.copy()
        b["dataset"] = labelB
        out.append(b)
    dfc = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["deliver_date", "quantity", "dataset"])
    fig = px.line(dfc, x="deliver_date", y="quantity", color="dataset", markers=True)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_compare_top_bar(dfA: pd.DataFrame, dfB: pd.DataFrame, col: str, top_n: int, labelA: str, labelB: str) -> go.Figure:
    if (dfA is None or dfA.empty) and (dfB is None or dfB.empty):
        return go.Figure()

    def top(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=[col, "quantity"])
        g = df.groupby(col)["quantity"].sum().reset_index()
        return g

    gA = top(dfA).rename(columns={"quantity": "qty_A"})
    gB = top(dfB).rename(columns={"quantity": "qty_B"})
    m = pd.merge(gA, gB, on=col, how="outer").fillna(0)
    m["total"] = m["qty_A"] + m["qty_B"]
    m = m.sort_values("total", ascending=False).head(top_n)

    plot = pd.DataFrame({
        col: m[col].astype(str).tolist() * 2,
        "dataset": [labelA] * len(m) + [labelB] * len(m),
        "quantity": m["qty_A"].tolist() + m["qty_B"].tolist(),
    })
    fig = px.bar(plot, x=col, y="quantity", color="dataset", barmode="group")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def compare_summary_markdown(sA: Dict[str, Any], sB: Dict[str, Any], labelA: str, labelB: str) -> str:
    def fmt(v): return f"{v:,}" if isinstance(v, int) else str(v)
    dqA = fmt(sA.get("total_quantity", 0))
    dqB = fmt(sB.get("total_quantity", 0))
    diff_qty = int(sA.get("total_quantity", 0)) - int(sB.get("total_quantity", 0))
    sign = "+" if diff_qty >= 0 else ""
    rowsA = int(sA.get("rows", 0))
    rowsB = int(sB.get("rows", 0))

    md = []
    md.append(f"# 兩份配送資料比較摘要：{labelA} vs {labelB}")
    md.append("")
    md.append("## KPI 對比（彙總）")
    md.append(f"- **Rows**：{labelA} = {rowsA:,}；{labelB} = {rowsB:,}")
    md.append(f"- **Total Quantity**：{labelA} = {dqA}；{labelB} = {dqB}；差異（A-B）= **{sign}{diff_qty:,}**")
    md.append(f"- **Unique Suppliers**：{labelA} = {int(sA.get('unique_suppliers',0)):,}；{labelB} = {int(sB.get('unique_suppliers',0)):,}")
    md.append(f"- **Unique Licenses**：{labelA} = {int(sA.get('unique_licenses',0)):,}；{labelB} = {int(sB.get('unique_licenses',0)):,}")
    md.append(f"- **Unique Models**：{labelA} = {int(sA.get('unique_models',0)):,}；{labelB} = {int(sB.get('unique_models',0)):,}")
    md.append(f"- **Unique Customers**：{labelA} = {int(sA.get('unique_customers',0)):,}；{labelB} = {int(sB.get('unique_customers',0)):,}")
    md.append("")
    md.append("## 日期範圍")
    md.append(f"- {labelA}：{sA.get('date_min','—')} → {sA.get('date_max','—')}")
    md.append(f"- {labelB}：{sB.get('date_min','—')} → {sB.get('date_max','—')}")
    md.append("")
    md.append("## Top 10（依 quantity）")
    md.append(f"### {labelA} Top Suppliers")
    md.append("\n".join([f"- {x['value']}: {x['quantity']:,}" for x in sA.get("top_suppliers", [])]) or "- —")
    md.append(f"\n### {labelB} Top Suppliers")
    md.append("\n".join([f"- {x['value']}: {x['quantity']:,}" for x in sB.get("top_suppliers", [])]) or "- —")
    md.append("")
    md.append("## 解讀建議（保守）")
    md.append("- 若 A/B 的時間範圍不同，請先對齊日期區間後再比較趨勢。")
    md.append("- 若資料來源不同（不同系統/不同清洗規則），建議先做欄位一致性與重複入帳檢核。")
    md.append("- 供應商/客戶/型號的 TopN 變化，適合搭配 Network / Sankey 觀察路徑結構差異。")
    return "\n".join(md).strip()


def node_filter(df: pd.DataFrame, layer: str, value: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    layer = layer.lower().strip()
    col_map = {
        "supplier": "supplier_id",
        "license": "license_no",
        "model": "model",
        "customer": "customer_id",
    }
    col = col_map.get(layer)
    if not col or col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df[col].astype(str) == str(value)].copy().reset_index(drop=True)


def build_compare_network_clickable(dfA: pd.DataFrame, dfB: pd.DataFrame, labelA: str, labelB: str,
                                   max_nodes_per_layer: int = 30) -> Tuple[go.Figure, List[Dict[str, str]]]:
    """
    Combined layered network:
    - Node color indicates membership: A-only / B-only / both
    - Returns (fig, node_meta_list) where node_meta_list index = pointIndex for click lookup
    """
    if (dfA is None or dfA.empty) and (dfB is None or dfB.empty):
        return go.Figure(), []

    def top_values(df: pd.DataFrame, col: str) -> List[str]:
        if df is None or df.empty:
            return []
        agg = df.groupby(col, dropna=False)["quantity"].sum().reset_index().fillna("∅")
        agg = agg.sort_values("quantity", ascending=False)
        return agg[col].astype(str).head(max_nodes_per_layer).tolist()

    # union top nodes across A/B for each layer
    suppliers = sorted(set(top_values(dfA, "supplier_id") + top_values(dfB, "supplier_id")))
    licenses = sorted(set(top_values(dfA, "license_no") + top_values(dfB, "license_no")))
    models = sorted(set(top_values(dfA, "model") + top_values(dfB, "model")))
    customers = sorted(set(top_values(dfA, "customer_id") + top_values(dfB, "customer_id")))

    layers = [
        ("Supplier", suppliers, 0.0),
        ("License", licenses, 1.0),
        ("Model", models, 2.0),
        ("Customer", customers, 3.0),
    ]

    # membership sets (for node coloring)
    def uniq_set(df: pd.DataFrame, col: str) -> set:
        if df is None or df.empty:
            return set()
        return set(df[col].dropna().astype(str).unique().tolist())

    sA = uniq_set(dfA, "supplier_id"); sB = uniq_set(dfB, "supplier_id")
    lA = uniq_set(dfA, "license_no"); lB = uniq_set(dfB, "license_no")
    mA = uniq_set(dfA, "model"); mB = uniq_set(dfB, "model")
    cA = uniq_set(dfA, "customer_id"); cB = uniq_set(dfB, "customer_id")

    presence_map = {
        "Supplier": (sA, sB),
        "License": (lA, lB),
        "Model": (mA, mB),
        "Customer": (cA, cB),
    }

    # color scheme: A-only / B-only / both
    col_A = "#5DADE2"
    col_B = "#F5B041"
    col_both = CORAL
    col_unknown = "#C0C0C0"

    pos = {}
    node_x, node_y, node_text, node_color, node_meta = [], [], [], [], []
    for lname, nodes, x in layers:
        n = max(1, len(nodes))
        setA, setB = presence_map.get(lname, (set(), set()))
        for i, v in enumerate(nodes):
            y = (i / (n - 1)) if n > 1 else 0.5
            key = f"{lname}:{v}"
            pos[key] = (x, y)
            node_x.append(x); node_y.append(y); node_text.append(key)

            vv = str(v)
            inA = vv in setA
            inB = vv in setB
            if inA and inB:
                node_color.append(col_both)
                presence = "both"
            elif inA:
                node_color.append(col_A)
                presence = "A"
            elif inB:
                node_color.append(col_B)
                presence = "B"
            else:
                node_color.append(col_unknown)
                presence = "none"

            node_meta.append({"layer": lname, "value": vv, "presence": presence, "node_key": key})

    def edges(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df is None or df.empty:
            empty = pd.DataFrame(columns=["a", "b", "quantity"])
            return empty, empty, empty
        g = df.groupby(["supplier_id", "license_no", "model", "customer_id"], dropna=False)["quantity"].sum().reset_index().fillna("∅")
        e1 = g.groupby(["supplier_id", "license_no"])["quantity"].sum().reset_index()
        e2 = g.groupby(["license_no", "model"])["quantity"].sum().reset_index()
        e3 = g.groupby(["model", "customer_id"])["quantity"].sum().reset_index()
        return e1, e2, e3

    e1A, e2A, e3A = edges(dfA)
    e1B, e2B, e3B = edges(dfB)

    # build a union edge list (limit by top flow to keep readable)
    def union_edges(eA: pd.DataFrame, eB: pd.DataFrame, left: str, right: str, left_tag: str, right_tag: str,
                    top_m: int = 300) -> pd.DataFrame:
        a = eA.rename(columns={left: "left", right: "right", "quantity": "qA"}) if not eA.empty else pd.DataFrame(columns=["left", "right", "qA"])
        b = eB.rename(columns={left: "left", right: "right", "quantity": "qB"}) if not eB.empty else pd.DataFrame(columns=["left", "right", "qB"])
        m = pd.merge(a, b, on=["left", "right"], how="outer").fillna(0)
        m["qT"] = m["qA"] + m["qB"]
        m = m.sort_values("qT", ascending=False).head(top_m)
        m["from"] = m["left"].astype(str)
        m["to"] = m["right"].astype(str)
        m["from_key"] = left_tag + ":" + m["from"]
        m["to_key"] = right_tag + ":" + m["to"]
        return m

    U1 = union_edges(e1A, e1B, "supplier_id", "license_no", "Supplier", "License")
    U2 = union_edges(e2A, e2B, "license_no", "model", "License", "Model")
    U3 = union_edges(e3A, e3B, "model", "customer_id", "Model", "Customer")
    U = pd.concat([U1, U2, U3], ignore_index=True)

    # build edges lines
    ex, ey = [], []
    for _, r in U.iterrows():
        a = r["from_key"]; b = r["to_key"]
        if a not in pos or b not in pos:
            continue
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    # Node label shortening
    display_labels = []
    for s in node_text:
        v = s.split(":", 1)[1]
        display_labels.append(v if len(v) <= 14 else (v[:12] + "…"))

    # build figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(width=1, color="rgba(180,180,200,0.28)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Node trace: pointIndex aligns to node_meta list
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=display_labels,
        textposition="middle right",
        marker=dict(size=11, color=node_color, line=dict(width=1, color="rgba(255,255,255,0.25)")),
        hovertext=[f"{m['node_key']} | presence={m['presence']} ({labelA}={m['presence'] in ['A','both']}, {labelB}={m['presence'] in ['B','both']})" for m in node_meta],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["Supplier", "License", "Model", "Customer"],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        title=f"Compare Network: {labelA} vs {labelB}",
    )
    return fig, node_meta


# ============================================================
# Streamlit setup + Session init
# ============================================================
st.set_page_config(page_title="WOW Distribution Analysis", layout="wide")


def ss_init():
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("lang", "zh-TW")
    st.session_state.setdefault("style", PAINTER_STYLES[0])
    st.session_state.setdefault("api_keys", {})

    st.session_state.setdefault("source_mode", "default")
    st.session_state.setdefault("paste_text", "")
    st.session_state.setdefault("raw_df", pd.DataFrame())
    st.session_state.setdefault("std_df", pd.DataFrame(columns=CANON))
    st.session_state.setdefault("std_report", "")

    st.session_state.setdefault("viz_instructions", "")

    # config files
    st.session_state.setdefault("skill_md_text", DEFAULT_SKILL_MD)
    st.session_state.setdefault("agents_yaml_text", DEFAULT_AGENTS_YAML)
    st.session_state.setdefault("agents_cfg", {"version": "1.0", "agents": []})
    st.session_state.setdefault("agents_yaml_diff", "")

    # agent execution
    st.session_state.setdefault("agent_runs", [])
    st.session_state.setdefault("agent_input_override", "")

    # compare module state
    st.session_state.setdefault("cmpA_mode", "paste")  # paste/upload/default
    st.session_state.setdefault("cmpB_mode", "paste")
    st.session_state.setdefault("cmpA_paste", "")
    st.session_state.setdefault("cmpB_paste", "")
    st.session_state.setdefault("cmpA_raw", pd.DataFrame())
    st.session_state.setdefault("cmpB_raw", pd.DataFrame())
    st.session_state.setdefault("cmpA_std", pd.DataFrame(columns=CANON))
    st.session_state.setdefault("cmpB_std", pd.DataFrame(columns=CANON))
    st.session_state.setdefault("cmpA_report", "")
    st.session_state.setdefault("cmpB_report", "")

    st.session_state.setdefault("cmpA_prompt", "請摘要資料集 A 的關鍵特徵與異常假說（保守，不捏造）。")
    st.session_state.setdefault("cmpB_prompt", "請摘要資料集 B 的關鍵特徵與異常假說（保守，不捏造）。")
    st.session_state.setdefault("cmpCompare_prompt", "請比較 A vs B 的差異（KPI、TopN、趨勢、路徑結構），並提出下一步。")

    st.session_state.setdefault("cmp_provider", "openai")
    st.session_state.setdefault("cmp_model", OPENAI_MODELS[0])
    st.session_state.setdefault("cmp_max_tokens", 4500)
    st.session_state.setdefault("cmp_temperature", 0.2)

    st.session_state.setdefault("cmp_ai_note", "")
    st.session_state.setdefault("cmp_clicked_node", None)  # dict: {layer,value,presence}


ss_init()

lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]

st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)
st.markdown("<div class='fab'>WOW</div><div class='fab-sub'>Distribution Studio</div>", unsafe_allow_html=True)


# ============================================================
# Status chips
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
    rows = len(df) if isinstance(df, pd.DataFrame) else 0
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

lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]
st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)


# ============================================================
# Sidebar: API Keys
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
# Navigation + dataset mode + optional instructions
# ============================================================
nav = st.columns([1.7, 1.3, 3.0], vertical_alignment="center")
with nav[0]:
    page = st.selectbox(
        "Navigation",
        [t(lang, "nav_dashboard"), t(lang, "nav_data"), t(lang, "nav_agents"), t(lang, "nav_compare"), t(lang, "nav_config")],
        index=0,
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
        placeholder="例：請強調異常路徑、顯示每週趨勢、只看 Top 5 供應商、聚焦某許可證…",
        key="viz_inst",
    )


# ============================================================
# Shared: Data loading UI (single dataset)
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

    with st.expander(t(lang, "preview_20") + " (raw)", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True, height=240)

    if auto_std:
        std_df, rep = standardize_distribution_df(raw_df)
        st.session_state["std_df"] = std_df
        st.session_state["std_report"] = rep


# ============================================================
# Config Studio: agents.yaml + SKILL.md module
# ============================================================
def config_studio_page():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'nav_config')}</h3></div>", unsafe_allow_html=True)

    tabs = st.tabs(["agents.yaml", "SKILL.md", "Agent Editor (UI)"])

    with tabs[0]:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'paste_yaml')}</b></div>", unsafe_allow_html=True)
            st.session_state["agents_yaml_text"] = st.text_area(
                "agents.yaml",
                value=st.session_state["agents_yaml_text"],
                height=420,
                key="agents_yaml_editor",
            )

            u = st.file_uploader(t(lang, "upload_yaml"), type=["yml", "yaml"], key="upload_agents_yaml")
            if u:
                st.session_state["agents_yaml_text"] = u.read().decode("utf-8", errors="ignore")
                st.rerun()

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(t(lang, "standardize_now"), use_container_width=True, key="std_agents_now"):
                    raw = st.session_state["agents_yaml_text"]
                    cfg, err = load_agents_yaml(raw)
                    if err:
                        st.error(f"{t(lang,'invalid_yaml')}: {err}")
                    else:
                        standardized = dump_agents_yaml(cfg)
                        st.session_state["agents_yaml_diff"] = unified_diff(raw, standardized)
                        st.session_state["agents_yaml_text"] = standardized
                        st.session_state["agents_cfg"] = cfg
                        st.success("Standardized (heuristic).")
                        st.rerun()
            with c2:
                if st.button(t(lang, "reset"), use_container_width=True, key="reset_agents_yaml"):
                    st.session_state["agents_yaml_text"] = DEFAULT_AGENTS_YAML
                    cfg, _ = load_agents_yaml(DEFAULT_AGENTS_YAML)
                    st.session_state["agents_cfg"] = cfg
                    st.session_state["agents_yaml_diff"] = ""
                    st.rerun()
            with c3:
                st.download_button(
                    t(lang, "download_yaml"),
                    data=st.session_state["agents_yaml_text"].encode("utf-8"),
                    file_name="agents.yaml",
                    use_container_width=True,
                    key="dl_agents_yaml",
                )

        with right:
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'diff')}</b></div>", unsafe_allow_html=True)
            st.code(st.session_state.get("agents_yaml_diff", "") or "(no diff yet)")

            st.divider()
            st.markdown(f"<div class='wow-mini'><b>LLM Auto-standardize（可選，較強）</b></div>", unsafe_allow_html=True)
            pmap = provider_model_map()
            prov = st.selectbox(t(lang, "provider"), list(pmap.keys()), index=0, key="llm_std_provider")
            model = st.selectbox(t(lang, "model_select"), pmap[prov], index=0, key="llm_std_model")

            if st.button("Use LLM to Standardize agents.yaml", use_container_width=True, key="llm_std_run"):
                env_primary = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[prov]
                api_key, _ = get_api_key(env_primary)
                if not api_key:
                    st.error(f"{env_primary} missing.")
                else:
                    try:
                        raw = st.session_state["agents_yaml_text"]
                        out = llm_standardize_agents_yaml(raw, prov, model, api_key, lang=lang)
                        cfg, err = load_agents_yaml(out)
                        if err:
                            st.error(f"LLM output still invalid: {err}")
                        else:
                            standardized = dump_agents_yaml(cfg)
                            st.session_state["agents_yaml_diff"] = unified_diff(raw, standardized)
                            st.session_state["agents_yaml_text"] = standardized
                            st.session_state["agents_cfg"] = cfg
                            st.success("LLM standardized & loaded.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"LLM standardize failed: {e}")
                        st.code(traceback.format_exc())

    with tabs[1]:
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'paste_md')}</b></div>", unsafe_allow_html=True)
            st.session_state["skill_md_text"] = st.text_area(
                "SKILL.md",
                value=st.session_state["skill_md_text"],
                height=420,
                key="skill_md_editor",
            )
            u = st.file_uploader(t(lang, "upload_md"), type=["md", "txt"], key="upload_skill_md")
            if u:
                st.session_state["skill_md_text"] = u.read().decode("utf-8", errors="ignore")
                st.rerun()

            c1, c2 = st.columns(2)
            with c1:
                if st.button(t(lang, "reset"), use_container_width=True, key="reset_skill_md"):
                    st.session_state["skill_md_text"] = DEFAULT_SKILL_MD
                    st.rerun()
            with c2:
                st.download_button(
                    t(lang, "download_md"),
                    data=st.session_state["skill_md_text"].encode("utf-8"),
                    file_name="SKILL.md",
                    use_container_width=True,
                    key="dl_skill_md",
                )

        with right:
            st.markdown(f"<div class='wow-mini'><b>Preview</b></div>", unsafe_allow_html=True)
            st.markdown(st.session_state["skill_md_text"] or "")

    with tabs[2]:
        cfg, err = load_agents_yaml(st.session_state["agents_yaml_text"])
        if err:
            st.error(f"{t(lang,'invalid_yaml')}: {err}")
            return
        st.session_state["agents_cfg"] = cfg

        agents = cfg.get("agents", [])
        if not agents:
            st.info("No agents found.")
            return

        st.markdown("<div class='wow-mini'><b>編輯代理（不用直接寫 YAML）</b></div>", unsafe_allow_html=True)
        names = [f"{a.get('name','')} ({a.get('id','')})" for a in agents]
        pick = st.selectbox(t(lang, "agent"), names, index=0, key="agent_editor_pick")
        a = agents[names.index(pick)]

        pmap = provider_model_map()
        provider = st.selectbox(t(lang, "provider"), list(pmap.keys()),
                                index=list(pmap.keys()).index(a.get("provider", "openai")) if a.get("provider", "openai") in pmap else 0,
                                key="agent_editor_provider")
        model = st.selectbox(t(lang, "model_select"), pmap[provider], index=0, key="agent_editor_model")
        max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=int(a.get("max_tokens", 2500)), step=256, key="agent_editor_max")
        temperature = st.slider(t(lang, "temperature"), 0.0, 1.0, float(a.get("temperature", 0.2)), 0.05, key="agent_editor_temp")

        a["name"] = st.text_input("Name", value=a.get("name", ""), key="agent_editor_name")
        a["description"] = st.text_input("Description", value=a.get("description", ""), key="agent_editor_desc")
        a["provider"] = provider
        a["model"] = model
        a["max_tokens"] = int(max_tokens)
        a["temperature"] = float(temperature)
        a["system_prompt"] = st.text_area(t(lang, "system_prompt"), value=a.get("system_prompt", ""), height=160, key="agent_editor_system")
        a["user_prompt"] = st.text_area(t(lang, "user_prompt"), value=a.get("user_prompt", ""), height=160, key="agent_editor_user")

        if st.button(t(lang, "apply"), use_container_width=True, key="agent_editor_apply"):
            agents[names.index(pick)] = a
            cfg["agents"] = agents
            st.session_state["agents_cfg"] = cfg
            st.session_state["agents_yaml_text"] = dump_agents_yaml(cfg)
            st.success("Updated agent + regenerated agents.yaml")
            st.rerun()


# ============================================================
# Data Studio
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
            key="dl_csv_std",
        )
    with c2:
        st.download_button(
            t(lang, "download_json"),
            data=df_to_json_records(std_df).encode("utf-8"),
            file_name="distribution_standardized.json",
            use_container_width=True,
            key="dl_json_std",
        )


# ============================================================
# Dashboard (all original features retained)
# ============================================================
def page_dashboard():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'dashboard')}</h3></div>", unsafe_allow_html=True)
    load_data_ui()

    df = st.session_state["std_df"]
    if df is None or df.empty:
        return

    st.markdown(f"<div class='wow-mini'><b>{t(lang,'filters')}</b></div>", unsafe_allow_html=True)

    supplier_opts = sorted(df["supplier_id"].dropna().unique().tolist())
    license_opts = sorted(df["license_no"].dropna().unique().tolist())
    model_opts = sorted(df["model"].dropna().unique().tolist())
    customer_opts = sorted(df["customer_id"].dropna().unique().tolist())

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        supplier_ids = st.multiselect(t(lang, "supplier_id"), supplier_opts, default=[], key="flt_supplier")
    with f2:
        license_nos = st.multiselect(t(lang, "license_no"), license_opts, default=[], key="flt_license")
    with f3:
        models = st.multiselect(t(lang, "model"), model_opts, default=[], key="flt_model")
    with f4:
        customer_ids = st.multiselect(t(lang, "customer_id"), customer_opts, default=[], key="flt_customer")

    cA, cB = st.columns([1.2, 2.8])
    with cA:
        dmin = df["deliver_date"].min()
        dmax = df["deliver_date"].max()
        date_rng = None
        if pd.isna(dmin) or pd.isna(dmax):
            st.caption(t(lang, "date_range") + "：（無有效日期）")
        else:
            default_range = (pd.to_datetime(dmin).date(), pd.to_datetime(dmax).date())
            picked = st.date_input(t(lang, "date_range"), value=default_range, key="flt_date")
            if isinstance(picked, tuple) and len(picked) == 2:
                date_rng = (picked[0], picked[1])
            else:
                date_rng = default_range
    with cB:
        q = st.text_input(t(lang, "search"), value="", key="flt_query")

    df_f = apply_filters(df, supplier_ids, license_nos, models, customer_ids, date_rng, q)

    st.divider()
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'summary')}</b></div>", unsafe_allow_html=True)
    s = compute_summary(df_f)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t(lang, "rows"), f"{s.get('rows', 0)}")
    k2.metric(t(lang, "quantity"), f"{s.get('total_quantity', 0)}")
    k3.metric("Suppliers", f"{s.get('unique_suppliers', 0)}")
    k4.metric("Customers", f"{s.get('unique_customers', 0)}")
    k5.metric("Models", f"{s.get('unique_models', 0)}")

    st.divider()
    tabs = st.tabs([
        "Flow (Sankey / Network)",
        "Trends (Time / Rhythm)",
        "Structure (Sunburst / Treemap / Heatmaps)",
        "Risk Lens (Pareto / Variability)",
        t(lang, "table"),
    ])

    with tabs[0]:
        st.markdown("<div class='wow-mini'><b>1) Sankey：Supplier → License → Model → Customer</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_sankey(df_f), use_container_width=True, key="viz_sankey")
        st.markdown("<div class='wow-mini'><b>2) 分層配送網路圖（Layered Network）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_layered_network(df_f), use_container_width=True, key="viz_network_fixed")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='wow-mini'><b>3) 配送趨勢（Daily）</b></div>", unsafe_allow_html=True)
            st.plotly_chart(build_timeseries(df_f), use_container_width=True, key="viz_ts")
        with c2:
            st.markdown("<div class='wow-mini'><b>WOW #9：週次 × 週內節奏（Heatmap）</b></div>", unsafe_allow_html=True)
            st.plotly_chart(build_weekday_week_heatmap(df_f), use_container_width=True, key="viz_week_rhythm")
        st.markdown("<div class='wow-mini'><b>4) Top 供應商（Bar）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_top_suppliers(df_f), use_container_width=True, key="viz_top_sup")

    with tabs[2]:
        st.markdown("<div class='wow-mini'><b>5) Sunburst（Supplier → License → Model → Customer）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_sunburst(df_f), use_container_width=True, key="viz_sunburst")
        st.markdown("<div class='wow-mini'><b>WOW #10：Treemap（Supplier → License → Model）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_treemap(df_f), use_container_width=True, key="viz_treemap")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='wow-mini'><b>6) Heatmap（Supplier × Model）</b></div>", unsafe_allow_html=True)
            st.plotly_chart(build_heatmap(df_f), use_container_width=True, key="viz_heatmap")
        with c2:
            st.markdown("<div class='wow-mini'><b>WOW #8：Supplier × Customer Flow Heatmap</b></div>", unsafe_allow_html=True)
            st.plotly_chart(build_supplier_customer_matrix(df_f), use_container_width=True, key="viz_sup_cust_matrix")

    with tabs[3]:
        st.markdown("<div class='wow-mini'><b>WOW #7：Pareto（Top Customers）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_pareto(df_f, "customer_id", top_n=20), use_container_width=True, key="viz_pareto_customer")
        st.markdown("<div class='wow-mini'><b>WOW #11：數量變異（Boxplot by Supplier）</b></div>", unsafe_allow_html=True)
        st.plotly_chart(build_quantity_box(df_f, by_col="supplier_id", top_n=15), use_container_width=True, key="viz_box_supplier")

    with tabs[4]:
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


# ============================================================
# Agent Studio (original features retained)
# ============================================================
def page_agents():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'nav_agents')}</h3></div>", unsafe_allow_html=True)
    load_data_ui()

    df = st.session_state["std_df"]
    if df is None or df.empty:
        return

    cfg, err = load_agents_yaml(st.session_state["agents_yaml_text"])
    if err:
        st.error(f"{t(lang,'invalid_yaml')}: {err}")
        st.caption("請到「設定工作室」先標準化 agents.yaml")
        return
    st.session_state["agents_cfg"] = cfg

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown(f"<div class='wow-mini'><b>{t(lang,'input_to_agent')}</b></div>", unsafe_allow_html=True)
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
            height=260,
            key="agent_input_override_area",
        )

        st.markdown(f"<div class='wow-mini'><b>SKILL.md（system 合併前預覽）</b></div>", unsafe_allow_html=True)
        st.caption("系統提示詞會以：SKILL.md + agent.system_prompt 合併後送出")
        st.text_area("SKILL.md", value=st.session_state["skill_md_text"], height=140, key="skill_preview_ro", disabled=True)

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

        max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=int(agent.get("max_tokens", 3500)), step=256, key="agent_max_tokens")
        temperature = st.slider(t(lang, "temperature"), 0.0, 1.0, float(agent.get("temperature", 0.2)), 0.05, key="agent_temp")

        system_prompt = st.text_area(t(lang, "system_prompt"), value=str(agent.get("system_prompt", "")), height=160, key="agent_system_prompt")
        user_prompt = st.text_area(t(lang, "user_prompt"), value=str(agent.get("user_prompt", "")), height=160, key="agent_user_prompt")

        if st.button(t(lang, "run_agent"), use_container_width=True, key="run_agent_btn"):
            env_primary = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[provider]
            api_key, _ = get_api_key(env_primary)
            if not api_key:
                st.error(f"{env_primary} missing.")
            else:
                try:
                    full_system = (st.session_state["skill_md_text"].strip() + "\n\n" + system_prompt.strip()).strip()
                    full_user = f"{user_prompt}\n\n---\nINPUT:\n{st.session_state['agent_input_override']}"
                    with st.spinner("Running agent..."):
                        out = call_llm_text(
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            system=full_system,
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
# NEW: Compare Two Datasets page (keeps all original features)
# ============================================================
def _load_compare_one(prefix: str, default_text: str = ""):
    """
    prefix: 'cmpA' or 'cmpB'
    Loads into:
      - {prefix}_raw (DataFrame)
      - {prefix}_std (DataFrame)
      - {prefix}_report (str)
    """
    mode_key = f"{prefix}_mode"
    paste_key = f"{prefix}_paste"
    raw_key = f"{prefix}_raw"
    std_key = f"{prefix}_std"
    rep_key = f"{prefix}_report"

    mode = st.session_state.get(mode_key, "paste")
    auto_std = True

    st.session_state[mode_key] = st.selectbox(
        "Mode",
        ["paste", "upload", "default"],
        index=["paste", "upload", "default"].index(mode),
        key=f"{prefix}_mode_sel",
    )

    mode = st.session_state[mode_key]
    raw_df = None

    if mode == "default":
        raw_df = parse_dataset_blob(DEFAULT_DISTRIBUTION_CSV)
        st.caption("Using built-in default dataset as placeholder.")

    elif mode == "paste":
        st.session_state[paste_key] = st.text_area(
            f"{t(lang,'paste')} dataset (CSV/JSON/TXT)",
            value=st.session_state.get(paste_key) or default_text,
            height=160,
            key=f"{prefix}_paste_area",
        )
        if st.button(t(lang, "parse_load"), use_container_width=True, key=f"{prefix}_parse_paste"):
            try:
                raw_df = parse_dataset_blob(st.session_state[paste_key])
                st.session_state[raw_key] = raw_df
            except Exception as e:
                st.error(f"Parse failed: {e}")
        raw_df = st.session_state.get(raw_key)

    else:
        up = st.file_uploader(f"{t(lang,'upload')} dataset (txt/csv/json)", type=["txt", "csv", "json"], key=f"{prefix}_upload_file")
        if up:
            try:
                raw_df = parse_dataset_blob(up.read(), filename=up.name)
                st.session_state[raw_key] = raw_df
            except Exception as e:
                st.error(f"Parse failed: {e}")
        raw_df = st.session_state.get(raw_key)

    if raw_df is None or raw_df.empty:
        st.warning("No dataset loaded yet.")
        return

    st.session_state[raw_key] = raw_df
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'preview_20')} (raw)</b></div>", unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True, height=240)

    if auto_std:
        std_df, rep = standardize_distribution_df(raw_df)
        st.session_state[std_key] = std_df
        st.session_state[rep_key] = rep

    st.markdown(f"<div class='wow-mini'><b>{t(lang,'preview_20')} (standardized)</b></div>", unsafe_allow_html=True)
    st.dataframe(st.session_state[std_key].head(20), use_container_width=True, height=240)


def compare_two_datasets_page():
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'nav_compare')}</h3></div>", unsafe_allow_html=True)

    top_tabs = st.tabs([t(lang, "dataset_a"), t(lang, "dataset_b"), "Compare Dashboard", "AI Prompts"])

    with top_tabs[0]:
        _load_compare_one("cmpA", default_text="")
        with st.expander(t(lang, "standardization_report"), expanded=False):
            st.markdown(st.session_state.get("cmpA_report", ""))

    with top_tabs[1]:
        _load_compare_one("cmpB", default_text="")
        with st.expander(t(lang, "standardization_report"), expanded=False):
            st.markdown(st.session_state.get("cmpB_report", ""))

    # Compare Dashboard (filters per dataset + 5 graphs + markdown)
    with top_tabs[2]:
        dfA = st.session_state.get("cmpA_std", pd.DataFrame(columns=CANON))
        dfB = st.session_state.get("cmpB_std", pd.DataFrame(columns=CANON))

        if (dfA is None or dfA.empty) and (dfB is None or dfB.empty):
            st.warning("請先在「資料集 A / B」頁籤載入資料。")
            return

        st.markdown("<div class='wow-mini'><b>各資料集篩選條件（獨立）</b></div>", unsafe_allow_html=True)
        colA, colB = st.columns(2, gap="large")

        def filter_ui(df: pd.DataFrame, key_prefix: str):
            if df is None or df.empty:
                st.info("No data")
                return [], [], [], [], None, ""
            supplier_opts = sorted(df["supplier_id"].dropna().unique().tolist())
            license_opts = sorted(df["license_no"].dropna().unique().tolist())
            model_opts = sorted(df["model"].dropna().unique().tolist())
            customer_opts = sorted(df["customer_id"].dropna().unique().tolist())

            supplier_ids = st.multiselect(t(lang, "supplier_id"), supplier_opts, default=[], key=f"{key_prefix}_supplier")
            license_nos = st.multiselect(t(lang, "license_no"), license_opts, default=[], key=f"{key_prefix}_license")
            models = st.multiselect(t(lang, "model"), model_opts, default=[], key=f"{key_prefix}_model")
            customer_ids = st.multiselect(t(lang, "customer_id"), customer_opts, default=[], key=f"{key_prefix}_customer")

            dmin = df["deliver_date"].min()
            dmax = df["deliver_date"].max()
            date_rng = None
            if pd.isna(dmin) or pd.isna(dmax):
                st.caption(t(lang, "date_range") + "：（無有效日期）")
            else:
                default_range = (pd.to_datetime(dmin).date(), pd.to_datetime(dmax).date())
                picked = st.date_input(t(lang, "date_range"), value=default_range, key=f"{key_prefix}_date")
                if isinstance(picked, tuple) and len(picked) == 2:
                    date_rng = (picked[0], picked[1])
                else:
                    date_rng = default_range

            q = st.text_input(t(lang, "search"), value="", key=f"{key_prefix}_q")
            return supplier_ids, license_nos, models, customer_ids, date_rng, q

        with colA:
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'dataset_a')} Filters</b></div>", unsafe_allow_html=True)
            a_sup, a_lic, a_mod, a_cus, a_date, a_q = filter_ui(dfA, "cmpAflt")
        with colB:
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'dataset_b')} Filters</b></div>", unsafe_allow_html=True)
            b_sup, b_lic, b_mod, b_cus, b_date, b_q = filter_ui(dfB, "cmpBflt")

        dfA_f = apply_filters(dfA, a_sup, a_lic, a_mod, a_cus, a_date, a_q) if dfA is not None else pd.DataFrame(columns=CANON)
        dfB_f = apply_filters(dfB, b_sup, b_lic, b_mod, b_cus, b_date, b_q) if dfB is not None else pd.DataFrame(columns=CANON)

        labelA = "A"
        labelB = "B"
        sA = compute_summary(dfA_f)
        sB = compute_summary(dfB_f)

        st.divider()
        st.markdown(f"<div class='wow-mini'><b>{t(lang,'compare_summary')}</b></div>", unsafe_allow_html=True)
        md = compare_summary_markdown(sA, sB, labelA, labelB)
        st.markdown(md)

        # 5 graphs (includes network graph)
        st.divider()
        st.markdown("<div class='wow-mini'><b>5 Graphs (Comparison)</b></div>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)
        with g1:
            st.caption("Graph 1) KPI Comparison")
            st.plotly_chart(build_compare_kpi_bar(sA, sB, labelA, labelB), use_container_width=True, key="cmp_kpi_bar")
        with g2:
            st.caption("Graph 2) Trend Overlay (Daily)")
            tsA = compute_timeseries(dfA_f, freq="D")
            tsB = compute_timeseries(dfB_f, freq="D")
            st.plotly_chart(build_compare_timeseries(tsA, tsB, labelA, labelB), use_container_width=True, key="cmp_ts_overlay")

        g3, g4 = st.columns(2)
        with g3:
            st.caption("Graph 3) Top Suppliers (A vs B)")
            st.plotly_chart(build_compare_top_bar(dfA_f, dfB_f, "supplier_id", 15, labelA, labelB), use_container_width=True, key="cmp_top_sup")
        with g4:
            st.caption("Graph 4) Top Models (A vs B)")
            st.plotly_chart(build_compare_top_bar(dfA_f, dfB_f, "model", 15, labelA, labelB), use_container_width=True, key="cmp_top_model")

        st.caption("Graph 5) Distribution Network Graph (click node → show details)")
        figN, node_meta = build_compare_network_clickable(dfA_f, dfB_f, labelA, labelB, max_nodes_per_layer=30)

        if HAS_PLOTLY_EVENTS:
            selected = plotly_events(figN, click_event=True, hover_event=False, select_event=False,
                                     override_height=620, override_width="100%")
            # selected: list of points, each has pointIndex
            if selected and "pointIndex" in selected[0]:
                idx = selected[0]["pointIndex"]
                if 0 <= idx < len(node_meta):
                    st.session_state["cmp_clicked_node"] = node_meta[idx]
        else:
            st.info(t(lang, "click_hint"))
            st.plotly_chart(figN, use_container_width=True, key="cmp_network_noevents")

        # Details panel for clicked node
        st.divider()
        st.markdown("<div class='wow-mini'><b>Node Details / Records</b></div>", unsafe_allow_html=True)
        clicked = st.session_state.get("cmp_clicked_node")
        if clicked:
            layer = clicked["layer"]
            value = clicked["value"]
            presence = clicked["presence"]
            st.markdown(f"- **Clicked**: `{layer}` = `{value}` | presence = **{presence}**")

            # map layer name -> filter column name
            layer_key = layer.lower()
            if layer_key.startswith("supplier"):
                key = "supplier"
            elif layer_key.startswith("license"):
                key = "license"
            elif layer_key.startswith("model"):
                key = "model"
            else:
                key = "customer"

            subA = node_filter(dfA_f, key, value)
            subB = node_filter(dfB_f, key, value)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='wow-mini'><b>Dataset A: matched rows = {len(subA)}</b></div>", unsafe_allow_html=True)
                if not subA.empty:
                    st.dataframe(subA.head(20), use_container_width=True, height=260)
                else:
                    st.write("—")
            with c2:
                st.markdown(f"<div class='wow-mini'><b>Dataset B: matched rows = {len(subB)}</b></div>", unsafe_allow_html=True)
                if not subB.empty:
                    st.dataframe(subB.head(20), use_container_width=True, height=260)
                else:
                    st.write("—")

            # show some related aggregation to help explain click
            st.markdown("<div class='wow-mini'><b>Related (Top connections)</b></div>", unsafe_allow_html=True)
            def related(dfsub: pd.DataFrame, title: str):
                if dfsub is None or dfsub.empty:
                    st.write(f"{title}: —")
                    return
                # pick next-step columns depending on layer
                if key == "supplier":
                    nxt = "license_no"
                elif key == "license":
                    nxt = "model"
                elif key == "model":
                    nxt = "customer_id"
                else:
                    nxt = "model"
                g = dfsub.groupby(nxt)["quantity"].sum().reset_index().sort_values("quantity", ascending=False).head(12)
                st.write(title)
                st.dataframe(g, use_container_width=True, height=240)

            r1, r2 = st.columns(2)
            with r1:
                related(subA, "A related")
            with r2:
                related(subB, "B related")
        else:
            st.caption("尚未選取節點（若未安裝 streamlit-plotly-events，可改用上方篩選器手動縮小範圍）。")

        # filtered table previews + downloads
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='wow-mini'><b>Dataset A (Filtered) Preview</b></div>", unsafe_allow_html=True)
            st.dataframe(dfA_f.head(20), use_container_width=True, height=260)
            st.download_button(
                "Download A (filtered) CSV",
                data=dfA_f.to_csv(index=False).encode("utf-8"),
                file_name="datasetA_filtered.csv",
                use_container_width=True,
                key="dl_cmpA_filtered_csv",
            )
        with c2:
            st.markdown("<div class='wow-mini'><b>Dataset B (Filtered) Preview</b></div>", unsafe_allow_html=True)
            st.dataframe(dfB_f.head(20), use_container_width=True, height=260)
            st.download_button(
                "Download B (filtered) CSV",
                data=dfB_f.to_csv(index=False).encode("utf-8"),
                file_name="datasetB_filtered.csv",
                use_container_width=True,
                key="dl_cmpB_filtered_csv",
            )

    # AI Prompts: keep prompts with datasets + compare AI summary
    with top_tabs[3]:
        st.markdown("<div class='wow-mini'><b>AI Prompts (keep with datasets)</b></div>", unsafe_allow_html=True)
        pmap = provider_model_map()
        st.session_state["cmp_provider"] = st.selectbox(t(lang, "provider"), list(pmap.keys()),
                                                        index=list(pmap.keys()).index(st.session_state["cmp_provider"]) if st.session_state["cmp_provider"] in pmap else 0,
                                                        key="cmp_provider_sel")
        st.session_state["cmp_model"] = st.selectbox(t(lang, "model_select"), pmap[st.session_state["cmp_provider"]],
                                                     index=0, key="cmp_model_sel")
        st.session_state["cmp_max_tokens"] = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000,
                                                             value=int(st.session_state["cmp_max_tokens"]), step=256, key="cmp_max_tokens")
        st.session_state["cmp_temperature"] = st.slider(t(lang, "temperature"), 0.0, 1.0, float(st.session_state["cmp_temperature"]), 0.05, key="cmp_temp")

        st.divider()
        st.session_state["cmpA_prompt"] = st.text_area(f"{t(lang,'dataset_a')} — {t(lang,'ai_prompt')}",
                                                       value=st.session_state["cmpA_prompt"], height=120, key="cmpA_prompt_area")
        st.session_state["cmpB_prompt"] = st.text_area(f"{t(lang,'dataset_b')} — {t(lang,'ai_prompt')}",
                                                       value=st.session_state["cmpB_prompt"], height=120, key="cmpB_prompt_area")
        st.session_state["cmpCompare_prompt"] = st.text_area(f"{t(lang,'compare_summary')} — {t(lang,'ai_prompt')}",
                                                             value=st.session_state["cmpCompare_prompt"], height=140, key="cmpCompare_prompt_area")

        st.caption("AI 會使用：A/B 的摘要（JSON）+ 前 20 筆樣本 + 你的提示詞；並遵守 SKILL.md 的「不得捏造」規則。")
        if st.button(t(lang, "ai_run"), use_container_width=True, key="cmp_ai_run_btn"):
            dfA = st.session_state.get("cmpA_std", pd.DataFrame(columns=CANON))
            dfB = st.session_state.get("cmpB_std", pd.DataFrame(columns=CANON))

            sA = compute_summary(dfA)
            sB = compute_summary(dfB)
            sampleA = dfA.head(20).to_csv(index=False) if dfA is not None else ""
            sampleB = dfB.head(20).to_csv(index=False) if dfB is not None else ""

            provider = st.session_state["cmp_provider"]
            model = st.session_state["cmp_model"]
            env_primary = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[provider]
            api_key, _ = get_api_key(env_primary)
            if not api_key:
                st.error(f"{env_primary} missing.")
            else:
                try:
                    sys = (st.session_state["skill_md_text"].strip() + "\n\n" +
                           "你將比較兩份配送資料集 A/B。輸出繁體中文 Markdown。不得捏造。若輸入不足請標示 Gap。").strip()

                    user = f"""## 資料集 A 提示詞
{st.session_state['cmpA_prompt']}

## 資料集 B 提示詞
{st.session_state['cmpB_prompt']}

## 比較提示詞
{st.session_state['cmpCompare_prompt']}

---

### A 摘要（JSON）
{json.dumps(sA, ensure_ascii=False, indent=2)}

### B 摘要（JSON）
{json.dumps(sB, ensure_ascii=False, indent=2)}

### A 樣本（前 20 筆 CSV）
{sampleA}

### B 樣本（前 20 筆 CSV）
{sampleB}
"""
                    with st.spinner("Running AI..."):
                        out = call_llm_text(
                            provider=provider,
                            model=model,
                            api_key=api_key,
                            system=sys,
                            user=user,
                            max_tokens=int(st.session_state["cmp_max_tokens"]),
                            temperature=float(st.session_state["cmp_temperature"]),
                        )
                    st.session_state["cmp_ai_note"] = out
                    st.success("AI summary generated.")
                except Exception as e:
                    st.error(f"AI failed: {e}")
                    st.code(traceback.format_exc())

        st.divider()
        st.markdown(f"<div class='wow-mini'><b>AI Output</b></div>", unsafe_allow_html=True)
        st.markdown(st.session_state.get("cmp_ai_note", "") or "—")


# ============================================================
# Router
# ============================================================
if page == t(lang, "nav_data"):
    page_data_studio()
elif page == t(lang, "nav_agents"):
    page_agents()
elif page == t(lang, "nav_config"):
    config_studio_page()
elif page == t(lang, "nav_compare"):
    compare_two_datasets_page()
else:
    page_dashboard()
