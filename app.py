import os
import re
import io
import base64
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import streamlit as st
import pandas as pd

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from PyPDF2 import PdfReader, PdfWriter

from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image


# -----------------------------
# i18n
# -----------------------------
STRINGS = {
    "en": {
        "app_title": "FDA 510(k) Review Studio — Regulatory Command Center",
        "settings": "Settings",
        "theme": "Theme",
        "language": "Language",
        "style": "Painter Style",
        "jackpot": "Jackpot",
        "library": "Library",
        "datasets": "Datasets",
        "agents": "Agents",
        "skills": "SKILL.md",
        "workspace": "Workspace",
        "source_material": "Source Material",
        "intelligence_deck": "Intelligence Deck",
        "pdf_viewer": "PDF Viewer",
        "ocr_editor": "OCR Editor",
        "raw_text": "Raw Text",
        "agent_outputs": "Agent Outputs",
        "search_results": "Search Results",
        "final_report": "Final Report",
        "run_agent": "Run Agent",
        "api_keys": "API Keys",
        "managed_by_system": "Managed by System",
        "missing_key": "Missing — enter on this page",
        "global_search": "Global Search (device, K#, UDI, product code...)",
        "note_keeper": "AI Note Keeper",
        "paste_note": "Paste text/markdown",
        "upload_note": "Upload PDF/TXT/MD",
        "ai_magics": "AI Magics",
        "max_tokens": "Max tokens",
        "model": "Model",
        "provider": "Provider",
        "system_prompt": "System prompt",
        "user_prompt": "User prompt",
        "execute_next": "Execute next agent",
        "edit_output_for_next": "Edit output used as input to next agent",
        "mode": "Mode",
        "command_center": "Command Center",
        "status": "Status",
        "ocr": "OCR",
        "trim_extract": "Trim + Extract",
        "trim_pages": "Trim pages",
        "page_ranges": "Page ranges",
        "ocr_engine": "OCR engine",
        "ocr_pages": "OCR pages",
        "vision_ocr": "Vision OCR",
        "local_ocr": "Local OCR (Tesseract)",
        "extract_text": "Extract text (PyPDF2)",
        "render_pdf": "Render PDF preview",
        "download_trimmed": "Download trimmed PDF",
        "danger_zone": "Danger Zone",
        "clear_session": "Clear session state",
        "dashboard": "Interactive Dashboard",
        "device360": "360° Device View",
        "kpi": "KPIs",
        "recall_class": "Recall Class",
        "mdr_count": "MDR Count",
        "loaded": "Loaded",
        "empty": "Empty",
        "agent_input_source": "Agent input source",
        "use_last_output": "Use last edited agent output (preferred)",
        "use_ocr_text": "Use OCR text",
        "use_raw_text": "Use raw extracted text",
        "notes_to_md": "Transform note into organized Markdown",
        "apply": "Apply",
        "run_magic": "Run Magic",
        "render": "Render",
        "markdown_edit": "Markdown Edit",
        "text_edit": "Text Edit",
        "keywords": "Keywords",
        "keyword_colors": "Keyword Colors",
        "download": "Download",
        "upload": "Upload",
        "save": "Save",
        "standardized_loaded": "agents.yaml standardized and loaded.",
        "invalid_agents": "agents.yaml invalid",
        "fab_hint": "FAB is visual; run agents in Agent Outputs tab.",
        "dataset_studio": "Dataset Studio",
        "dataset_type": "Dataset type",
        "paste_dataset": "Paste dataset (CSV/JSON/text)",
        "upload_dataset": "Upload dataset file (CSV/JSON/TXT)",
        "auto_detect": "Auto-detect",
        "parse_load": "Parse & Load",
        "standardize": "Standardize",
        "preview": "Preview (first 20 records)",
        "download_csv": "Download CSV",
        "download_json": "Download JSON",
        "reset_defaults": "Reset to default datasets",
        "dataset_summary": "Dataset Summary (Markdown)",
        "generate_summary": "Generate Summary",
        "dataset_query": "Dataset Query (LLM / keyword search)",
        "keyword_search": "Keyword search",
        "llm_prompt": "LLM prompt",
        "run_query": "Run Query",
        "query_results": "Query results (Markdown)",
        "rows_in_context": "Rows in context",
        "use_filtered": "Use filtered rows in context",
        "filter_results": "Filter results",
        "loaded_rows": "Loaded rows",
        "standardization_report": "Standardization report",
    },
    "zh-TW": {
        "app_title": "FDA 510(k) 審查工作室 — 法規指揮中心",
        "settings": "設定",
        "theme": "主題",
        "language": "語言",
        "style": "畫家風格",
        "jackpot": "隨機開獎",
        "library": "資料庫",
        "datasets": "資料集",
        "agents": "代理 (Agents)",
        "skills": "SKILL.md",
        "workspace": "工作區",
        "source_material": "來源資料",
        "intelligence_deck": "智慧卡組",
        "pdf_viewer": "PDF 檢視",
        "ocr_editor": "OCR 編輯器",
        "raw_text": "原始文字",
        "agent_outputs": "代理輸出",
        "search_results": "搜尋結果",
        "final_report": "最終報告",
        "run_agent": "執行代理",
        "api_keys": "API 金鑰",
        "managed_by_system": "由系統管理",
        "missing_key": "未設定 — 請在網頁輸入",
        "global_search": "全域搜尋（裝置、K號、UDI、產品代碼…）",
        "note_keeper": "AI 筆記整理",
        "paste_note": "貼上文字/Markdown",
        "upload_note": "上傳 PDF/TXT/MD",
        "ai_magics": "AI 魔法",
        "max_tokens": "最大 tokens",
        "model": "模型",
        "provider": "供應商",
        "system_prompt": "系統提示詞",
        "user_prompt": "使用者提示詞",
        "execute_next": "執行下一個代理",
        "edit_output_for_next": "編輯輸出（作為下一代理的輸入）",
        "mode": "模式",
        "command_center": "指揮中心",
        "status": "狀態",
        "ocr": "OCR",
        "trim_extract": "裁切 + 擷取",
        "trim_pages": "裁切頁面",
        "page_ranges": "頁碼範圍",
        "ocr_engine": "OCR 引擎",
        "ocr_pages": "OCR 頁面",
        "vision_ocr": "視覺 OCR（Vision）",
        "local_ocr": "本機 OCR（Tesseract）",
        "extract_text": "文字擷取（PyPDF2）",
        "render_pdf": "顯示 PDF 預覽",
        "download_trimmed": "下載裁切 PDF",
        "danger_zone": "危險區",
        "clear_session": "清除 session 狀態",
        "dashboard": "互動儀表板",
        "device360": "360° 裝置視角",
        "kpi": "關鍵指標",
        "recall_class": "召回等級",
        "mdr_count": "不良事件數",
        "loaded": "已載入",
        "empty": "空",
        "agent_input_source": "代理輸入來源",
        "use_last_output": "使用上次代理的已編修輸出（優先）",
        "use_ocr_text": "使用 OCR 文字",
        "use_raw_text": "使用原始擷取文字",
        "notes_to_md": "將筆記轉為有組織的 Markdown",
        "apply": "套用",
        "run_magic": "執行魔法",
        "render": "呈現",
        "markdown_edit": "Markdown 編輯",
        "text_edit": "文字編輯",
        "keywords": "關鍵詞",
        "keyword_colors": "關鍵詞上色",
        "download": "下載",
        "upload": "上傳",
        "save": "儲存",
        "standardized_loaded": "agents.yaml 已標準化並載入。",
        "invalid_agents": "agents.yaml 無效",
        "fab_hint": "浮動按鈕為視覺效果；請在「代理輸出」分頁執行。",
        "dataset_studio": "資料集工作室",
        "dataset_type": "資料集類型",
        "paste_dataset": "貼上資料集（CSV/JSON/文字）",
        "upload_dataset": "上傳資料集檔案（CSV/JSON/TXT）",
        "auto_detect": "自動偵測",
        "parse_load": "解析並載入",
        "standardize": "標準化",
        "preview": "預覽（前 20 筆）",
        "download_csv": "下載 CSV",
        "download_json": "下載 JSON",
        "reset_defaults": "重置為預設資料集",
        "dataset_summary": "資料集摘要（Markdown）",
        "generate_summary": "產生摘要",
        "dataset_query": "資料集查詢（LLM / 關鍵字搜尋）",
        "keyword_search": "關鍵字搜尋",
        "llm_prompt": "LLM 提示詞",
        "run_query": "執行查詢",
        "query_results": "查詢結果（Markdown）",
        "rows_in_context": "上下文列數",
        "use_filtered": "使用篩選後資料作為上下文",
        "filter_results": "篩選結果",
        "loaded_rows": "已載入筆數",
        "standardization_report": "標準化報告",
    },
}


def t(lang: str, key: str) -> str:
    return STRINGS.get(lang, STRINGS["en"]).get(key, key)


# -----------------------------
# Painter styles (20)
# -----------------------------
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


# -----------------------------
# CSS theme injection (Glassmorphism + Coral)
# -----------------------------
def inject_css(theme: str, painter_accent: str, coral: str = "#FF7F50"):
    if theme == "light":
        bg = "#F6F7FB"
        fg = "#0B1020"
        card = "rgba(10, 16, 32, 0.04)"
        border = "rgba(10, 16, 32, 0.10)"
        shadow = "rgba(10, 16, 32, 0.10)"
    else:
        bg = "#0B1020"
        fg = "#EAF0FF"
        card = "rgba(255,255,255,0.06)"
        border = "rgba(255,255,255,0.10)"
        shadow = "rgba(0,0,0,0.35)"

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
          radial-gradient(1200px 600px at 20% 0%, rgba(255,127,80,0.12), transparent 60%),
          radial-gradient(900px 500px at 80% 10%, rgba(0,200,255,0.10), transparent 55%),
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
        font-weight: 800;
      }}

      .editor-frame {{
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px;
        background: rgba(0,0,0,0.00);
      }}

      div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
      }}

      .fab {{
        position: fixed;
        bottom: 22px;
        right: 26px;
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
        bottom: 70px;
        right: 26px;
        z-index: 9999;
        font-size: 12px;
        padding: 8px 10px;
        border-radius: 12px;
        background: var(--card);
        border: 1px solid var(--border);
        color: var(--fg);
        backdrop-filter: blur(10px);
      }}
      h1, h2, h3, h4 {{ margin-top: 0.2rem; }}
    </style>
    """


# -----------------------------
# Default datasets (embedded)
# -----------------------------
# (Same as your previous version; kept intact)
DEFAULT_510K = [
    {
        "k_number": "K240123",
        "decision_date": "2024-02-14",
        "decision": "SESE",
        "device_name": "FlowPilot FP-2 Infusion Pump",
        "applicant": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_code": "FRN",
        "regulation_number": "880.5725",
        "device_class": "II",
        "panel": "Anesthesiology",
        "review_advisory_committee": "General Hospital",
        "predicate_k_numbers": ["K201111", "K210455"],
        "summary": "Substantial equivalence based on intended use and technological characteristics; added battery monitoring enhancement.",
    },
    {
        "k_number": "K240305",
        "decision_date": "2024-03-27",
        "decision": "SESE",
        "device_name": "StapleWave SW-45 Surgical Stapler",
        "applicant": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_code": "GAG",
        "regulation_number": "878.4750",
        "device_class": "II",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": ["K193210"],
        "summary": "SE determination with design changes in handle ergonomics; staple formation equivalent under bench testing.",
    },
    {
        "k_number": "K240402",
        "decision_date": "2024-04-30",
        "decision": "SESE",
        "device_name": "StapleWave Cartridge SWC-45",
        "applicant": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_code": "GAB",
        "regulation_number": "878.4750",
        "device_class": "II",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": ["K182909"],
        "summary": "SE based on equivalent staple line performance; labeling updated for compatible stapler models.",
    },
    {
        "k_number": "K240588",
        "decision_date": "2024-06-21",
        "decision": "SESE",
        "device_name": "RespiraScan Panel",
        "applicant": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_code": "OUI",
        "regulation_number": "866.3980",
        "device_class": "II",
        "panel": "Microbiology",
        "review_advisory_committee": "Microbiology",
        "predicate_k_numbers": ["K220900"],
        "summary": "SE for multiplex respiratory panel; performance evaluated against comparator methods and reference materials.",
    },
    {
        "k_number": "K230990",
        "decision_date": "2023-12-08",
        "decision": "SESE",
        "device_name": "RespiraScan Analyzer RSA-200",
        "applicant": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_code": "OHT",
        "regulation_number": "862.2570",
        "device_class": "II",
        "panel": "Clinical Chemistry",
        "review_advisory_committee": "Chemistry",
        "predicate_k_numbers": ["K210120"],
        "summary": "SE based on equivalent amplification/detection workflow; software features documented in cybersecurity file.",
    },
    {
        "k_number": "K241010",
        "decision_date": "2024-10-02",
        "decision": "SESE",
        "device_name": "OrchiFill Dermal Filler",
        "applicant": "Orchid Aesthetics Corp.",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "product_code": "LMH",
        "regulation_number": "878.3500",
        "device_class": "III",
        "panel": "General & Plastic Surgery",
        "review_advisory_committee": "Surgery",
        "predicate_k_numbers": ["K221777"],
        "summary": "SE with emphasis on biocompatibility and clinical performance evidence provided by applicant.",
    },
    {
        "k_number": "K240777",
        "decision_date": "2024-08-19",
        "decision": "SESE",
        "device_name": "HarborDrive HD-8 Powered Wheelchair",
        "applicant": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_code": "ITI",
        "regulation_number": "890.3860",
        "device_class": "II",
        "panel": "Physical Medicine",
        "review_advisory_committee": "Rehabilitation",
        "predicate_k_numbers": ["K200333"],
        "summary": "SE based on equivalent mobility performance; updated control firmware and battery configuration.",
    },
]

DEFAULT_ADR = [
    {
        "adverse_event_id": "MDR-2024-000001",
        "report_date": "2024-04-22",
        "event_type": "Malfunction",
        "patient_outcome": "Serious Injury",
        "device_problem": "Misfire / Failure to staple",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave",
        "product_code": "GAG",
        "device_class": "II",
        "udi_di": "00666099000034",
        "recall_number_link": "Z-0421-2024",
        "narrative": "During surgery, stapler misfired; surgeon used alternative device. Patient experienced bleeding requiring additional intervention.",
    },
    {
        "adverse_event_id": "MDR-2024-000002",
        "report_date": "2024-03-18",
        "event_type": "Malfunction",
        "patient_outcome": "Hospitalization",
        "device_problem": "Lead fracture",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure Lead",
        "product_code": "DTB",
        "device_class": "III",
        "udi_di": "00812345000029",
        "recall_number_link": "Z-0333-2024",
        "narrative": "Loss of capture suspected; imaging indicated possible lead integrity issue. Patient hospitalized for revision planning.",
    },
    {
        "adverse_event_id": "MDR-2024-000003",
        "report_date": "2024-09-19",
        "event_type": "Malfunction",
        "patient_outcome": "Death",
        "device_problem": "Premature battery depletion",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure",
        "product_code": "DXY",
        "device_class": "III",
        "udi_di": "00812345000012",
        "recall_number_link": "Z-0777-2024",
        "narrative": "Device reached ERI unexpectedly; therapy interruption suspected. Patient later died; causality not confirmed in report.",
    },
    {
        "adverse_event_id": "MDR-2024-000004",
        "report_date": "2024-02-05",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Unexpected shutdown",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot",
        "product_code": "FRN",
        "device_class": "II",
        "udi_di": "00777001000018",
        "recall_number_link": "Z-0189-2024",
        "narrative": "Pump shut down during routine use; alarm sounded. No reported injury; infusion was restarted after battery replacement.",
    },
    {
        "adverse_event_id": "MDR-2024-000005",
        "report_date": "2024-06-04",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Leakage",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot Set",
        "product_code": "FPA",
        "device_class": "II",
        "udi_di": "00777001000025",
        "recall_number_link": "Z-0510-2024",
        "narrative": "Connector leakage observed; user replaced infusion set. No injury reported.",
    },
    {
        "adverse_event_id": "MDR-2025-000006",
        "report_date": "2025-01-22",
        "event_type": "Injury",
        "patient_outcome": "Serious Injury",
        "device_problem": "False negative test result",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Panel",
        "product_code": "OUI",
        "device_class": "II",
        "udi_di": "00999111000057",
        "recall_number_link": "Z-0103-2025",
        "narrative": "Patient initially tested negative; later confirmed positive. Delay in treatment alleged. Investigation pending.",
    },
    {
        "adverse_event_id": "MDR-2023-000007",
        "report_date": "2023-11-03",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Software error code",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Analyzer",
        "product_code": "OHT",
        "device_class": "II",
        "udi_di": "00999111000064",
        "recall_number_link": "Z-0602-2023",
        "narrative": "Analyzer displayed intermittent error; test run restarted. No patient impact reported.",
    },
    {
        "adverse_event_id": "MDR-2024-000008",
        "report_date": "2024-12-20",
        "event_type": "Injury",
        "patient_outcome": "Other Serious",
        "device_problem": "Allergic reaction",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill Kit",
        "product_code": "FMF",
        "device_class": "II",
        "udi_di": "00444988000101",
        "recall_number_link": "Z-0934-2024",
        "narrative": "Patient reported allergic reaction consistent with latex sensitivity; IFU did not indicate latex presence per reporter.",
    },
    {
        "adverse_event_id": "MDR-2024-000009",
        "report_date": "2024-08-26",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "Battery overheating",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive Battery",
        "product_code": "KJP",
        "device_class": "II",
        "udi_di": "00555123000088",
        "recall_number_link": None,
        "narrative": "Battery pack became warm during charging; user discontinued charging. No injury; device inspected by service center.",
    },
    {
        "adverse_event_id": "MDR-2025-000010",
        "report_date": "2025-01-09",
        "event_type": "Malfunction",
        "patient_outcome": "No Injury",
        "device_problem": "User manual missing troubleshooting section",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive",
        "product_code": "ITI",
        "device_class": "II",
        "udi_di": "00555123000071",
        "recall_number_link": "Z-0144-2025",
        "narrative": "User reported inability to resolve warning indicator due to missing troubleshooting content in manual; no injury.",
    },
]

DEFAULT_GUDID = [
    {
        "primary_di": "00812345000012",
        "udi_di": "00812345000012",
        "device_description": "Implantable cardiac pulse generator",
        "device_class": "III",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure",
        "product_code": "DXY",
        "gmdn_term": "Cardiac pulse generator, implantable",
        "mri_safety": "MR Conditional",
        "sterile": True,
        "single_use": False,
        "implantable": True,
        "contains_nrl": False,
        "version_or_model_number": "PS-3000",
        "catalog_number": "AC-PS3000",
        "record_status": "Published",
        "publish_date": "2024-03-14",
        "company_contact_email": "regulatory@acmemedtech.example",
        "company_contact_phone": "+1-301-555-0101",
        "company_state": "MD",
        "company_country": "US",
    },
    {
        "primary_di": "00812345000029",
        "udi_di": "00812345000029",
        "device_description": "Cardiac lead, pacing, silicone insulated",
        "device_class": "III",
        "manufacturer_name": "Acme MedTech, Inc.",
        "brand_name": "PulseSure Lead",
        "product_code": "DTB",
        "gmdn_term": "Cardiac pacing lead",
        "mri_safety": "MR Conditional",
        "sterile": True,
        "single_use": True,
        "implantable": True,
        "contains_nrl": False,
        "version_or_model_number": "PSL-20",
        "catalog_number": "AC-PSL20",
        "record_status": "Published",
        "publish_date": "2024-05-01",
        "company_contact_email": "regulatory@acmemedtech.example",
        "company_contact_phone": "+1-301-555-0101",
        "company_state": "MD",
        "company_country": "US",
    },
    {
        "primary_di": "00777001000018",
        "udi_di": "00777001000018",
        "device_description": "Infusion pump, programmable ambulatory",
        "device_class": "II",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot",
        "product_code": "FRN",
        "gmdn_term": "Infusion pump, ambulatory",
        "mri_safety": "Not Evaluated",
        "sterile": False,
        "single_use": False,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "FP-2",
        "catalog_number": "NR-FP2",
        "record_status": "Published",
        "publish_date": "2023-11-20",
        "company_contact_email": "qa@northriver.example",
        "company_contact_phone": "+1-617-555-0123",
        "company_state": "MA",
        "company_country": "US",
    },
    {
        "primary_di": "00777001000025",
        "udi_di": "00777001000025",
        "device_description": "Infusion set, sterile, single-use",
        "device_class": "II",
        "manufacturer_name": "NorthRiver Devices LLC",
        "brand_name": "FlowPilot Set",
        "product_code": "FPA",
        "gmdn_term": "Infusion set",
        "mri_safety": "Not Evaluated",
        "sterile": True,
        "single_use": True,
        "implantable": False,
        "contains_nrl": True,
        "version_or_model_number": "FS-1",
        "catalog_number": "NR-FS1",
        "record_status": "Published",
        "publish_date": "2024-01-18",
        "company_contact_email": "qa@northriver.example",
        "company_contact_phone": "+1-617-555-0123",
        "company_state": "MA",
        "company_country": "US",
    },
    {
        "primary_di": "00666099000034",
        "udi_di": "00666099000034",
        "device_description": "Surgical stapler, single-use",
        "device_class": "II",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave",
        "product_code": "GAG",
        "gmdn_term": "Surgical stapler, disposable",
        "mri_safety": "Not Evaluated",
        "sterile": True,
        "single_use": True,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "SW-45",
        "catalog_number": "BW-SW45",
        "record_status": "Published",
        "publish_date": "2024-06-09",
        "company_contact_email": "complaints@bluewavesurg.example",
        "company_contact_phone": "+1-312-555-0190",
        "company_state": "IL",
        "company_country": "US",
    },
    {
        "primary_di": "00666099000041",
        "udi_di": "00666099000041",
        "device_description": "Staple cartridge, sterile, single-use",
        "device_class": "II",
        "manufacturer_name": "BlueWave Surgical Co.",
        "brand_name": "StapleWave Cartridge",
        "product_code": "GAB",
        "gmdn_term": "Staple cartridge",
        "mri_safety": "Not Evaluated",
        "sterile": True,
        "single_use": True,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "SWC-45",
        "catalog_number": "BW-SWC45",
        "record_status": "Published",
        "publish_date": "2024-06-09",
        "company_contact_email": "complaints@bluewavesurg.example",
        "company_contact_phone": "+1-312-555-0190",
        "company_state": "IL",
        "company_country": "US",
    },
    {
        "primary_di": "00999111000057",
        "udi_di": "00999111000057",
        "device_description": "In vitro diagnostic test, respiratory panel",
        "device_class": "II",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Panel",
        "product_code": "OUI",
        "gmdn_term": "Respiratory pathogen assay",
        "mri_safety": "Not Applicable",
        "sterile": False,
        "single_use": True,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "RSP-1",
        "catalog_number": "SD-RSP1",
        "record_status": "Published",
        "publish_date": "2024-02-07",
        "company_contact_email": "support@sunrisedx.example",
        "company_contact_phone": "+1-206-555-0142",
        "company_state": "WA",
        "company_country": "US",
    },
    {
        "primary_di": "00999111000064",
        "udi_di": "00999111000064",
        "device_description": "In vitro diagnostic instrument, PCR analyzer",
        "device_class": "II",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "brand_name": "RespiraScan Analyzer",
        "product_code": "OHT",
        "gmdn_term": "PCR analyzer",
        "mri_safety": "Not Applicable",
        "sterile": False,
        "single_use": False,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "RSA-200",
        "catalog_number": "SD-RSA200",
        "record_status": "Published",
        "publish_date": "2023-09-28",
        "company_contact_email": "support@sunrisedx.example",
        "company_contact_phone": "+1-206-555-0142",
        "company_state": "WA",
        "company_country": "US",
    },
    {
        "primary_di": "00555123000071",
        "udi_di": "00555123000071",
        "device_description": "Wheelchair, powered",
        "device_class": "II",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive",
        "product_code": "ITI",
        "gmdn_term": "Wheelchair, powered",
        "mri_safety": "Not Evaluated",
        "sterile": False,
        "single_use": False,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "HD-8",
        "catalog_number": "HM-HD8",
        "record_status": "Published",
        "publish_date": "2023-06-03",
        "company_contact_email": "service@harbormobility.example",
        "company_contact_phone": "+1-503-555-0166",
        "company_state": "OR",
        "company_country": "US",
    },
    {
        "primary_di": "00555123000088",
        "udi_di": "00555123000088",
        "device_description": "Wheelchair battery pack",
        "device_class": "II",
        "manufacturer_name": "Harbor Mobility Systems",
        "brand_name": "HarborDrive Battery",
        "product_code": "KJP",
        "gmdn_term": "Battery, rechargeable",
        "mri_safety": "Not Evaluated",
        "sterile": False,
        "single_use": False,
        "implantable": False,
        "contains_nrl": False,
        "version_or_model_number": "HB-24",
        "catalog_number": "HM-HB24",
        "record_status": "Published",
        "publish_date": "2024-08-12",
        "company_contact_email": "service@harbormobility.example",
        "company_contact_phone": "+1-503-555-0166",
        "company_state": "OR",
        "company_country": "US",
    },
    {
        "primary_di": "00444988000095",
        "udi_di": "00444988000095",
        "device_description": "Dermal filler, hyaluronic acid",
        "device_class": "III",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill",
        "product_code": "LMH",
        "gmdn_term": "Dermal filler",
        "mri_safety": "Not Applicable",
        "sterile": True,
        "single_use": True,
        "implantable": True,
        "contains_nrl": False,
        "version_or_model_number": "OF-HA2",
        "catalog_number": "OA-OFHA2",
        "record_status": "Published",
        "publish_date": "2024-10-05",
        "company_contact_email": "safety@orchidaesthetics.example",
        "company_contact_phone": "+1-213-555-0188",
        "company_state": "CA",
        "company_country": "US",
    },
    {
        "primary_di": "00444988000101",
        "udi_di": "00444988000101",
        "device_description": "Syringe kit for dermal filler, sterile",
        "device_class": "II",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "brand_name": "OrchiFill Kit",
        "product_code": "FMF",
        "gmdn_term": "Syringe, single-use",
        "mri_safety": "Not Applicable",
        "sterile": True,
        "single_use": True,
        "implantable": False,
        "contains_nrl": True,
        "version_or_model_number": "OK-10",
        "catalog_number": "OA-OK10",
        "record_status": "Published",
        "publish_date": "2024-10-05",
        "company_contact_email": "safety@orchidaesthetics.example",
        "company_contact_phone": "+1-213-555-0188",
        "company_state": "CA",
        "company_country": "US",
    },
]

DEFAULT_RECALL = [
    {
        "recall_number": "Z-0421-2024",
        "recall_class": "I",
        "event_date": "2024-04-16",
        "termination_date": None,
        "status": "Ongoing",
        "firm_name": "BlueWave Surgical Co.",
        "manufacturer_name": "BlueWave Surgical Co.",
        "product_description": "Surgical stapler StapleWave SW-45 may misfire causing incomplete staple formation.",
        "product_code": "GAG",
        "code_info": "Lots 24A01-24A45",
        "reason_for_recall": "Potential for misfire leading to bleeding and prolonged surgery.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 18500,
        "country": "US",
        "state": "IL",
    },
    {
        "recall_number": "Z-0510-2024",
        "recall_class": "II",
        "event_date": "2024-05-28",
        "termination_date": "2024-11-02",
        "status": "Terminated",
        "firm_name": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_description": "Infusion set FlowPilot Set may leak at connector under high backpressure.",
        "product_code": "FPA",
        "code_info": "Lots 23F10-24F07",
        "reason_for_recall": "Leakage may result in under-infusion.",
        "distribution_pattern": "US and Canada",
        "quantity_in_commerce": 42000,
        "country": "US",
        "state": "MA",
    },
    {
        "recall_number": "Z-0777-2024",
        "recall_class": "I",
        "event_date": "2024-09-03",
        "termination_date": None,
        "status": "Ongoing",
        "firm_name": "Acme MedTech, Inc.",
        "manufacturer_name": "Acme MedTech, Inc.",
        "product_description": "PulseSure PS-3000 implantable pulse generator may experience premature battery depletion.",
        "product_code": "DXY",
        "code_info": "Serial range PS3K-2401XXXX to PS3K-2406XXXX",
        "reason_for_recall": "Battery depletion could lead to loss of therapy.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 3200,
        "country": "US",
        "state": "MD",
    },
    {
        "recall_number": "Z-0103-2025",
        "recall_class": "II",
        "event_date": "2025-01-15",
        "termination_date": None,
        "status": "Ongoing",
        "firm_name": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_description": "RespiraScan Panel may yield false negative results under specific reagent storage conditions.",
        "product_code": "OUI",
        "code_info": "Kits expiring 2025-03 to 2025-06",
        "reason_for_recall": "False negatives may delay appropriate treatment.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 12000,
        "country": "US",
        "state": "WA",
    },
    {
        "recall_number": "Z-0218-2024",
        "recall_class": "III",
        "event_date": "2024-02-22",
        "termination_date": "2024-04-12",
        "status": "Terminated",
        "firm_name": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_description": "HarborDrive Battery HB-24 label may list incorrect charging current specification.",
        "product_code": "KJP",
        "code_info": "Label revision L-02",
        "reason_for_recall": "Labeling correction; no reported injuries.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 7800,
        "country": "US",
        "state": "OR",
    },
    {
        "recall_number": "Z-0934-2024",
        "recall_class": "II",
        "event_date": "2024-12-06",
        "termination_date": None,
        "status": "Ongoing",
        "firm_name": "Orchid Aesthetics Corp.",
        "manufacturer_name": "Orchid Aesthetics Corp.",
        "product_description": "OrchiFill Kit syringes may contain natural rubber latex in plunger seal not stated on IFU.",
        "product_code": "FMF",
        "code_info": "Lots OK10-24K01 to OK10-24K22",
        "reason_for_recall": "Potential allergic reactions in latex-sensitive patients.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 9500,
        "country": "US",
        "state": "CA",
    },
    {
        "recall_number": "Z-0602-2023",
        "recall_class": "II",
        "event_date": "2023-10-11",
        "termination_date": "2024-01-08",
        "status": "Terminated",
        "firm_name": "Sunrise Diagnostics Ltd.",
        "manufacturer_name": "Sunrise Diagnostics Ltd.",
        "product_description": "RespiraScan Analyzer RSA-200 may display intermittent error codes due to firmware issue.",
        "product_code": "OHT",
        "code_info": "Firmware v1.0.0 to v1.0.2",
        "reason_for_recall": "Instrument interruptions could delay results reporting.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 640,
        "country": "US",
        "state": "WA",
    },
    {
        "recall_number": "Z-0333-2024",
        "recall_class": "I",
        "event_date": "2024-03-05",
        "termination_date": None,
        "status": "Ongoing",
        "firm_name": "Acme MedTech, Inc.",
        "manufacturer_name": "Acme MedTech, Inc.",
        "product_description": "PulseSure Lead PSL-20 may fracture under repeated flexion.",
        "product_code": "DTB",
        "code_info": "Lots PSL20-23H01 to PSL20-24A12",
        "reason_for_recall": "Lead fracture may result in ineffective pacing.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 8700,
        "country": "US",
        "state": "MD",
    },
    {
        "recall_number": "Z-0189-2024",
        "recall_class": "II",
        "event_date": "2024-01-30",
        "termination_date": "2024-07-19",
        "status": "Terminated",
        "firm_name": "NorthRiver Devices LLC",
        "manufacturer_name": "NorthRiver Devices LLC",
        "product_description": "FlowPilot FP-2 infusion pump battery may discharge faster than expected.",
        "product_code": "FRN",
        "code_info": "Batches FP2-BAT-23Q4",
        "reason_for_recall": "Unexpected shutdown could interrupt infusion.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 5100,
        "country": "US",
        "state": "MA",
    },
    {
        "recall_number": "Z-0144-2025",
        "recall_class": "III",
        "event_date": "2025-01-06",
        "termination_date": "2025-01-28",
        "status": "Terminated",
        "firm_name": "Harbor Mobility Systems",
        "manufacturer_name": "Harbor Mobility Systems",
        "product_description": "HarborDrive HD-8 user manual missing a troubleshooting section in certain print runs.",
        "product_code": "ITI",
        "code_info": "Manual rev 1.1",
        "reason_for_recall": "Documentation correction.",
        "distribution_pattern": "US Nationwide",
        "quantity_in_commerce": 2100,
        "country": "US",
        "state": "OR",
    },
]


# -----------------------------
# Highlighting (Coral)
# -----------------------------
DEFAULT_ONTOLOGY = [
    "predicate",
    "warning",
    "contraindication",
    "sterile",
    "biocompatibility",
    "malfunction",
    "recall",
    "mri",
    "latex",
    "serious injury",
    "death",
    "cybersecurity",
    "software",
    "failure",
    "misfire",
    "udi",
    "510(k)",
    "k-number",
    "k number",
    "intended use",
]


def coral_highlight(text: str, keywords: Optional[List[str]] = None) -> str:
    if not text:
        return ""
    kws = keywords or DEFAULT_ONTOLOGY
    kws = sorted(set([k.strip() for k in kws if k and k.strip()]), key=len, reverse=True)
    out = text

    def repl(m):
        w = m.group(0)
        return f'<span class="coral"><b>{w}</b></span>'

    for k in kws:
        pattern = re.compile(rf"(?i)({re.escape(k)})")
        out = pattern.sub(repl, out)
    return out


# -----------------------------
# Search engine
# -----------------------------
@dataclass
class SearchResult:
    dataset: str
    score: int
    record: Dict[str, Any]


class RegulatorySearchEngine:
    def __init__(self, df_510k: pd.DataFrame, df_adr: pd.DataFrame, df_gudid: pd.DataFrame, df_recall: pd.DataFrame):
        self.df_510k = df_510k
        self.df_adr = df_adr
        self.df_gudid = df_gudid
        self.df_recall = df_recall

    def _fuzzy_hits(self, df: pd.DataFrame, cols: List[str], query: str, min_score=75, limit=25):
        q = (query or "").strip().lower()
        hits = []
        if not q or df is None or df.empty:
            return hits
        for _, row in df.iterrows():
            best = 0
            for c in cols:
                val = str(row.get(c, "")).lower()
                if not val:
                    continue
                best = max(best, fuzz.partial_ratio(q, val))
            if best >= min_score:
                hits.append((best, row.to_dict()))
        hits.sort(key=lambda x: x[0], reverse=True)
        return hits[:limit]

    def search_all(self, query: str) -> Dict[str, List[SearchResult]]:
        results: Dict[str, List[SearchResult]] = {"510k": [], "adr": [], "gudid": [], "recall": []}
        if not (query or "").strip():
            return results

        for score, rec in self._fuzzy_hits(
            self.df_510k,
            ["k_number", "device_name", "applicant", "manufacturer_name", "product_code", "summary"],
            query,
        ):
            results["510k"].append(SearchResult("510k", score, rec))

        for score, rec in self._fuzzy_hits(
            self.df_adr,
            ["adverse_event_id", "brand_name", "manufacturer_name", "product_code", "udi_di", "device_problem", "narrative"],
            query,
        ):
            results["adr"].append(SearchResult("adr", score, rec))

        for score, rec in self._fuzzy_hits(
            self.df_gudid,
            ["udi_di", "primary_di", "brand_name", "manufacturer_name", "product_code", "device_description", "gmdn_term"],
            query,
        ):
            results["gudid"].append(SearchResult("gudid", score, rec))

        for score, rec in self._fuzzy_hits(
            self.df_recall,
            ["recall_number", "firm_name", "manufacturer_name", "product_code", "reason_for_recall", "product_description"],
            query,
        ):
            results["recall"].append(SearchResult("recall", score, rec))

        q_upper = (query or "").strip().upper()
        top_k = None
        for r in results["510k"]:
            if str(r.record.get("k_number", "")).upper() == q_upper:
                top_k = r.record
                break
        if top_k:
            preds = top_k.get("predicate_k_numbers", []) or []
            for k in preds:
                for score, rec in self._fuzzy_hits(self.df_510k, ["k_number", "device_name"], k, min_score=88, limit=10):
                    results["510k"].append(SearchResult("510k", score, rec))

        return results

    def device_360_view(self, query: str) -> Dict[str, Any]:
        r = self.search_all(query)
        top_510k = r["510k"][0].record if r["510k"] else None
        product_code = (top_510k or {}).get("product_code")
        device_name = (top_510k or {}).get("device_name")

        recalls, mdrs, gudid = [], [], []
        if product_code:
            recalls = [x.record for x in r["recall"] if x.record.get("product_code") == product_code]
            mdrs = [x.record for x in r["adr"] if x.record.get("product_code") == product_code]
            gudid = [x.record for x in r["gudid"] if x.record.get("product_code") == product_code]
        if (not product_code) and device_name:
            recalls = [x.record for x in r["recall"]][:8]
            mdrs = [x.record for x in r["adr"]][:8]
            gudid = [x.record for x in r["gudid"]][:8]

        top_recall_class = None
        if recalls:
            priority = {"I": 3, "II": 2, "III": 1}
            recalls_sorted = sorted(recalls, key=lambda rr: priority.get(str(rr.get("recall_class", "")).upper(), 0), reverse=True)
            top_recall_class = recalls_sorted[0].get("recall_class")

        return {
            "top_510k": top_510k,
            "recalls": recalls[:8],
            "mdr_count": len(mdrs),
            "mdr_examples": mdrs[:6],
            "gudid_examples": gudid[:6],
            "top_recall_class": top_recall_class,
        }


# -----------------------------
# agents.yaml config manager (Pydantic)
# -----------------------------
class AgentDef(BaseModel):
    id: str
    name: str
    description: str = ""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 4000
    system_prompt: str
    user_prompt: str = "Analyze the provided content."


class AgentsConfig(BaseModel):
    version: str = "1.0"
    agents: List[AgentDef] = Field(default_factory=list)


def load_and_standardize_agents_yaml(raw_text: str) -> AgentsConfig:
    data = yaml.safe_load(raw_text) or {}
    if isinstance(data, list):
        data = {"version": "1.0", "agents": data}
    data.setdefault("version", "1.0")
    data.setdefault("agents", [])

    fixed = []
    for a in data["agents"]:
        if not isinstance(a, dict):
            continue
        a.setdefault("provider", "openai")
        a.setdefault("model", "gpt-4o-mini")
        a.setdefault("temperature", 0.2)
        a.setdefault("max_tokens", 4000)
        a.setdefault("description", "")
        a.setdefault("user_prompt", "Analyze the provided content.")
        fixed.append(a)
    data["agents"] = fixed
    return AgentsConfig.model_validate(data)


def dump_agents_yaml(cfg: AgentsConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True)


# -----------------------------
# LLM providers
# -----------------------------
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini"]
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-3-pro-preview"]
ANTHROPIC_MODELS = ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
XAI_MODELS = ["grok-4-fast-reasoning", "grok-4-1-fast-non-reasoning"]


def provider_model_map():
    return {"openai": OPENAI_MODELS, "gemini": GEMINI_MODELS, "anthropic": ANTHROPIC_MODELS, "xai": XAI_MODELS}


def env_or_session(env_name: str) -> Optional[str]:
    if os.environ.get(env_name):
        return os.environ.get(env_name)
    return st.session_state.get("api_keys", {}).get(env_name)


def call_llm_text(
    provider: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
) -> str:
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
        m = genai.GenerativeModel(model_name=model, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
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


def call_vision_ocr(provider: str, model: str, api_key: str, images: List[Image.Image], lang: str, max_tokens: int = 12000) -> str:
    provider = (provider or "").lower().strip()
    sys = "You are an OCR engine for regulatory PDFs. Preserve tables when possible. Output plain text (no markdown)."
    if lang == "zh-TW":
        sys = "你是法規 PDF 的 OCR 引擎。盡可能保留表格結構與數值。輸出純文字（不要 Markdown）。"

    prompt = "Extract all readable text from this page. Preserve tables and headings."
    if lang == "zh-TW":
        prompt = "請從此頁影像擷取所有可讀文字，盡可能保留表格與標題結構。"

    chunks = []
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        for i, img in enumerate(images, start=1):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"

            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": data_url}]},
                ],
                max_output_tokens=max_tokens,
                temperature=0.0,
            )
            chunks.append(f"\n\n--- PAGE {i} ---\n{(resp.output_text or '').strip()}")
        return "\n".join(chunks).strip()

    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model_name=model, generation_config={"temperature": 0.0, "max_output_tokens": max_tokens})
        for i, img in enumerate(images, start=1):
            r = m.generate_content([sys + "\n" + prompt, img])
            chunks.append(f"\n\n--- PAGE {i} ---\n{(r.text or '').strip()}")
        return "\n".join(chunks).strip()

    raise ValueError("Vision OCR only supported for provider=openai or gemini in this build.")


# -----------------------------
# PDF tools
# -----------------------------
def parse_page_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    ranges_str = (ranges_str or "").strip()
    if not ranges_str:
        return []
    out = []
    parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
            if a > b:
                a, b = b, a
            out.append((a, b))
        else:
            n = int(p)
            out.append((n, n))
    return out


def trim_pdf_bytes(pdf_bytes: bytes, page_ranges: List[Tuple[int, int]]) -> bytes:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    n = len(reader.pages)

    for (s, e) in page_ranges:
        s = max(1, s)
        e = min(n, e)
        for i in range(s - 1, e):
            writer.add_page(reader.pages[i])

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def extract_text_pypdf2(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join([(p.extract_text() or "") for p in reader.pages]).strip()


def render_pdf_iframe(pdf_bytes: bytes, height: int = 520) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"""
    <iframe
        src="data:application/pdf;base64,{b64}"
        width="100%"
        height="{height}"
        style="border: 1px solid var(--border); border-radius: 14px; background: white;"
        type="application/pdf">
    </iframe>
    """


# -----------------------------
# Dataset Studio: parsing + standardization
# -----------------------------
CANON_510K = [
    "k_number",
    "decision_date",
    "decision",
    "device_name",
    "applicant",
    "manufacturer_name",
    "product_code",
    "regulation_number",
    "device_class",
    "panel",
    "review_advisory_committee",
    "predicate_k_numbers",
    "summary",
]
CANON_ADR = [
    "adverse_event_id",
    "report_date",
    "event_type",
    "patient_outcome",
    "device_problem",
    "manufacturer_name",
    "brand_name",
    "product_code",
    "device_class",
    "udi_di",
    "recall_number_link",
    "narrative",
]
CANON_GUDID = [
    "primary_di",
    "udi_di",
    "device_description",
    "device_class",
    "manufacturer_name",
    "brand_name",
    "product_code",
    "gmdn_term",
    "mri_safety",
    "sterile",
    "single_use",
    "implantable",
    "contains_nrl",
    "version_or_model_number",
    "catalog_number",
    "record_status",
    "publish_date",
    "company_contact_email",
    "company_contact_phone",
    "company_state",
    "company_country",
]
CANON_RECALL = [
    "recall_number",
    "recall_class",
    "event_date",
    "termination_date",
    "status",
    "firm_name",
    "manufacturer_name",
    "product_description",
    "product_code",
    "code_info",
    "reason_for_recall",
    "distribution_pattern",
    "quantity_in_commerce",
    "country",
    "state",
]

SYNONYMS = {
    "510k": {
        "k_number": ["k_number", "knumber", "k", "k_no", "k#", "knum", "submission_number"],
        "decision_date": ["decision_date", "date", "clearance_date"],
        "decision": ["decision", "decision_code", "se_decision", "determination"],
        "device_name": ["device_name", "device", "name", "device title", "device_title"],
        "applicant": ["applicant", "submitter", "company", "applicant_name"],
        "manufacturer_name": ["manufacturer_name", "manufacturer", "mfr", "firm_name"],
        "product_code": ["product_code", "procode", "productcode", "code"],
        "regulation_number": ["regulation_number", "regulation", "21cfr", "cfr"],
        "device_class": ["device_class", "class"],
        "panel": ["panel", "medical_specialty", "review_panel"],
        "review_advisory_committee": ["review_advisory_committee", "committee"],
        "predicate_k_numbers": ["predicate_k_numbers", "predicates", "predicate", "predicate_k", "predicate_knumbers"],
        "summary": ["summary", "clearance_summary", "decision_summary", "description"],
    },
    "adr": {
        "adverse_event_id": ["adverse_event_id", "mdr_id", "event_id", "report_id"],
        "report_date": ["report_date", "date"],
        "event_type": ["event_type", "type"],
        "patient_outcome": ["patient_outcome", "outcome"],
        "device_problem": ["device_problem", "problem", "issue"],
        "manufacturer_name": ["manufacturer_name", "manufacturer", "mfr", "firm_name"],
        "brand_name": ["brand_name", "brand", "device_brand"],
        "product_code": ["product_code", "code"],
        "device_class": ["device_class", "class"],
        "udi_di": ["udi_di", "udi", "primary_di"],
        "recall_number_link": ["recall_number_link", "recall_number", "linked_recall"],
        "narrative": ["narrative", "description", "event_description", "text"],
    },
    "gudid": {
        "primary_di": ["primary_di", "primarydi", "udi_di", "udi"],
        "udi_di": ["udi_di", "primary_di", "udi", "di"],
        "device_description": ["device_description", "description", "device_desc"],
        "device_class": ["device_class", "class"],
        "manufacturer_name": ["manufacturer_name", "manufacturer", "mfr", "company"],
        "brand_name": ["brand_name", "brand", "device_brand"],
        "product_code": ["product_code", "code"],
        "gmdn_term": ["gmdn_term", "gmdn", "gmdn_name"],
        "mri_safety": ["mri_safety", "mri", "mri_status"],
        "sterile": ["sterile", "is_sterile"],
        "single_use": ["single_use", "singleuse", "is_single_use"],
        "implantable": ["implantable", "is_implantable"],
        "contains_nrl": ["contains_nrl", "nrl", "latex", "contains_latex"],
        "version_or_model_number": ["version_or_model_number", "model", "model_number", "version"],
        "catalog_number": ["catalog_number", "catalog", "cat_no"],
        "record_status": ["record_status", "status"],
        "publish_date": ["publish_date", "date"],
        "company_contact_email": ["company_contact_email", "email"],
        "company_contact_phone": ["company_contact_phone", "phone", "telephone"],
        "company_state": ["company_state", "state"],
        "company_country": ["company_country", "country"],
    },
    "recall": {
        "recall_number": ["recall_number", "recall", "recall_id", "z_number", "z#"],
        "recall_class": ["recall_class", "class"],
        "event_date": ["event_date", "date", "initiation_date"],
        "termination_date": ["termination_date", "term_date"],
        "status": ["status", "recall_status"],
        "firm_name": ["firm_name", "firm", "company", "applicant"],
        "manufacturer_name": ["manufacturer_name", "manufacturer", "mfr", "firm_name"],
        "product_description": ["product_description", "description", "product"],
        "product_code": ["product_code", "code"],
        "code_info": ["code_info", "lots", "serials", "batch"],
        "reason_for_recall": ["reason_for_recall", "reason", "root_cause", "cause"],
        "distribution_pattern": ["distribution_pattern", "distribution"],
        "quantity_in_commerce": ["quantity_in_commerce", "quantity", "qty"],
        "country": ["country"],
        "state": ["state"],
    },
}


def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def detect_format(text: str) -> str:
    t0 = (text or "").lstrip()
    if not t0:
        return "unknown"
    if t0.startswith("[") or t0.startswith("{"):
        return "json"
    if "," in t0 and "\n" in t0:
        return "csv"
    return "text"


def parse_dataset_blob(blob: Union[str, bytes], filename: Optional[str] = None) -> pd.DataFrame:
    if isinstance(blob, bytes):
        text = blob.decode("utf-8", errors="ignore")
    else:
        text = blob

    if filename:
        fn = filename.lower()
        if fn.endswith(".json"):
            fmt = "json"
        elif fn.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = detect_format(text)
    else:
        fmt = detect_format(text)

    if fmt == "json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ["data", "records", "items", "rows"]:
                if k in obj and isinstance(obj[k], list):
                    obj = obj[k]
                    break
            if isinstance(obj, dict):
                obj = [obj]
        if not isinstance(obj, list):
            raise ValueError("JSON must be a list of objects or an object/records wrapper.")
        return pd.DataFrame(obj)

    if fmt == "csv":
        return pd.read_csv(io.StringIO(text))

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = [obj]
        return pd.DataFrame(obj)
    except Exception:
        return pd.read_csv(io.StringIO(text))


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


def standardize_df(dataset_type: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    dataset_type = dataset_type.lower()
    if df is None or df.empty:
        return pd.DataFrame(), "No data to standardize."

    if dataset_type == "510k":
        canon, syn = CANON_510K, SYNONYMS["510k"]
    elif dataset_type == "adr":
        canon, syn = CANON_ADR, SYNONYMS["adr"]
    elif dataset_type == "gudid":
        canon, syn = CANON_GUDID, SYNONYMS["gudid"]
    elif dataset_type == "recall":
        canon, syn = CANON_RECALL, SYNONYMS["recall"]
    else:
        raise ValueError("Unknown dataset type")

    original_cols = list(df.columns)
    mapped = {}
    report_lines = ["### Standardization Mapping", "", "| Canonical field | Source column |", "|---|---|"]

    for cfield in canon:
        src = _best_match_column(original_cols, syn.get(cfield, [cfield]))
        mapped[cfield] = src
        report_lines.append(f"| `{cfield}` | `{src if src else '— (missing)'}` |")

    out = pd.DataFrame()
    for cfield in canon:
        src = mapped[cfield]
        out[cfield] = df[src] if (src and src in df.columns) else None

    if dataset_type == "510k":
        def to_list(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                parts = [p.strip() for p in re.split(r"[;,]+", x) if p.strip()]
                return parts
            return [str(x)]
        out["predicate_k_numbers"] = out["predicate_k_numbers"].apply(to_list)

    if dataset_type == "gudid":
        def to_bool(v):
            if isinstance(v, bool):
                return v
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            s = str(v).strip().lower()
            if s in ["true", "t", "yes", "y", "1"]:
                return True
            if s in ["false", "f", "no", "n", "0"]:
                return False
            return None
        for bcol in ["sterile", "single_use", "implantable", "contains_nrl"]:
            out[bcol] = out[bcol].apply(to_bool)

    if dataset_type == "recall":
        def to_int(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            try:
                return int(float(str(v).replace(",", "").strip()))
            except Exception:
                return None
        out["quantity_in_commerce"] = out["quantity_in_commerce"].apply(to_int)

    def row_has_signal(r):
        for c in canon:
            v = r.get(c)
            if isinstance(v, list) and len(v) > 0:
                return True
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if str(v).strip() != "":
                return True
        return False

    out = out[out.apply(row_has_signal, axis=1)].reset_index(drop=True)
    report_lines += ["", f"**Rows:** {len(out)}", f"**Original columns:** {len(original_cols)}"]
    return out, "\n".join(report_lines)


def df_to_json_records(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2)


def keyword_filter_df(df: pd.DataFrame, keyword: str, limit: int = 50) -> pd.DataFrame:
    if df is None or df.empty or not (keyword or "").strip():
        return pd.DataFrame()
    kw = keyword.strip().lower()

    def hit_row(row) -> bool:
        for v in row.values:
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if isinstance(v, list):
                if any(kw in str(x).lower() for x in v):
                    return True
            else:
                if kw in str(v).lower():
                    return True
        return False

    return df[df.apply(hit_row, axis=1)].head(limit).copy()


def dataset_context_markdown(dataset_type: str, df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return f"Dataset `{dataset_type}` is empty."

    n = len(df)
    cols = list(df.columns)

    missing = []
    for c in cols:
        miss = df[c].isna().mean() if c in df.columns else 1.0
        missing.append((c, miss))
    missing_sorted = sorted(missing, key=lambda x: x[1], reverse=True)

    key_cols_map = {
        "510k": ["k_number", "device_name", "applicant", "product_code", "decision"],
        "adr": ["adverse_event_id", "brand_name", "product_code", "patient_outcome", "device_problem", "recall_number_link"],
        "gudid": ["udi_di", "brand_name", "product_code", "sterile", "single_use", "implantable", "contains_nrl", "mri_safety"],
        "recall": ["recall_number", "product_code", "recall_class", "status", "firm_name", "reason_for_recall"],
    }
    keys = [c for c in key_cols_map.get(dataset_type, []) if c in df.columns]

    top_tables = []
    for c in keys:
        try:
            vc = df[c].astype(str).value_counts().head(10)
            tbl = "| value | count |\n|---|---|\n" + "\n".join([f"| `{v}` | {int(k)} |" for v, k in vc.items()])
            top_tables.append(f"### Top values: `{c}`\n{tbl}\n")
        except Exception:
            continue

    miss_tbl = "| column | missing_rate |\n|---|---|\n" + "\n".join([f"| `{c}` | {m:.2%} |" for c, m in missing_sorted[: min(20, len(missing_sorted))]])
    sample_records = df.head(max_rows).to_dict(orient="records")
    sample_json = json.dumps(sample_records, ensure_ascii=False, indent=2)

    return "\n".join(
        [
            f"## Dataset Context: `{dataset_type}`",
            f"- Rows: **{n}**",
            f"- Columns: **{len(cols)}**",
            "",
            "### Columns",
            ", ".join([f"`{c}`" for c in cols]),
            "",
            "### Missingness (top 20 by missing rate)",
            miss_tbl,
            "",
            *top_tables,
            f"### Sample records (first {min(max_rows, n)} rows as JSON)",
            "```json",
            sample_json,
            "```",
        ]
    )


# -----------------------------
# AI Note Keeper
# -----------------------------
MAGICS = [
    "Organize Note (Markdown)",
    "Executive Summary",
    "Action Items + Owners",
    "Risk/Deficiency Finder",
    "Compliance Checklist Generator",
    "AI Keywords Highlighter",
]


def magic_run(magic_name: str, provider: str, model: str, api_key: str, raw_note: str, lang: str, max_tokens: int = 6000) -> str:
    if lang == "zh-TW":
        system = "你是資深法規與技術編輯助理。請回傳乾淨、結構化的 Markdown。內容需保守、不可捏造，缺資料請用 Gap 標示。"
    else:
        system = "You are an expert regulatory assistant and technical editor. Return clean, structured Markdown. Be conservative; do not fabricate. Mark missing info as Gap."

    if magic_name == "Organize Note (Markdown)":
        user = f"請把以下筆記整理成結構化 Markdown（含標題、重點、待辦、風險/缺口、關鍵詞）：\n\n{raw_note}" if lang == "zh-TW" else f"Organize the following note into structured Markdown with headings, bullets, action items, gaps, and keywords:\n\n{raw_note}"
        return call_llm_text(provider, model, api_key, system, user, max_tokens=max_tokens, temperature=0.2)

    if magic_name == "Executive Summary":
        user = f"請產出一段高密度的主管摘要（Markdown，3~7 點重點）：\n\n{raw_note}" if lang == "zh-TW" else f"Create an executive summary (Markdown) with 3-7 key points:\n\n{raw_note}"
        return call_llm_text(provider, model, api_key, system, user, max_tokens=max_tokens, temperature=0.2)

    if magic_name == "Action Items + Owners":
        user = f"請從筆記抽取可執行待辦事項，輸出 Markdown 表格：Action、Owner(建議)、Due date(建議)、Rationale。\n\n{raw_note}" if lang == "zh-TW" else f"Extract action items. Output a Markdown table: Action, Owner (suggested), Due date (suggested), Rationale.\n\n{raw_note}"
        return call_llm_text(provider, model, api_key, system, user, max_tokens=max_tokens, temperature=0.2)

    if magic_name == "Risk/Deficiency Finder":
        user = f"請找出法規風險/缺失點，並以 [High/Med/Low] 分級；每點需包含證據摘錄（引用原文）。\n\n{raw_note}" if lang == "zh-TW" else f"Identify regulatory risks/deficiencies with [High/Med/Low] severity and evidence quotes.\n\n{raw_note}"
        return call_llm_text(provider, model, api_key, system, user, max_tokens=max_tokens, temperature=0.2)

    if magic_name == "Compliance Checklist Generator":
        user = f"請依常見 510(k) 審查主題產出合規核對清單（Markdown checkbox），如：biocompatibility、sterility、labeling、cybersecurity、software V&V。\n\n{raw_note}" if lang == "zh-TW" else f"Generate a compliance checklist (Markdown checkboxes) for common 510(k) topics.\n\n{raw_note}"
        return call_llm_text(provider, model, api_key, system, user, max_tokens=max_tokens, temperature=0.2)

    raise ValueError("AI Keywords Highlighter handled in UI.")


def apply_keyword_colors(html_or_text: str, keyword_color_pairs: List[Tuple[str, str]]) -> str:
    out = coral_highlight(html_or_text)
    for kw, color in keyword_color_pairs:
        if not kw.strip():
            continue
        out = out.replace(kw, f'<span style="color:{color}; font-weight:900; text-shadow:0 0 18px rgba(255,255,255,0.08)">{kw}</span>')
    return out


# -----------------------------
# Streamlit app state
# -----------------------------
st.set_page_config(page_title="510(k) Review Studio", layout="wide")


def ss_init():
    st.session_state.setdefault("theme", "dark")
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("style", PAINTER_STYLES[0])

    st.session_state.setdefault("api_keys", {})
    st.session_state.setdefault("global_query", "")

    st.session_state.setdefault("pdf_bytes", None)
    st.session_state.setdefault("trimmed_pdf_bytes", None)

    st.session_state.setdefault("raw_text", "")
    st.session_state.setdefault("ocr_text", "")

    st.session_state.setdefault("agents_yaml_text", "")
    st.session_state.setdefault("skill_md", "")
    st.session_state.setdefault("agent_outputs", [])
    st.session_state.setdefault("final_report", "")

    st.session_state.setdefault("note_raw", "")
    st.session_state.setdefault("note_md", "")
    st.session_state.setdefault("note_render_html", "")
    st.session_state.setdefault("keyword_pairs", [("", "#FF7F50"), ("", "#00B3B3"), ("", "#F4D03F")])

    st.session_state.setdefault("df_510k", pd.DataFrame(DEFAULT_510K))
    st.session_state.setdefault("df_adr", pd.DataFrame(DEFAULT_ADR))
    st.session_state.setdefault("df_gudid", pd.DataFrame(DEFAULT_GUDID))
    st.session_state.setdefault("df_recall", pd.DataFrame(DEFAULT_RECALL))

    st.session_state.setdefault("ds_input_text", "")
    st.session_state.setdefault("ds_std_report", "")
    st.session_state.setdefault("ds_summary_md", "")
    st.session_state.setdefault("ds_summary_prompt", "")
    st.session_state.setdefault("ds_query_prompt", "")
    st.session_state.setdefault("ds_query_md", "")
    st.session_state.setdefault("ds_keyword", "")
    st.session_state.setdefault("ds_filtered_df", pd.DataFrame())


ss_init()


def load_text_file(path: str, default: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return default


if not st.session_state["agents_yaml_text"]:
    st.session_state["agents_yaml_text"] = load_text_file("agents.yaml", "version: '1.0'\nagents: []\n")
if not st.session_state["skill_md"]:
    st.session_state["skill_md"] = load_text_file("SKILL.md", "# SKILL\n\n")


lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]
st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)


def build_engine_from_session() -> RegulatorySearchEngine:
    return RegulatorySearchEngine(
        st.session_state["df_510k"],
        st.session_state["df_adr"],
        st.session_state["df_gudid"],
        st.session_state["df_recall"],
    )


engine = build_engine_from_session()


# -----------------------------
# Top bar (WOW)
# -----------------------------
def key_status_chip(label: str, env_name: str) -> str:
    if os.environ.get(env_name):
        dot, status = "var(--ok)", t(lang, "managed_by_system")
    elif st.session_state["api_keys"].get(env_name):
        dot, status = "var(--warn)", "Session"
    else:
        dot, status = "var(--bad)", t(lang, "missing_key")
    return f"<span class='chip'><span class='dot' style='background:{dot}'></span>{label}: {status}</span>"


top = st.container()
with top:
    c1, c2, c3 = st.columns([2.2, 3.0, 1.2], vertical_alignment="center")

    with c1:
        st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'app_title')}</h3></div>", unsafe_allow_html=True)

    with c2:
        df_510k = st.session_state["df_510k"]
        df_adr = st.session_state["df_adr"]
        df_gudid = st.session_state["df_gudid"]
        df_recall = st.session_state["df_recall"]

        chips = ""
        chips += key_status_chip("OpenAI", "OPENAI_API_KEY")
        chips += key_status_chip("Gemini", "GEMINI_API_KEY")
        chips += key_status_chip("Anthropic", "ANTHROPIC_API_KEY")
        chips += key_status_chip("xAI", "XAI_API_KEY")

        dataset_chip = f"<span class='chip'><span class='dot'></span>510k:{len(df_510k)} ADR:{len(df_adr)} GUDID:{len(df_gudid)} Recall:{len(df_recall)}</span>"
        ocr_chip = (
            f"<span class='chip'><span class='dot' style='background:var(--ok)'></span>{t(lang,'ocr')}: {t(lang,'loaded')}</span>"
            if st.session_state["ocr_text"].strip()
            else f"<span class='chip'><span class='dot' style='background:var(--warn)'></span>{t(lang,'ocr')}: {t(lang,'empty')}</span>"
        )

        st.markdown(f"<div class='wow-card'>{chips}{dataset_chip}{ocr_chip}</div>", unsafe_allow_html=True)
        mana = (len(st.session_state["agent_outputs"]) % 10) / 10.0
        st.progress(mana, text="Review Mana (agent runs)")

    with c3:
        with st.popover(t(lang, "settings")):
            st.session_state["theme"] = st.radio(t(lang, "theme"), ["dark", "light"], index=0 if theme == "dark" else 1)
            st.session_state["lang"] = st.radio(t(lang, "language"), ["en", "zh-TW"], index=0 if lang == "en" else 1)

            style_names = [s["name"] for s in PAINTER_STYLES]
            curr = st.session_state["style"]["name"]
            ix = style_names.index(curr) if curr in style_names else 0
            pick = st.selectbox(t(lang, "style"), style_names, index=ix)
            st.session_state["style"] = next(s for s in PAINTER_STYLES if s["name"] == pick)

            if st.button(t(lang, "jackpot"), use_container_width=True):
                st.session_state["style"] = jackpot_style()
                st.rerun()

lang = st.session_state["lang"]
theme = st.session_state["theme"]
style = st.session_state["style"]
st.markdown(inject_css(theme, style["accent"]), unsafe_allow_html=True)

st.markdown(f"<div class='fab'>RUN</div><div class='fab-sub'>{t(lang,'fab_hint')}</div>", unsafe_allow_html=True)


# -----------------------------
# Sidebar (Library + API keys + config)
# -----------------------------
with st.sidebar:
    st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'library')}</h4></div>", unsafe_allow_html=True)
    st.caption(t(lang, "api_keys"))

    key_specs = [("OpenAI", "OPENAI_API_KEY"), ("Gemini", "GEMINI_API_KEY"), ("Anthropic", "ANTHROPIC_API_KEY"), ("xAI", "XAI_API_KEY")]
    for label, env_name in key_specs:
        if os.environ.get(env_name):
            st.markdown(f"<div class='wow-mini'><b>{label}</b><br/><span class='chip'>{t(lang,'managed_by_system')}</span></div>", unsafe_allow_html=True)
        else:
            v = st.text_input(f"{label} key", type="password", value=st.session_state["api_keys"].get(env_name, ""))
            if v:
                st.session_state["api_keys"][env_name] = v

    st.divider()

    st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'agents')}</h4></div>", unsafe_allow_html=True)
    up = st.file_uploader(f"{t(lang,'upload')} agents.yaml", type=["yaml", "yml"])
    if up:
        st.session_state["agents_yaml_text"] = up.read().decode("utf-8", errors="ignore")
    st.session_state["agents_yaml_text"] = st.text_area("agents.yaml", st.session_state["agents_yaml_text"], height=220)

    agents_cfg = None
    try:
        agents_cfg = load_and_standardize_agents_yaml(st.session_state["agents_yaml_text"])
        st.success(t(lang, "standardized_loaded"))
        st.download_button(f"{t(lang,'download')} agents.yaml", dump_agents_yaml(agents_cfg), file_name="agents.yaml")
    except Exception as e:
        st.error(f"{t(lang,'invalid_agents')}: {e}")

    st.divider()
    st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'skills')}</h4></div>", unsafe_allow_html=True)
    st.session_state["skill_md"] = st.text_area("SKILL.md", st.session_state["skill_md"], height=220)
    st.download_button(f"{t(lang,'download')} SKILL.md", st.session_state["skill_md"], file_name="SKILL.md")

    st.divider()
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'danger_zone')}</b></div>", unsafe_allow_html=True)
    if st.button(t(lang, "clear_session"), use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# -----------------------------
# Global search + mode
# -----------------------------
qcol1, qcol2 = st.columns([2.4, 1.2], vertical_alignment="center")
with qcol1:
    st.session_state["global_query"] = st.text_input(t(lang, "global_search"), value=st.session_state["global_query"])
with qcol2:
    view_mode = st.selectbox(t(lang, "mode"), [t(lang, "command_center"), t(lang, "note_keeper")], index=0)

query = st.session_state["global_query"].strip()
d360 = engine.device_360_view(query) if query else None

st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'dashboard')}</h4></div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    top_k = (d360 or {}).get("top_510k") or {}
    st.markdown(f"<div class='wow-mini'><b>K#</b><br/>{top_k.get('k_number','—')}</div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='wow-mini'><b>Decision</b><br/>{top_k.get('decision','—')}</div>", unsafe_allow_html=True)
with k3:
    mdr_count = (d360 or {}).get("mdr_count", 0) if d360 else 0
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'mdr_count')}</b><br/>{mdr_count}</div>", unsafe_allow_html=True)
with k4:
    rc = (d360 or {}).get("top_recall_class") if d360 else None
    rc_html = f"<span class='coral'>{rc}</span>" if rc else "—"
    st.markdown(f"<div class='wow-mini'><b>{t(lang,'recall_class')}</b><br/>{rc_html}</div>", unsafe_allow_html=True)


# -----------------------------
# Mode: AI Note Keeper
# -----------------------------
if view_mode == t(lang, "note_keeper"):
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'note_keeper')}</h3></div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown(f"<div class='wow-mini'><b>{t(lang,'paste_note')}</b></div>", unsafe_allow_html=True)
        st.session_state["note_raw"] = st.text_area("", st.session_state["note_raw"], height=220)

        upl = st.file_uploader(t(lang, "upload_note"), type=["pdf", "txt", "md"])
        if upl:
            b = upl.read()
            name = upl.name.lower()
            st.session_state["note_raw"] = extract_text_pypdf2(b) if name.endswith(".pdf") else b.decode("utf-8", errors="ignore")

        st.markdown(f"<div class='wow-mini'><b>{t(lang,'ai_magics')}</b></div>", unsafe_allow_html=True)
        pmap = provider_model_map()
        provider = st.selectbox(t(lang, "provider"), list(pmap.keys()), index=0, key="note_provider")
        model = st.selectbox(t(lang, "model"), pmap[provider], index=0, key="note_model")
        max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=6000, step=256, key="note_max_tokens")

        magic = st.selectbox("Magic", MAGICS, index=0, key="note_magic")

        if magic != "AI Keywords Highlighter":
            if st.button(t(lang, "run_magic"), use_container_width=True, key="note_run_magic"):
                env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[provider]
                api_key = env_or_session(env_name)
                if not api_key:
                    st.error(f"{env_name} missing.")
                else:
                    out = magic_run(magic, provider, model, api_key, st.session_state["note_raw"], lang=lang, max_tokens=int(max_tokens))
                    st.session_state["note_md"] = out
                    st.session_state["note_render_html"] = coral_highlight(out)
        else:
            st.caption(t(lang, "keyword_colors"))
            pairs = st.session_state["keyword_pairs"]
            for i in range(len(pairs)):
                ckw, ccol = st.columns([1.2, 0.8], vertical_alignment="center")
                with ckw:
                    pairs[i] = (st.text_input(f"{t(lang,'keywords')} #{i+1}", value=pairs[i][0], key=f"kw_{i}"), pairs[i][1])
                with ccol:
                    pairs[i] = (pairs[i][0], st.color_picker(f"Color #{i+1}", value=pairs[i][1], key=f"kc_{i}"))
            st.session_state["keyword_pairs"] = pairs

            if st.button(t(lang, "apply"), use_container_width=True, key="note_apply_keywords"):
                base = st.session_state["note_md"] or st.session_state["note_raw"]
                st.session_state["note_render_html"] = apply_keyword_colors(base, st.session_state["keyword_pairs"])

    with right:
        tabA, tabB, tabC = st.tabs([t(lang, "markdown_edit"), t(lang, "text_edit"), t(lang, "render")])
        with tabA:
            st.session_state["note_md"] = st.text_area("Markdown", st.session_state["note_md"], height=540, key="note_md_edit")
        with tabB:
            st.session_state["note_raw"] = st.text_area("Text", st.session_state["note_raw"], height=540, key="note_raw_edit")
        with tabC:
            html = st.session_state["note_render_html"] or coral_highlight(st.session_state["note_md"] or st.session_state["note_raw"])
            st.markdown(f"<div class='wow-card editor-frame'>{html}</div>", unsafe_allow_html=True)

# -----------------------------
# Mode: Command Center
# -----------------------------
else:
    st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'workspace')}</h3></div>", unsafe_allow_html=True)
    paneA, paneB = st.columns([1.05, 1.0], gap="large")

    with paneA:
        st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'source_material')}</h4></div>", unsafe_allow_html=True)
        a1, a2, a3 = st.tabs([t(lang, "pdf_viewer"), t(lang, "ocr_editor"), t(lang, "raw_text")])

        with a1:
            pdf = st.file_uploader("Upload PDF", type=["pdf"], key="cc_pdf_upload")
            if pdf:
                st.session_state["pdf_bytes"] = pdf.read()
                st.session_state["trimmed_pdf_bytes"] = None

            if st.session_state["pdf_bytes"]:
                st.markdown(f"<div class='wow-mini'><b>{t(lang,'trim_pages')}</b></div>", unsafe_allow_html=True)
                ranges = st.text_input(t(lang, "page_ranges"), value="1-2", key="cc_ranges")
                do_preview = st.checkbox(t(lang, "render_pdf"), value=True, key="cc_preview")

                colA, colB = st.columns([1, 1])
                with colA:
                    if st.button(t(lang, "trim_extract"), use_container_width=True, key="cc_trim_extract"):
                        try:
                            pr = parse_page_ranges(ranges) or [(1, 1)]
                            trimmed = trim_pdf_bytes(st.session_state["pdf_bytes"], pr)
                            st.session_state["trimmed_pdf_bytes"] = trimmed
                            st.session_state["raw_text"] = extract_text_pypdf2(trimmed)
                            st.session_state["ocr_text"] = st.session_state["raw_text"]
                        except Exception as e:
                            st.error(f"Trim/Extract failed: {e}")

                with colB:
                    trimmed = st.session_state["trimmed_pdf_bytes"] or st.session_state["pdf_bytes"]
                    st.download_button(t(lang, "download_trimmed"), data=trimmed, file_name="trimmed.pdf", use_container_width=True, key="cc_download_trimmed")

                if do_preview:
                    st.markdown(render_pdf_iframe(st.session_state["trimmed_pdf_bytes"] or st.session_state["pdf_bytes"]), unsafe_allow_html=True)

                st.divider()
                st.markdown(f"<div class='wow-mini'><b>{t(lang,'ocr_engine')}</b></div>", unsafe_allow_html=True)
                ocr_engine = st.selectbox(t(lang, "ocr_engine"), [t(lang, "extract_text"), t(lang, "local_ocr"), t(lang, "vision_ocr")], index=0, key="cc_ocr_engine")
                ocr_ranges = st.text_input(t(lang, "ocr_pages"), value=ranges, key="cc_ocr_ranges")

                if st.button(f"{t(lang,'ocr')} {t(lang,'run_agent')}", use_container_width=True, key="cc_run_ocr"):
                    try:
                        pr = parse_page_ranges(ocr_ranges) or [(1, 1)]
                        pdf_bytes = st.session_state["trimmed_pdf_bytes"] or st.session_state["pdf_bytes"]
                        trimmed_for_ocr = trim_pdf_bytes(pdf_bytes, pr)

                        if ocr_engine == t(lang, "extract_text"):
                            st.session_state["ocr_text"] = extract_text_pypdf2(trimmed_for_ocr)
                        elif ocr_engine == t(lang, "local_ocr"):
                            images = convert_from_bytes(trimmed_for_ocr, dpi=220)
                            pages = []
                            for i, img in enumerate(images, start=1):
                                pages.append(f"\n\n--- PAGE {i} ---\n{pytesseract.image_to_string(img)}")
                            st.session_state["ocr_text"] = "\n".join(pages).strip()
                        else:
                            vprov = st.selectbox("Vision provider", ["openai", "gemini"], index=0, key="cc_vision_provider")
                            vmodel = st.selectbox("Vision model", provider_model_map()[vprov], index=0, key="cc_vision_model")
                            env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[vprov]
                            api_key = env_or_session(env_name)
                            if not api_key:
                                st.error(f"{env_name} missing.")
                            else:
                                images = convert_from_bytes(trimmed_for_ocr, dpi=220)
                                st.session_state["ocr_text"] = call_vision_ocr(vprov, vmodel, api_key, images, lang=lang, max_tokens=12000)
                    except Exception as e:
                        st.error(f"OCR failed: {e}")

        with a2:
            st.session_state["ocr_text"] = st.text_area("OCR Text", st.session_state["ocr_text"], height=520, key="cc_ocr_text")
            st.caption("Coral highlights render on the Intelligence side.")

        with a3:
            st.session_state["raw_text"] = st.text_area("Raw extracted text", st.session_state["raw_text"], height=520, key="cc_raw_text")

    with paneB:
        st.markdown(f"<div class='wow-card'><h4 style='margin:0'>{t(lang,'intelligence_deck')}</h4></div>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.tabs([t(lang, "agent_outputs"), t(lang, "search_results"), t(lang, "final_report"), t(lang, "dataset_studio")])

        with b2:
            if query:
                st.markdown(f"<div class='wow-mini'><b>{t(lang,'device360')}</b></div>", unsafe_allow_html=True)
                d = d360 or {}
                top = d.get("top_510k") or {}
                st.markdown(
                    f"<div class='wow-card'><b>{top.get('device_name','—')}</b><br/>K#: {top.get('k_number','—')} | Product Code: {top.get('product_code','—')}<br/>Applicant: {top.get('applicant','—')}</div>",
                    unsafe_allow_html=True,
                )

                rc = d.get("recalls") or []
                md = d.get("mdr_examples") or []
                gu = d.get("gudid_examples") or []

                cA, cB, cC = st.columns(3)
                with cA:
                    st.markdown("<div class='wow-mini'><b>Recall</b></div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(rc), use_container_width=True, height=220) if rc else st.write("—")
                with cB:
                    st.markdown("<div class='wow-mini'><b>MDR/ADR</b></div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(md), use_container_width=True, height=220) if md else st.write("—")
                with cC:
                    st.markdown("<div class='wow-mini'><b>GUDID</b></div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(gu), use_container_width=True, height=220) if gu else st.write("—")

                st.divider()
                results = engine.search_all(query)
                for name in ["510k", "recall", "adr", "gudid"]:
                    st.markdown(f"<div class='wow-mini'><b>{name.upper()}</b> ({len(results[name])})</div>", unsafe_allow_html=True)
                    if results[name]:
                        st.dataframe(pd.DataFrame([r.record for r in results[name]]), use_container_width=True, height=220)

        with b1:
            if not agents_cfg or not agents_cfg.agents:
                st.warning("No agents loaded. Upload or edit agents.yaml in the sidebar.")
            else:
                agent_names = [f"{a.name} ({a.id})" for a in agents_cfg.agents]
                pick = st.selectbox("Select agent", agent_names, index=0, key="agent_pick")
                agent = agents_cfg.agents[agent_names.index(pick)]

                pmap = provider_model_map()
                provider = st.selectbox(t(lang, "provider"), list(pmap.keys()), index=(list(pmap.keys()).index(agent.provider) if agent.provider in pmap else 0), key="agent_provider")
                model = st.selectbox(t(lang, "model"), pmap[provider], index=0, key="agent_model")
                max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=12000, step=256, key="agent_max_tokens")

                system_prompt = st.text_area(t(lang, "system_prompt"), value=agent.system_prompt, height=140, key="agent_system_prompt")
                user_prompt = st.text_area(t(lang, "user_prompt"), value=agent.user_prompt, height=140, key="agent_user_prompt")

                source_choice = st.radio(
                    t(lang, "agent_input_source"),
                    [t(lang, "use_last_output"), t(lang, "use_ocr_text"), t(lang, "use_raw_text")],
                    index=0,
                    key="agent_input_source",
                )

                last = st.session_state["agent_outputs"][-1]["edited_output"] if st.session_state["agent_outputs"] else ""
                if source_choice == t(lang, "use_last_output") and last.strip():
                    base_input = last
                elif source_choice == t(lang, "use_raw_text"):
                    base_input = st.session_state["raw_text"]
                else:
                    base_input = st.session_state["ocr_text"]

                run_colA, run_colB = st.columns([1, 1])
                with run_colA:
                    if st.button(t(lang, "execute_next"), use_container_width=True, key="agent_execute"):
                        env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[provider]
                        api_key = env_or_session(env_name)
                        if not api_key:
                            st.error(f"{env_name} missing.")
                        else:
                            full_system = (st.session_state["skill_md"].strip() + "\n\n" + system_prompt.strip()).strip()
                            full_user = f"{user_prompt.strip()}\n\n---\nINPUT:\n{base_input}"
                            try:
                                out = call_llm_text(provider, model, api_key, full_system, full_user, max_tokens=int(max_tokens), temperature=float(agent.temperature))
                                st.session_state["agent_outputs"].append(
                                    {"agent_id": agent.id, "name": agent.name, "provider": provider, "model": model, "input": base_input, "output": out, "edited_output": out}
                                )
                            except Exception as e:
                                st.error(f"Agent run failed: {e}")

                with run_colB:
                    if st.button("Append last output to Final Report", use_container_width=True, key="agent_append_final"):
                        if st.session_state["agent_outputs"]:
                            st.session_state["final_report"] += "\n\n" + st.session_state["agent_outputs"][-1]["edited_output"]

                st.divider()
                if st.session_state["agent_outputs"]:
                    for i, run in enumerate(reversed(st.session_state["agent_outputs"])):
                        idx = len(st.session_state["agent_outputs"]) - 1 - i
                        st.markdown(f"<div class='wow-mini'><b>Run {idx+1}</b> — {run['name']} ({run['provider']} / {run['model']})</div>", unsafe_allow_html=True)
                        v1, v2 = st.tabs([f"Render #{idx+1}", f"{t(lang,'edit_output_for_next')} #{idx+1}"])
                        with v1:
                            html = coral_highlight(run["output"])
                            st.markdown(f"<div class='wow-card editor-frame'>{html}</div>", unsafe_allow_html=True)
                        with v2:
                            st.session_state["agent_outputs"][idx]["edited_output"] = st.text_area("", value=run["edited_output"], height=220, key=f"edited_{idx}")

        with b3:
            report_tabs = st.tabs([t(lang, "markdown_edit"), t(lang, "render")])
            with report_tabs[0]:
                st.session_state["final_report"] = st.text_area("Final Report (Markdown)", st.session_state["final_report"], height=540, key="final_md")
            with report_tabs[1]:
                html = coral_highlight(st.session_state["final_report"])
                st.markdown(f"<div class='wow-card editor-frame'>{html}</div>", unsafe_allow_html=True)

        with b4:
            st.markdown(f"<div class='wow-card'><h3 style='margin:0'>{t(lang,'dataset_studio')}</h3></div>", unsafe_allow_html=True)

            ds_type = st.selectbox(t(lang, "dataset_type"), ["510k", "recall", "adr", "gudid"], index=0, key="ds_type")

            st.markdown(f"<div class='wow-mini'><b>{t(lang,'paste_dataset')}</b></div>", unsafe_allow_html=True)

            # BUGFIX: widget key must NOT be the same as the session_state key we assign to.
            st.session_state["ds_input_text"] = st.text_area(
                "",
                st.session_state["ds_input_text"],
                height=160,
                key="ds_input_text_widget",  # changed from "ds_input_text"
            )

            upl = st.file_uploader(t(lang, "upload_dataset"), type=["csv", "json", "txt", "md"], key="ds_upload")
            st.caption("Supported: CSV/JSON/TXT. JSON can be a list of objects or wrapped under keys like data/records/items.")

            colL, colR = st.columns([1, 1])
            with colL:
                if st.button(t(lang, "parse_load"), use_container_width=True, key="ds_parse_load"):
                    try:
                        if upl is not None:
                            df_in = parse_dataset_blob(upl.read(), filename=upl.name)
                        else:
                            df_in = parse_dataset_blob(st.session_state["ds_input_text"], filename=None)

                        df_std, rep = standardize_df(ds_type, df_in)
                        st.session_state["ds_std_report"] = rep

                        if ds_type == "510k":
                            st.session_state["df_510k"] = df_std
                        elif ds_type == "adr":
                            st.session_state["df_adr"] = df_std
                        elif ds_type == "gudid":
                            st.session_state["df_gudid"] = df_std
                        else:
                            st.session_state["df_recall"] = df_std

                        st.session_state["ds_filtered_df"] = pd.DataFrame()
                        st.session_state["ds_summary_md"] = ""
                        st.session_state["ds_query_md"] = ""

                        engine = build_engine_from_session()
                        st.success(f"{ds_type} loaded & standardized. {t(lang,'loaded_rows')}: {len(df_std)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")

            with colR:
                if st.button(t(lang, "reset_defaults"), use_container_width=True, key="ds_reset_defaults"):
                    st.session_state["df_510k"] = pd.DataFrame(DEFAULT_510K)
                    st.session_state["df_adr"] = pd.DataFrame(DEFAULT_ADR)
                    st.session_state["df_gudid"] = pd.DataFrame(DEFAULT_GUDID)
                    st.session_state["df_recall"] = pd.DataFrame(DEFAULT_RECALL)
                    st.session_state["ds_std_report"] = ""
                    st.session_state["ds_filtered_df"] = pd.DataFrame()
                    st.session_state["ds_summary_md"] = ""
                    st.session_state["ds_query_md"] = ""
                    engine = build_engine_from_session()
                    st.rerun()

            if st.session_state["ds_std_report"]:
                with st.expander(t(lang, "standardization_report"), expanded=False):
                    st.markdown(st.session_state["ds_std_report"])

            cur_df = (
                st.session_state["df_510k"] if ds_type == "510k"
                else st.session_state["df_adr"] if ds_type == "adr"
                else st.session_state["df_gudid"] if ds_type == "gudid"
                else st.session_state["df_recall"]
            )

            st.markdown(f"<div class='wow-mini'><b>{t(lang,'preview')}</b></div>", unsafe_allow_html=True)
            st.caption(f"{t(lang,'loaded_rows')}: {len(cur_df)}")
            st.dataframe(cur_df.head(20), use_container_width=True, height=260)

            dcol1, dcol2 = st.columns([1, 1])
            with dcol1:
                st.download_button(t(lang, "download_csv"), data=cur_df.to_csv(index=False).encode("utf-8"), file_name=f"{ds_type}_standardized.csv", use_container_width=True, key="ds_dl_csv")
            with dcol2:
                st.download_button(t(lang, "download_json"), data=df_to_json_records(cur_df).encode("utf-8"), file_name=f"{ds_type}_standardized.json", use_container_width=True, key="ds_dl_json")

            st.divider()
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'dataset_summary')}</b></div>", unsafe_allow_html=True)

            pmap = provider_model_map()
            sum_provider = st.selectbox(t(lang, "provider"), list(pmap.keys()), index=0, key="ds_sum_provider")
            sum_model = st.selectbox(t(lang, "model"), pmap[sum_provider], index=0, key="ds_sum_model")
            sum_max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=12000, step=256, key="ds_sum_max_tokens")

            default_sum_prompt = (
                "請根據提供的資料集（已標準化）撰寫一份全面摘要（Markdown），目標長度 1000~2000 字。\n"
                "摘要需包含：\n"
                "1) 資料集概況（筆數、欄位、缺漏、可用性）\n"
                "2) 關鍵欄位的主要分佈/常見值（例如 product_code、recall_class、patient_outcome、mri_safety 等，視資料集類型而定）\n"
                "3) 明顯異常/缺口（例如關鍵欄位大量缺漏、格式不一致、可疑值）\n"
                "4) 可用於法規審查的洞察（例如：召回等級與原因模式、MDR 事件模式、GUDID 屬性風險線索、510(k) predicate 線索）\n"
                "5) 建議下一步（適合在本系統內用關鍵字搜尋/代理分析的方向）\n"
                "注意：不可捏造資料；只能根據提供內容推導，推定需標註為推定。"
            ) if lang == "zh-TW" else (
                "Write a comprehensive Markdown summary of the standardized dataset (target 1000–2000 words). Include:\n"
                "1) dataset overview (rows, columns, missingness, usability)\n"
                "2) key field distributions/top values\n"
                "3) anomalies/gaps (missing critical fields, inconsistent formats)\n"
                "4) regulatory review insights\n"
                "5) recommended next steps for keyword search and agent analysis\n"
                "Do not fabricate; only infer from provided content; label assumptions."
            )

            if not st.session_state["ds_summary_prompt"].strip():
                st.session_state["ds_summary_prompt"] = default_sum_prompt

            # BUGFIX: widget key must NOT equal session_state key we assign to.
            st.session_state["ds_summary_prompt"] = st.text_area(
                "Summary prompt",
                value=st.session_state["ds_summary_prompt"],
                height=180,
                key="ds_summary_prompt_widget",  # changed from "ds_summary_prompt"
            )

            if st.button(t(lang, "generate_summary"), use_container_width=True, key="ds_generate_summary"):
                env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[sum_provider]
                api_key = env_or_session(env_name)
                if not api_key:
                    st.error(f"{env_name} missing.")
                else:
                    ctx = dataset_context_markdown(ds_type, cur_df, max_rows=50)
                    sys = "You are a regulatory data analyst. Output Markdown." if lang != "zh-TW" else "你是法規資料分析專家，請輸出 Markdown。"
                    user = st.session_state["ds_summary_prompt"].strip() + "\n\n---\n" + ctx
                    try:
                        st.session_state["ds_summary_md"] = call_llm_text(sum_provider, sum_model, api_key, sys, user, max_tokens=int(sum_max_tokens), temperature=0.2)
                    except Exception as e:
                        st.error(f"Summary failed: {e}")

            sum_tabs = st.tabs([t(lang, "markdown_edit"), t(lang, "render")])
            with sum_tabs[0]:
                st.session_state["ds_summary_md"] = st.text_area("Summary Markdown", value=st.session_state["ds_summary_md"], height=320, key="ds_summary_md_edit")
            with sum_tabs[1]:
                html = coral_highlight(st.session_state["ds_summary_md"])
                st.markdown(f"<div class='wow-card editor-frame'>{html}</div>", unsafe_allow_html=True)

            st.divider()
            st.markdown(f"<div class='wow-mini'><b>{t(lang,'dataset_query')}</b></div>", unsafe_allow_html=True)

            st.session_state["ds_keyword"] = st.text_input(t(lang, "keyword_search"), value=st.session_state["ds_keyword"], key="ds_keyword_input")

            fcol1, fcol2 = st.columns([1, 1])
            with fcol1:
                if st.button(t(lang, "filter_results"), use_container_width=True, key="ds_filter_btn"):
                    st.session_state["ds_filtered_df"] = keyword_filter_df(cur_df, st.session_state["ds_keyword"], limit=50)
            with fcol2:
                use_filtered = st.checkbox(t(lang, "use_filtered"), value=True, key="ds_use_filtered")

            filtered_df = st.session_state["ds_filtered_df"]
            if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
                st.caption(f"Filtered rows: {len(filtered_df)}")
                st.dataframe(filtered_df.head(20), use_container_width=True, height=220)
            elif st.session_state["ds_keyword"].strip():
                st.caption("No matches (or not filtered yet).")

            qmap = provider_model_map()
            q_provider = st.selectbox(t(lang, "provider"), list(qmap.keys()), index=0, key="ds_q_provider")
            q_model = st.selectbox(t(lang, "model"), qmap[q_provider], index=0, key="ds_q_model")
            q_max_tokens = st.number_input(t(lang, "max_tokens"), min_value=512, max_value=12000, value=12000, step=256, key="ds_q_max_tokens")
            rows_in_ctx = st.number_input(t(lang, "rows_in_context"), min_value=5, max_value=200, value=50, step=5, key="ds_rows_in_ctx")

            default_q_prompt = (
                "請根據提供的資料集回答以下問題，輸出 Markdown。\n"
                "- 請先列出你使用到的欄位與推導步驟（可簡要）。\n"
                "- 若資料不足，請以 Gap 標註並說明需要的欄位或資料。\n\n"
                "問題：\n"
            ) if lang == "zh-TW" else (
                "Answer the question based on the provided dataset context. Output Markdown.\n"
                "- List the fields used and reasoning steps briefly.\n"
                "- If insufficient data, mark as Gap and specify needed fields.\n\n"
                "Question:\n"
            )

            if not st.session_state["ds_query_prompt"].strip():
                st.session_state["ds_query_prompt"] = default_q_prompt

            # BUGFIX: widget key must NOT equal session_state key we assign to.
            st.session_state["ds_query_prompt"] = st.text_area(
                t(lang, "llm_prompt"),
                value=st.session_state["ds_query_prompt"],
                height=160,
                key="ds_query_prompt_widget",  # changed from "ds_query_prompt"
            )

            if st.button(t(lang, "run_query"), use_container_width=True, key="ds_run_query"):
                env_name = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "xai": "XAI_API_KEY"}[q_provider]
                api_key = env_or_session(env_name)
                if not api_key:
                    st.error(f"{env_name} missing.")
                else:
                    use_df = filtered_df if (use_filtered and isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty) else cur_df
                    ctx = dataset_context_markdown(ds_type, use_df, max_rows=int(rows_in_ctx))
                    sys = "You are a regulatory dataset analyst. Output Markdown." if lang != "zh-TW" else "你是法規資料集分析助理，請輸出 Markdown。"
                    user = st.session_state["ds_query_prompt"].strip() + "\n\n---\n" + ctx
                    try:
                        st.session_state["ds_query_md"] = call_llm_text(q_provider, q_model, api_key, sys, user, max_tokens=int(q_max_tokens), temperature=0.2)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

            q_tabs = st.tabs([t(lang, "markdown_edit"), t(lang, "render")])
            with q_tabs[0]:
                st.session_state["ds_query_md"] = st.text_area(t(lang, "query_results"), value=st.session_state["ds_query_md"], height=320, key="ds_query_md_edit")
            with q_tabs[1]:
                html = coral_highlight(st.session_state["ds_query_md"])
                st.markdown(f"<div class='wow-card editor-frame'>{html}</div>", unsafe_allow_html=True)
