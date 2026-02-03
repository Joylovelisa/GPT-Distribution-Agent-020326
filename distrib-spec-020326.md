WOW 配送/流向分析工作室（Hugging Face Spaces / Streamlit）— 綜合技術規格書（Technical Specification）
本文件描述「WOW 配送/流向分析工作室」的完整技術規格，涵蓋資料匯入與標準化、互動儀表板（含 11 種 WOW 視覺化）、兩份資料集比較模組（含 5 張比較圖與可點擊網路節點明細）、Agentic AI 工作流程（agents.yaml + SKILL.md）、多供應商模型路由（OpenAI / Gemini / Anthropic / Grok）、WOW UI 主題系統（亮/暗、語言切換、20 畫家風格）、部署與安全性、狀態指標與錯誤處理等。

1. 系統目標與使用情境
1.1 目標
系統旨在讓使用者在瀏覽器中快速完成以下工作：

匯入配送/流向資料（貼上或上傳 CSV/JSON/TXT，或使用預設資料集）。
自動將非標準資料轉換為系統的 canonical schema（標準化資料模型），並提供標準化報告。
在 WOW 互動儀表板中，以可篩選、可下鑽、可下載的方式探索資料。
比較兩份配送資料集（A/B），在各自獨立篩選後輸出比較摘要（Markdown）與 5 張比較視覺化（含網路圖）。
使用 Agentic AI：可編輯 prompts、模型、max_tokens（預設 12000 上限）、逐一執行 agent，並允許使用者把 agent 輸出編修後作為下一 agent 的輸入。
管理 agents.yaml 與 SKILL.md：可貼上、上傳、下載、重置。若 agents.yaml 非標準 schema，系統可進行 heuristic 標準化，必要時可用 LLM 進行強化標準化。
1.2 使用者角色
業務/營運分析師：關注供應商→客戶路徑、TopN、週期節奏、集中度與異常。
稽核/合規：關注許可證一致性、UDI/序號/批號紀錄品質、可追溯性與疑似重複入帳。
資料工程/產品：關注資料標準化 mapping、欄位缺失、輸出下載格式、效能與可維運性。
管理者：透過 AI 產出高密度摘要、決策建議與下一步行動。
2. 部署與執行環境
2.1 平台
Hugging Face Spaces（建議使用 Streamlit Space）。
執行框架：Streamlit（單一 app.py）。
視覺化：Plotly（plotly.express + graph_objects）。
LLM API：OpenAI / Google Gemini / Anthropic Claude / xAI Grok（透過 API keys）。
2.2 設定檔與資料檔（Repo 內）
app.py：主程式（UI、資料處理、視覺化、agent 管線）。
agents.yaml：可由 UI 管理的 agent 定義（標準 schema）。
SKILL.md：全域 system prompt（與 agent 的 system_prompt 合併）。
defaultdataset.json：預設配送資料集（要求從 app.py 移出，由 app 啟動載入）。
（可選）requirements.txt：依賴清單，若需網路節點點擊功能，包含 streamlit-plotly-events。
3. 核心資料模型（Canonical Schema）
3.1 Canonical 欄位定義
系統統一輸出/分析使用下列欄位（snake_case）：

Canonical 欄位	說明	類型（目標）
supplier_id	供應商代碼	string
deliver_date	出貨/交貨日期	datetime（pandas Timestamp）
customer_id	客戶代碼	string
license_no	許可證字號	string
category	分類/品項類別	string
udi_di	UDI/DI	string
device_name	品名/裝置名稱	string
lot_no	批號	string
serial_no	序號	string
model	型號	string
quantity	數量	int
3.2 資料標準化（Standardization）
系統支援來源資料欄位同義詞（含中英、不同大小寫、去符號化）模糊對映。
日期欄位支援：
YYYYMMDD（8 碼）解析成 datetime。
其他格式使用 pd.to_datetime(..., errors="coerce")。
數量欄位支援：
字串數字、含逗點格式。
失敗則回落為 0（策略可在 spec 中視需求調整）。
清理規則：
中文引號 “ ” → 標準引號 ".
strip() 去頭尾空白。
篩掉「完全無訊號」列（所有欄位空/NaN 且 quantity=0）。
3.3 標準化報告（Standardization Report）
以 Markdown 表格呈現：

Canonical field → Source column mapping。
缺失欄位列表。
Rows（標準化後筆數）、Original columns 數量。
4. 頁面與模組規格（UI / UX）
4.1 WOW UI 外觀系統
Theme：Light / Dark。
Language：English / 繁體中文（zh-TW）。
Painter Style（20 種）：如 Monet、Van Gogh、Hokusai…每個 style 定義 accent 色。
Jackpot：隨機抽一種畫家風格。
WOW 風格 CSS 特徵：
Glassmorphism（半透明卡片 + blur）。
背景 radial gradients。
chip 狀態膠囊（顯示 API key、資料列數等）。
4.2 WOW 狀態指標（Status Indicators）
頂部狀態列顯示：

OpenAI / Gemini / Anthropic / xAI：狀態（env / session / missing）。
資料列數（Rows）。
可擴充顯示：JSON export OK/FAIL、compare loaded 等（可做為後續強化）。
4.3 API Key 輸入規則
若從環境變數取得（env），不顯示 key 輸入框，避免誤洩漏。
若環境無 key，允許在 UI 輸入（password 欄位），存於 session state（不寫入 repo）。
5. 資料匯入與 Dataset Studio
5.1 資料來源模式
主資料集支援：

Default dataset：啟動時載入 defaultdataset.json。
Paste：貼上 CSV/JSON/TXT（系統嘗試判別並解析）。
Upload：上傳 txt/csv/json。
5.2 defaultdataset.json 規格
支援兩種封裝（建議採 records 形式）：

{"version":"1.0","records":[{...},{...}]}
{"version":"1.0","format":"csv","data":"...csv text..."}
5.3 重新載入預設資料集
提供「Reload default dataset」按鈕：
重新讀取 defaultdataset.json。
重新進行標準化並覆蓋當前 session 的 main dataset。
可選擇是否同步重置 compare 模組 A/B（建議一致）。
5.4 預覽與下載
預覽：顯示 raw 前 20 筆 + standardized 前 20 筆。
下載：
CSV：df.to_csv(index=False)
JSON：輸出 records 格式（見第 9.2 的 JSON 序列化注意事項）。
6. 互動儀表板（Dashboard）與 11 種視覺化
6.1 全域篩選器
supplier_id、license_no、model、customer_id：多選。
date range：日期區間。
keyword search：針對 device_name/category/udi_di/lot_no/serial_no/license_no/model/supplier_id/customer_id 等欄位做 contains 查詢。
篩選順序：集合篩選（多選）→ 日期範圍 → 關鍵字搜尋。
6.2 KPI 摘要
Rows、Total Quantity、Unique suppliers/customers/models/licenses。
日期範圍（min/max）。
6.3 核心 6 視覺化（原始需求）
Sankey 流向圖：Supplier → License → Model → Customer（value = sum(quantity)）。
分層配送網路圖（Layered Network Graph）：四層節點 + 邊（可降載 TopN）。
時間序列趨勢：按日彙總 quantity（可擴充 W/M）。
Top Suppliers：供應商 quantity bar chart。
Sunburst：階層占比 Supplier → License → Model → Customer。
Heatmap：Supplier × Model（TopN）數量熱圖。
6.4 額外 5 個 WOW 視覺化（新增）
Pareto（Bar + cumulative %）：例如 Top Customers 的集中度視角。
Supplier × Customer Flow Heatmap：二維流向強度矩陣（Top suppliers/customers）。
Week × Weekday Rhythm Heatmap：週次與週內節奏熱圖，顯示營運節律。
Treemap（Supplier → License → Model）：更直覺的分割區塊占比。
Boxplot（quantity 分布）：例如 by Supplier 的 quantity 變異視角。
6.5 互動表格與下載
顯示篩選後 df（可設定顯示高度）。
允許下載 CSV/JSON（注意 JSON 序列化）。
7. 兩份資料集比較模組（Compare Two Datasets）
7.1 匯入與標準化
Dataset A / Dataset B 各自支援：
Paste / Upload / Default（讀 defaultdataset.json）。
系統在載入後自動標準化為 canonical schema。
預覽：raw 前 20、standardized 前 20。
A/B 各自生成 standardization report。
7.2 A/B 獨立篩選器
A 與 B 各自有 supplier/license/model/customer/date/search 的獨立篩選器。
系統將篩選後資料分別記為 dfA_f、dfB_f，用於比較 KPI/圖表/摘要。
7.3 比較摘要（Markdown）
內建模板輸出：

Rows / Total Quantity / unique counts 對比。
日期範圍對比。
解讀建議（保守，不捏造）。
7.4 5 張比較圖（含網路圖）
KPI Comparison bar（A vs B）。
Trend Overlay line（A vs B）。
Top Suppliers comparison（group bar）。
Top Models comparison（group bar）。
Compare Network Graph（distribution network graph）：
節點顏色標示 A-only / B-only / both（例如藍/橘/珊瑚色）。
邊使用 A+B 彙總後的 Top flow 顯示，避免爆量。
7.5 點擊網路節點顯示詳細資訊
若安裝 streamlit-plotly-events：
使用者點擊節點（pointIndex）→ 系統查 node_meta → 得到 layer/value/presence → 在側欄/下方顯示明細。
明細內容：
Dataset A matched rows（前 20）+ Dataset B matched rows（前 20）。
Related top connections（例如點 Supplier，顯示其 top Licenses；點 Model，顯示其 top Customers）。
若未安裝事件套件：
fallback：僅顯示提示，使用者透過篩選器手動下鑽。
8. Agentic AI 系統（Agent Studio）
8.1 核心設計
agents.yaml 定義多個 agent（每個 agent 可指定 provider/model/temperature/max_tokens/system_prompt/user_prompt）。
SKILL.md 作為全域 system prompt（在每次 agent 呼叫時與 agent 的 system_prompt 合併）。
執行模式：使用者選 agent → 自訂 prompt、max_tokens、model → 執行 → 產生輸出。
8.2 逐一執行與可編修輸出
每次 run 會記錄：
ts、agent_id/name、provider/model、temperature/max_tokens、system/user prompts、input、output、edited_output。
使用者可在 UI 編輯 edited_output，並將它作為下一個 agent 的輸入（形成人工可控的 agent pipeline）。
8.3 Agent 輸入內容（預設組合）
dataset summary（JSON）
使用者可選視覺化/分析指令（viz_instructions）
standardized 前 20 筆樣本（CSV） -（可延伸）TopN 彙總表、路徑彙總表以提高 AI 準確性（建議後續增加）。
9. Config Studio（agents.yaml / SKILL.md 管理）
9.1 agents.yaml 管理
功能：

Paste 編輯 / Upload / Download / Reset to defaults。
Standardize now：
heuristic 標準化：將多種 YAML 形狀轉為標準 schema。
顯示 diff（before vs after）。
LLM Auto-standardize（可選）：
當 YAML 難以解析或結構非常混亂時，使用 LLM 轉成標準 schema（只輸出 YAML）。
需 API key，並提供 provider/model 選擇。
9.2 SKILL.md 管理
Paste / Upload / Download / Reset to defaults。
Preview：render markdown 供檢視。
9.3 Agent Editor（UI 編輯器）
以表單方式編輯單一 agent 的：
provider/model/max_tokens/temperature
system_prompt/user_prompt
name/description
套用後自動回寫到 session 的 agents_cfg 並重新輸出 agents.yaml 文本。
10. 錯誤處理與已知 Bug 類型
10.1 Plotly marker.color 型別錯誤（已知類型）
在 go.Scatter(marker.color=...) 中使用任意字串（如 “Supplier”）會造成 plotly validator 失敗。
解法策略（規格層面）：
將 layer/category 映射為合法色碼（HEX/RGB）或數值 + colorscale。
10.2 JSON 序列化錯誤（Timestamp not JSON serializable）
json.dumps(df.to_dict(...)) 遇到 pandas Timestamp/NaT/numpy scalar 可能失敗。
規格建議：
優先使用 df.to_json(orient="records", date_format="iso") 或在輸出前將 datetime 轉 ISO 字串。
下載 JSON 前可加入 export self-test 與狀態燈。
10.3 defaultdataset.json 缺失
若 repo 未包含 defaultdataset.json：
UI 顯示警告（default missing）。
允許使用者改用 paste/upload 模式。
11. 安全性與隱私
11.1 API Key
由 env 提供時不在 UI 顯示輸入框。
session key 只存於 runtime session state，不寫入 repo，不顯示明文。
11.2 資料最小揭露
預設表格預覽只顯示前 20 或前 50 行（可調）。
AI 輸入建議採用 summary + 小樣本，避免上送完整敏感資料。
建議後續加入遮蔽策略（customer_id/serial_no/lot_no/udi_di）。
12. 效能與擴展性
12.1 大資料量策略（建議）
所有 network/sankey/heatmap 預設採 TopN 或 min-flow 門檻降載。
避免在 UI 每次 rerun 都做大型 groupby：建議後續加入 caching（st.cache_data）並以 filters 作為 cache key。
compare network edges 以 TopM 限制。
12.2 互動事件的依賴
點擊節點明細依賴 streamlit-plotly-events，未安裝則 fallback。
建議在 requirements 內明確列為 optional dependency，並在 UI 顯示 “events enabled/disabled”。
13. 測試與驗收（UAT）範圍摘要
Default dataset：啟動載入、按鈕重載。
Paste/Upload：CSV/JSON/TXT 解析。
Standardization：mapping 正確性、缺欄提示。
Dashboard：11 圖可渲染、filters 生效、下載可用。
Compare：A/B 載入、獨立篩選、5 圖比較、網路節點點擊明細（或 fallback）。
Config：agents.yaml / SKILL.md 的 paste/upload/download/reset、heuristic standardize、LLM standardize。
Agent pipeline：自訂模型、max_tokens、prompt、逐一執行、edited_output 傳遞。
20 個後續釐清問題（Comprehensive Follow-up Questions）
defaultdataset.json 你希望固定採 records list 格式，還是同時支援 CSV string 形式以減少檔案大小？

重新載入預設資料集時，你希望是否要 重置所有篩選器狀態與 compare 的點擊節點狀態，避免使用者誤以為仍在原資料篩選結果上？

defaultdataset.json 是否需要包含 metadata（例如 name、version、updated_at、source）並在 WOW 狀態列顯示？

你希望 Compare 模組中 A/B 預設都載入 defaultdataset，還是 A 載入 default、B 保持空白讓使用者上傳？

在 Dashboard 的 11 個視覺化中，哪些必須固定顯示，哪些應該做成「可選擴充（Advanced）」以免頁面過長？

Network/Sankey 在資料量大時，你希望預設 TopN 的策略是依節點總量、還是依邊流量（min flow threshold）？需要 UI slider 嗎？

點擊節點明細目前顯示 matched rows + related top connections；你是否希望進一步顯示 A vs B 差異表（delta quantity、rank change）？

Compare 的 5 張圖目前包含 Top Suppliers/Top Models；你是否想改成 Top Customers/Top Licenses 或允許使用者選擇比較維度？

趨勢圖目前是按日（D）；你是否需要 D/W/M 切換，並在 A/B 日期範圍不同時提供「交集/聯集/補 0」三種模式？

你是否需要在儀表板加入「資料品質指標」卡片（日期無效比例、quantity=0 比例、缺欄比例、疑似重複比例）？

你希望 JSON 下載的 deliver_date 以 YYYY-MM-DD 或 ISO 8601 含時間 輸出？是否要統一到 UTC？

Agent 的預設輸入目前是 summary + 前 20 筆樣本；你是否希望加入「TopN 彙總表/路徑彙總表」以降低 AI 依賴樣本、提高可靠性？

是否需要提供「敏感欄位遮蔽」開關，控制 AI 輸入與 UI 表格中 customer_id/serial_no/lot_no/udi_di 的顯示方式？

agents.yaml 目前支援 heuristic standardize 與 LLM standardize；你希望哪個是預設？是否要在上傳時自動標準化？

Agent Editor 是否需要批量操作（例如一次改所有 agents 的 provider/model/max_tokens）以便快速切換成本策略？

Compare Network Graph 是否也需要支援「點擊邊」顯示明細（某 Supplier→License 的所有紀錄），還是點節點就足夠？

你希望 defaultdataset.json 可由使用者在 UI 產生/下載（把目前資料存成新的預設檔）嗎？在 HF Spaces 下對 repo 的寫入是否需要避免？

是否需要新增「報告匯出」功能：把 compare 摘要、5 張圖、AI 輸出整合成 Markdown（甚至 PDF）一鍵下載？

你希望 WOW UI 的畫家風格是否也影響 Plotly 圖表的 theme（配色、背景、gridline）以達到一致的視覺語言？

系統未來是否要支援多資料集版本管理（同時載入多個檔案、合併、比較多於兩份、或按版本標籤追蹤差異）以符合真實營運流程？
