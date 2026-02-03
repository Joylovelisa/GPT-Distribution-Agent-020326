# Comprehensive Technical Specification: FDA Agentic AI System (Improved Design)

## 1. Executive Summary

This document outlines the technical specification for an enhanced **FDA Agentic AI System**, a specialized "Regulatory Command Center" designed to assist medical device professionals, regulatory affairs specialists, and FDA reviewers. The system integrates advanced Large Language Model (LLM) agents with traditional data processing tools to streamline the analysis of FDA datasets (510(k), Recalls, ADR, GUDID) and the review of regulatory documents (PDFs).

The core improvements focus on a "WOW" User Interface (UI) that delivers high-impact visualizations, a unified multi-dataset search engine with intelligent suggestions, and a flexible, agent-driven document processing pipeline. The system provides a seamless workflow from document ingestion (PDF trimming, OCR) to deep semantic analysis using customizable AI agents. Furthermore, a dedicated batch processing capability automates the structural analysis of large repositories of PDF files, generating master Table of Contents (ToC) and summaries for high-level oversight.

The system is architected to be deployed on **Hugging Face Spaces** using **Streamlit**, leveraging **OpenAI** and **Google Gemini** APIs for intelligence, while maintaining strict data privacy and user control over API keys.

---

## 2. System Architecture

The system follows a modular micro-services-like architecture within a monolithic Streamlit application. It is composed of four primary layers:

### 2.1. Presentation Layer (The "WOW" UI)
*   **Framework**: Streamlit (Python).
*   **Design System**: Glassmorphism aesthetic with "Coral" accent colors (#FF7F50) for high-contrast highlighting. Custom CSS injection for cards, floating action buttons (FABs), and sophisticated typography.
*   **Components**:
    *   **Global Command Bar**: Unified search input for all datasets.
    *   **Interactive Dashboard**: Plotly/Altair charts for data visualization (timelines, distributions).
    *   **Document Workspace**: Split-screen view for PDF source vs. AI Analysis/Markdown.
    *   **Agent Studio**: Interface for managing, editing, and standardizing `agents.yaml` configurations.

### 2.2. Application Logic Layer
*   **Session State Manager**: Handles user context, API keys, active datasets, and loaded documents.
*   **Search Engine**: A unified query parser that acts as an orchestration layer over individual dataset indices. Implements fuzzy matching (`rapidfuzz`) and keyword expansion.
*   **Agent Engine**: Converting user intent and `agents.yaml` definitions into executable LLM API calls. Handles system prompt injection, context window management, and output parsing.

### 2.3. Data Processing Layer
*   **Document Pipeline**:
    *   **Ingestion**: PyPDF2 for parsing and trimming.
    *   **OCR**: Modular interface supporting `pytesseract` (local Tesseract) and `gpt-4o-vision` / `gemini-1.5-pro` (Vision LLM) for high-fidelity extraction.
    *   **Normalization**: Text-to-Markdown conversion with automated "Coral" keyword highlighting for regulatory signals.
*   **Batch Processor**: A background worker pattern for iterating through file directories, performing batch trimming/summarization, and aggregating results into a Master ToC.

### 2.4. Data Storage & External Interfaces
*   **Datasets**: In-memory loading of JSON/CSV datasets (510k, Recall, ADR, GUDID). Optional local persistence for user-uploaded datasets.
*   **LLM Providers**:
    *   **OpenAI API**: `gpt-4o`, `gpt-4o-mini` (General reasoning, Vision OCR).
    *   **Google Gemini API**: `gemini-2.5-flash`, `gemini-1.5-pro` (Long-context analysis, fast reasoning).
*   **Artifact Store**: Local file system access for saving `agents.yaml` updates, exported Markdown reports, and processed PDFs.

---

## 3. User Interface Specification (WOW UI)

The user interface is critical to the "Review Studio" concept. It must feel premium, responsive, and "alive".

### 3.1. Design Philosophy
*   **Visual Style**: Dark mode default with "Glassmorphism" (translucent cards with blur effects).
*   **Color Palette**: Deep Navy Background (`#0B1020`), White Text, Coral (`#FF7F50`) for regulatory alerts and primary actions, Electric Blue for information.
*   **Typography**: Clean sans-serif (Inter or Roboto), optimized for readability of dense technical text.

### 3.2. Core Views

#### 3.2.1. The "Intelligence Deck" (Dashboard)
This is the landing page.
*   **Hero Section**: A large, centered "Global Search" bar. Placeholder text cycles through examples: "Search for 'Infusion Pump'", "Check 'K240123'", "Analyze 'Latex Allergy'".
*   **Smart Suggestions**: As the user types, a dropdown appears not just with autocompletion, but with *intent suggestions*:
    *   "Search 'Pacemaker' in **510(k)**"
    *   "Find Recalls related to 'Battery'"
    *   "Compare 'Device A' vs 'Device B'"
*   **Live Metrics**: 4 Key Performance Indicator (KPI) cards at the top showing total loaded records for each dataset users have active.

#### 3.2.2. Multi-Dataset Search Results
When a search is executed:
*   **Tabbed Interface**: Results are organized by dataset tabs: `[All]`, `[510(k)]` (Coral badge count), `[Recalls]` (Red badge count), `[ADR]`, `[GUDID]`.
*   **Interactive Visualization**:
    *   **Timeline**: A dot plot showing decision dates or recall dates over time.
    *   **Distribution**: Pie/Donut charts for Device Class, Decision Code (SE/SESE), or Recall severity.
    *   **Interaction**: Clicking a slice in a pie chart filters the list below automatically.
*   **Result Cards**: Each list item is a "Card" containing:
    *   Device Name (Bold).
    *   K-Number / ID (Clickable chip).
    *   Summary excerpt with search terms highlighted in **Coral**.
    *   **Action Buttons**: "Analyze with Agent", "Diff vs Current Doc".

#### 3.2.3. The Document "Operating Room" (Workspace)
For working with a specific file (PDF/Text).
*   **Layout**: Two-column split pane (adjustable ratio).
    *   **Left Pane (Source)**: PDF Viewer (iframe) or Raw Text editor. Features a "Page Selector" for trimming.
    *   **Right Pane (Intelligence)**: Markdown Editor / Agent Output stream.
*   **Toolbar**: Above the workspace.
    *   "Upload/Paste": Toggle input mode.
    *   "Trim & OCR": Opens a modal to configure page ranges and OCR engine.
    *   "Magic Wand": Dropdown to select an Agent from `agents.yaml`.
*   **Coral Highlighting**: In the Right Pane, specific keywords (Predicate, Warning, Contraindication) are automatically wrapped in `<span class="coral">` tags to stand out visually against the dark text.

#### 3.2.4. Agent Studio
A dedicated page for configuring the "brain" of the system.
*   **YAML Editor**: A syntax-highlighted editor for `agents.yaml`.
*   **Agent Cards**: A grid view of loaded agents, showing Name, Model, and Description.
*   **Controls**: "Upload YAML", "Download YAML", "Reset to Default".
*   **Validation**: Real-time error checking for YAML syntax conformance.

---

## 4. Feature Specifications

### 4.1. Unified Multi-Dataset Search & Visualization
**Requirement**: "User can use keyword to search multiple datasets at the same time... interactive visualization."

**Implementation Detail**:
*   **Search Algorithm**:
    1.  Receive query string $Q$.
    2.  Tokenize $Q$ and generate n-grams.
    3.  Compute relevance scores against `device_name`, `applicant`, `summary`, and `k_number` fields across all loaded JSON datasets.
    4.  Apply `Rapidfuzz` for fuzzy matching (threshold > 80%).
    5.  Aggregated Results are sorted by relevance score.
*   **Search Settings Panel**:
    *   **Filters**: Users can toggle "Exact Match", "Fuzzy Level", or filter by Date Range.
    *   **Dataset Toggles**: Checkboxes to include/exclude specific datasets (e.g., "Search only Recalls").
*   **Visualization Engine**:
    *   **Library**: `plotly.express` integrated into Streamlit.
    *   **Dynamic generation**: Based on the result set, the system detects categorical fields (e.g., 'panel', 'device_class') and automatically generates an aggregator chart.
    *   **Drill-down**: Streamlit's `selection` event on charts triggers a callback to filter the `st.dataframe` or card views below.

### 4.2. Advanced Document Workflow (Upload -> Trim -> OCR -> Analyze)
**Requirement**: "First user paste/upload... preview pdf... trim and do OCR... reorganized the doc... select agents."

**Workflow Steps**:
1.  **Input**: User creates a "Case". Uploads `file.pdf` or pastes text.
    *   *System Action*: Stores raw file in `st.session_state`.
2.  **PDF Preview**:
    *   *Backend*: `pdf2image` converts PDF pages to images.
    *   *Frontend*: Renders images in a scrollable container.
3.  **Trimming & OCR Configuration**:
    *   **UI Controls**: Range Slider (e.g., "Pages 1-5") or Comma-separated Input ("1, 3, 5").
    *   **Engine Selection**: Radio button: "Local (Tesseract)" (Fast, Free) vs. "Cloud Vision (GPT-4o/Gemini)" (High Accuracy, Cost).
    *   **Execution**:
        *   If Local: `PyPDF2` extracts text layer. If empty/scanned, `pytesseract` processes images.
        *   If Cloud: Selected pages are converted to Base64 images and sent to the LLM with a prompt: "Transcribe this regulatory document verbatim. Preserve table structures."
4.  **Normalization & Cleanup**:
    *   **Agent**: `ocr_quality_auditor` (Auto-run optional).
    *   **Formatting**: System applies a "Markdown Reconstruct" pass using regex to fix broken headers and list items.
    *   **Coral Injection**: A keyword dictionary (from `regulatory_keyword_signalizer`) scans the text and wraps matches (e.g., "predicate device", "patient death", "class III") in styling tags.
5.  **User Modification**:
    *   The resulting Markdown is displayed in a `st.text_area` or `st_ace` editor. User can manually correct OCR errors.
6.  **Agent Execution**:
    *   User selects an agent (e.g., `substantial_equivalence_writer`).
    *   **Prompt Injection**: System takes the **Current Document Content** + **User's Additional Instructions** + **Agent System Prompt**.
    *   **Model Routing**: User selects `gemini-2.5-flash`, `gemini-1.5-pro`, or `gpt-4o-mini`. System sets the `client` accordingly.
    *   **Streaming Output**: Response is streamed token-by-token into the "Agent Outputs" container.

### 4.3. Agent Management & Standardization
**Requirement**: "Modify/download/upload agents.yaml... transform non-standard to standard."

**Logic**:
*   **Standard Schema**:
    ```json
    {
      "id": "string (unique)",
      "name": "string",
      "description": "string",
      "provider": "enum(openai, gemini)",
      "model": "string",
      "system_prompt": "string",
      "user_prompt": "string"
    }
    ```
*   **Upload Handler**:
    *   When a user uploads a YAML file, the system parses it.
    *   **Validator**: Check for missing keys (`id`, `system_prompt`).
    *   **Auto-Standardizer**: If keys don't match (e.g., file uses `instructions` instead of `system_prompt`), an "Agent Configurer" Agent (LLM) is triggered:
        *   *Prompt*: "Map this arbitrary YAML structure to the standard provided schema. Invent descriptions if missing."
    *   **Result**: The user is presented with a "Diff" showing the standardized version for approval before loading.

### 4.4. Batch Processing System (The "Factory")
**Requirement**: "Find all pdf... trim first page... create summary... create ToC doc... run agents on ToC."

**Module Specification**:
This runs as a separate mode or "Sidebar App".

1.  **File Scanner**:
    *   User inputs a Root Directory path.
    *   System performs a recursive `glob('**/*.pdf')`.
    *   UI displays a "Manifest Table": File Name, Path, Size.
2.  **Batch Actions**:
    *   **Action 1: Trim**: Automates `PdfReader`, extracts Page 1 (Cover Sheet/Summary) of every PDF. Saves as `filename_trimmed.pdf`.
    *   **Action 2: Summarize**:
        *   Iterates through trimmed content.
        *   Sends to LLM (e.g., `gpt-4o-mini`).
        *   *Prompt*: "Summarize this regulatory cover sheet in 3 sentences: Device Name, Applicant, and Primary Indication."
        *   Saves as `filename.md`.
3.  **Master ToC Generation**:
    *   Aggregates all `filename.md` contents.
    *   Formats into a single `Project_Master_ToC.md` document.
    *   Structure:
        ```markdown
        # Project Master Index
        ## 1. [File Name A]
        * Device: ...
        * Applicant: ...
        * Summary: ...
        
        ## 2. [File Name B]
        ...
        ```
4.  **Meta-Analysis**:
    *   The `Project_Master_ToC.md` becomes the "Current Document".
    *   User can now run high-level agents (e.g., `cross_database_context_summarizer`) on this Master ToC to find patterns across the entire file repository.

---

## 5. API & Data Specifications

### 5.1. Dataset Schema (Mock Data Structure)
The system relies on consistent keys to link data.
*   **510(k)**: `k_number` (Primary Key), `device_name`, `applicant`, `product_code`, `decision_date`.
*   **Recalls**: `recall_number`, `k_number` (Foreign Key), `root_cause_description`, `firm_name`.
*   **ADR (Adverse Events)**: `mdr_report_key`, `device_problem_code`, `patient_outcome`.
*   **GUDID**: `primary_di` (UDI), `device_count_in_base_packaging`, `mri_safety_status`.

### 5.2. API Key Management
*   **Storage**: `st.session_state['api_keys']` (Volatile, clears on refresh).
*   **Environment Interaction**:
    *   On launch, check `os.environ.get("OPENAI_API_KEY")` and `os.environ.get("GOOGLE_API_KEY")`.
    *   **Privacy Rule**: If found in Env, user input fields in UI are hidden and replaced with "Authenticated via Environment". If missing, UI shows text input fields.

### 5.3. Deployment Configuration (Hugging Face)
*   `packages.txt`: `poppler-utils` (Required for `pdf2image`), `tesseract-ocr`.
*   `requirements.txt`: `streamlit`, `openai`, `google-generativeai`, `pandas`, `rapidfuzz`, `pdf2image`, `pytesseract`, `PyPDF2`, `pyyaml`, `plotly`.
*   **Secrets**: API keys should be added to Hugging Face Space secrets, which are injected as environment variables.

---

## 6. Implementation Plan & Roadmap

### Phase 1: Core Foundation
1.  Setup Streamlit layout with "Coral" theme injection.
2.  Implement `app.py` routing (Dashboard, Workspace, Settings).
3.  Load Mock Datasets into Pandas DataFrames.

### Phase 2: The Search & Document Engine
1.  Build `search_engine()` with fuzzy logic.
2.  Create `PDFManager` class for handling `pdf2image` and `PyPDF2` operations.
3.  Implement `OCREngine` with both Tesseract and LLM-Vision strategies.

### Phase 3: The Intelligence Layer
1.  Integrate `AgentEngine` class to parse `agents.yaml`.
2.  Connect to OpenAI and Gemini clients.
3.  Implement "Coral Keyword Signalizer" (Regex/LLM hybrid).

### Phase 4: Batch Processing & Refinement
1.  Develop the `BatchProcessor` module for recursive PDF handling.
2.  Implement the `Master_ToC` generator.
3.  Final Polish: Tooltips, Loading spinners, Error boundary handling (e.g., "PDF Encrypted" alerts).

---

## 7. Security and Privacy Considerations

*   **Data Persistence**: No user data is saved to a database. All processing is transient in the Streamlit session runner RAM or Temp directory.
*   **API Security**: Keys are never logged. They are passed strictly in headers to the LLM providers.
*   **OCR Handling**: When using Cloud OCR, document image data is transmitted to 3rd party APIs (OpenAI/Google). The UI must include a disclaimer: *"Cloud OCR transmits data to model provider. Use Local OCR for sensitive/non-public data."*

## 8. Conclusion

The standard FDA review process is document-heavy and disconnected. This Technical Specification proposes a unified "Command Center" that not only centralizes data access but actively augments the human reviewer's capabilities through Agentic AI. By combining "WOW" visualization, flexible document modification, and background batch processing, the system transforms static PDF reading into dynamic, actionable intelligence gathering.
