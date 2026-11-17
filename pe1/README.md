# CivicFix AI — Smart Public Grievance Guidance System

CivicFix AI is a web application designed to help citizens understand how to formally report public problems, receive verified government guidelines through Retrieval-Augmented Generation (RAG), identify the responsible municipal department, and generate official complaint letters for instant PDF download.

---

## 1. Focused Citizen Architecture

The application is designed specifically for citizens with a clean, modern, and intuitive Single Page Interface featuring 4 core services:

1. **AI Guidance**:
   - Problem identification via keyword matching with category selection fallback.
   - Ambiguity detection prompting the citizen for more details if the description is too brief.
   - Responsible department recommendation with official municipal jurisdiction disclaimer.
   - Retrieval of top-3 relevant government guideline sections using FAISS vector search and embeddings.
   - Step-by-step complaint procedures and required evidence checklist.
   - Citation of source document names and exact page numbers.
   - Clear indicators when verified official guidelines are unavailable.
2. **Report Problem**:
   - Select from 5 strict civic categories (*Road Damage, Drainage Overflow, Garbage Accumulation, Streetlight Failure, Water Leakage*).
   - Enter problem description and location/landmark.
   - Optional evidence photo upload.
   - Immediate reference tracking number assignment (`CF-2026-XXXX`) and department routing.
3. **My Complaints**:
   - View submitted grievances with date and live status (`Pending`, `In Progress`, `Resolved`, `Rejected`).
   - Modal inspection of grievance details and photographic proof.
   - Delete complaints.
4. **Complaint Letter**:
   - Auto-generated formal grievance letter addressing the responsible department.
   - Real-time in-browser letter editor.
   - Instant download as an official PDF generated via ReportLab.

---

## 2. Supported Categories & Department Mapping

| Category | Responsible Department | Administrative Jurisdiction Scope |
| :--- | :--- | :--- |
| **Road Damage** | Road Maintenance Authority | Municipal roads, pavements, bituminous layers, potholes, paver blocks. |
| **Drainage Overflow** | Drainage / Sanitation Department | Storm-water drains, primary & secondary sewer lines, manhole clearing. |
| **Garbage Accumulation** | Sanitation Department | Solid waste management, community dustbins, street cleaning. |
| **Streetlight Failure** | Electrical Maintenance Department | Public street lamps, LED fixtures, timer switches, light poles. |
| **Water Leakage** | Water Supply Department | Potable water mains distribution, pipeline ruptures, public taps. |

> **Official Jurisdiction Disclaimer:** *Department recommendations are rule-based and subject to verification by your local municipal corporation or ward authority jurisdiction.*

---

## 3. Technology Stack

- **Frontend**: HTML5, Vanilla JavaScript (ES6+), CSS3, Bootstrap 5.3, Bootstrap Icons.
- **Backend**: Python 3.10+, FastAPI, Uvicorn.
- **Database**: SQLite with SQLAlchemy ORM locally; PostgreSQL for persistent Vercel deployments.
- **AI & RAG**:
  - `pymupdf` (PDF text extraction)
  - Gemini `gemini-embedding-001` (hosted 768-dimensional embeddings with local 768-d fallback)
  - `faiss-cpu` (cosine similarity vector index)
- **PDF Generation**: `reportlab` (programmatic vector PDF compiler).

---

## 4. How to Run the Application

### Step 1: Start the Web Server
```bash
python run.py
```

### Step 2: Open in Your Browser
- **CivicFix AI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 5. Automated Verification
To run the automated verification test suites:
```bash
python run_suite.py
python -m unittest tests/test_civicfix.py -v
```
