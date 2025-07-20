# 🏀 AI-Powered NBA Scout

An advanced, conversational AI platform providing deep, precise analysis of NBA players and teams. This project leverages a sophisticated hybrid AI architecture to deliver professional-grade scouting reports from simple, natural language questions.

---

## Key Features

* **💬 Natural Language Interface:** Ask complex questions in plain English. The AI understands context and delivers nuanced answers.
* **🧮 Precise Quantitative Analysis:** Performs on-the-fly calculations for averages, totals, career highs, and seasonal progressions across a wide range of stats.
* **📚 Deep Contextual Insights:** Analyzes data from every regular season and playoff game for every active player, providing rich, context-aware answers.
* **🎨 Automated Professional Formatting:** Responses are delivered in clean Markdown, including tables and lists, perfect for direct use in reports.
* **🔄 Resumable Data Pipeline:** The robust data ingestion engine is resumable, capable of handling network errors and building a massive database over time.
* **🧠 Hybrid AI Architecture:** Combines Retrieval-Augmented Generation (RAG) with LLM Function Calling for the best of both worlds: narrative insight and mathematical precision.

---

## 🤖 How It Works: A Hybrid AI Architecture

The scout's intelligence comes from two powerful techniques working together under the direction of an LLM Agent (**OpenAI's GPT-4o-mini**).

### 1. Retrieval-Augmented Generation (RAG)

For qualitative questions like "What is a player's style?", the system searches a specialized **Vector Database** (`ChromaDB`) containing thousands of text summaries. It retrieves the most relevant context and feeds it to the LLM to generate a rich, narrative answer.

### 2. LLM Function Calling

For quantitative questions like "What was Kobe's scoring average?", the LLM acts as an intelligent agent. It selects the right Python function (a "tool") to perform the exact calculation from a structured database. It then translates the numerical result into a human-readable sentence.

---

## 🛠️ Technology Stack

* **Backend:** Python & FastAPI
* **AI Engine:** **OpenAI (GPT-4o-mini)**
* **Data Analysis:** Pandas
* **Data Scraping:** nba_api
* **Vector Search:** SentenceTransformers & ChromaDB
* **Frontend:** HTML, CSS (Tailwind), JS
* **Rendering:** Marked.js

---

## 🚀 Setup & Usage

Get your own instance running in three steps.

### Step 1: Setup

Clone the repository, install dependencies from `requirements.txt` (which now includes the `openai` library), and add your OpenAI API key to a `.env` file.

```bash
pip install -r requirements.txt
```

### Step 2: Get Your API Key

1.  Create an account on the [OpenAI Platform](https://platform.openai.com/).
2.  Set up billing and navigate to the **API Keys** section.
3.  Create a new secret key and copy it.
4.  Create a file named `.env` in the root of the project folder and add your key:
    ```
    OPENAI_API_KEY="sk-YourSecretKeyHere"
    ```

### Step 3: Data Ingestion

Run the ingestion script. This is a one-time, resumable process that builds the local databases (`chroma_db` and `all_games.csv`).

```bash
python data_ingestion.py
```

### Step 4: Run the App

Once ingestion is complete, start the FastAPI server and navigate to the local URL.

```bash
uvicorn main:app --reload
```

You can access the scout at **http://127.0.0.1:8000**.

---

## 💡 Example Queries

* How has Joel Embiid progressed in the offensive end during his time at the NBA?
* What are certain weaknesses to James Harden's game?
* How good a defender is Jrue Holiday?
* Who averages the most points against LAL?
* Taking only assists and turnovers into consideration who is better Trae Young or Luka Doncic?
* How did Kawhi Leonard perform in the season he spent at TOR?
