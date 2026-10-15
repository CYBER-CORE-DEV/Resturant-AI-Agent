# 🍽️ Restaurant AI Agent using LangGraph

An intelligent, multi-step restaurant assistant built using **LangGraph** and Python. This agent is designed to manage customer interactions, handle menu inquiries, process table reservations, and assist with order-taking through structured state management and graph-based workflows.

---

## 🚀 Features

* **Stateful Workflow Management:** Powered by LangGraph to handle complex conversation flows and decision-making steps.
* **Interactive Assistant:** Capable of answering questions about the menu, specials, and restaurant timings.
* **Booking & Ordering System:** Guides users smoothly through making reservations or placing food orders.
* **Environment Configuration:** Integrates securely with API keys using `.env` management.

---

## 🛠️ Tech Stack

* **Python** (3.10+)
* **LangGraph** / LangChain
* **UV** (for fast package management)

---

## 📁 Project Structure

```text
day19/
├── .env                  # Environment variables (API keys, secrets)
├── .gitignore            # Files and directories ignored by Git
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
├── __init__.py           # Package initialization
└── new.py                # Main application script / AI agent logic
```

---

## ⚙️ Setup and Installation

1. **Clone the repository and navigate to the project folder:**
   ```bash
   cd day19
   ```

2. **Set up your environment variables:**
   Create a `.env` file and add your required API keys:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Install dependencies:**
   Using `uv` for fast dependency management:
   ```bash
   uv sync
   ```

---

## ▶️ Usage

Run the main application script to start interacting with the Restaurant AI Agent:

```bash
python new.py
```

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
