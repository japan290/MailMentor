# MailMentor

MailMentor is a Streamlit-based application designed to help you manage and understand your email inbox more effectively. By connecting to your Gmail account, it provides an intelligent dashboard, a searchable inbox powered by a Retrieval-Augmented Generation (RAG) system, and tools to identify actionable items and key insights from your emails.

---

## ✨ Features

- **Secure Authentication**: Log in with your Google account to securely connect to Gmail.  
- **Intuitive Dashboard**: Get a quick overview of your inbox with key metrics, charts, and a list of recent emails.  
- **Smart Email Search**: Use a powerful RAG system to ask natural language questions about your emails.  
- **Actionable Insights**: Automatically identify and categorize important action items.  
- **Real-time Data**: Refresh your inbox to fetch the latest emails.  
- **Data Export**: Download a CSV of all your emails for offline analysis.  

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have the following installed:

- Python **3.8+**
- pip (Python package manager)

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/mailmentor.git
cd mailmentor
```

### 3. Set up Google API Credentials

1. Go to the **Google Cloud Console**.  
2. Create a new project for MailMentor.  
3. Enable the **Gmail API**.  
4. Navigate to **APIs & Services > Credentials**.  
5. Create credentials → **OAuth Client ID** → Choose **Desktop App**.  
6. Download the `client_secret.json` file and place it in the **project root folder** (next to `main.py`).  

### 4. Configure Environment Variables

Create a `.env` file in the root of your project and add:

```env
DATABASE_URL=postgresql://user:password@localhost/your_db_name
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run main.py
```

Then open the link provided (usually `http://localhost:8501/`) in your browser.

---

## 📁 Project Structure

```
MailMentor/
├── app/
│   ├── __init__.py
│   ├── ai_services.py      # AI model interactions (embeddings, LLM calls)
│   ├── auth.py             # User authentication and Google OAuth
│   ├── config.py           # App configuration settings
│   ├── data_handler.py     # Database interactions and data processing
│   ├── email_processor.py  # Gmail API fetching and saving logic
│   ├── models.py           # SQLAlchemy database models (User, Email)
│   ├── rag.py              # Core RAG pipeline logic
│   ├── rag_builder.py      # Script to build the RAG index
│   └── ui.py               # Streamlit UI components
├── main.py                 # Main entry point
├── client_secret.json      # Google OAuth credentials
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (DATABASE_URL, GROQ_API_KEY)
```

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit  
- **Database**: PostgreSQL (SQLAlchemy ORM)  
- **Vector DB**: Milvus  
- **AI Processing**: Groq API (RAG, LLM calls)  
- **Authentication**: Google OAuth2  

---

## 📌 Notes

- Ensure your `DATABASE_URL` points to a valid PostgreSQL instance.  
- You must set your `GROQ_API_KEY` inside `.env`.  
- Do **not** commit your `.env` file or `client_secret.json` to GitHub.  

---

## 📄 License

This project is licensed under the MIT License.
