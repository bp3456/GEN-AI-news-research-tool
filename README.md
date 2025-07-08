# 🧠 Gen AI News Research Tool

A powerful Streamlit-based web app that lets you analyze, compare, and ask questions about multiple online news articles using Google Gemini (LLM), LangChain, FAISS, and document embeddings.

---

## 🚀 Features

- 🔗 Accepts and processes multiple **news article URLs**
- 📄 Loads and splits article content into meaningful chunks
- 🧠 Uses **Google Gemini** via LangChain to:
  - Answer user questions about the news content
  - Summarize articles
  - Suggest smart follow-up questions
  - Compare two articles side-by-side in a tabular view
- 📊 Uses **FAISS vector database** to semantically search across articles
- 🔁 Maintains **chat history** with article sources
- 💻 Fully interactive Streamlit web UI

---

## 🏗️ Tech Stack

| Tool/Library              | Purpose                                        |
|--------------------------|------------------------------------------------|
| **Streamlit**            | Web interface                                  |
| **LangChain**            | LLM orchestration and prompt management        |
| **Google Gemini API**    | Language model for Q&A, summary, comparison    |
| **FAISS**                | Vector store for document embeddings           |
| **NLTK**                 | Tokenization and text preprocessing            |
| **Unstructured**         | Loads text from URLs                           |

---

## 📦 Requirements

See [`requirements.txt`](./requirements.txt) for full package list:

txt
streamlit
nltk
langchain
faiss-cpu
unstructured
langchain-google-genai==2.1.5
langchain-community
libmagic
google-generativeai
Install them with:

bash
Copy
Edit
pip install -r requirements.txt
🛠️ Local Setup
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/genai-news-research.git
cd genai-news-research
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set your API keys

Create a file .streamlit/secrets.toml with your keys:

toml
Copy
Edit
api_key = "your_gemini_api_key"
newsapi_key = "your_newsapi_key"
🔐 Don't commit real keys — use .gitignore to protect your secrets.

▶️ Run the App
bash
Copy
Edit
streamlit run main.py

📁 File Structure
bash
Copy
Edit
genai-news-research/
├── main.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets_template.toml  


📸 Screenshot

![Screenshot 2025-07-08 113959](https://github.com/user-attachments/assets/445cef15-9288-48dd-8817-c18be7188e99)

📜 License
MIT License — free for personal or academic use.

🙌 Credits
Built with ❤️ using:

Google Generative AI

LangChain

Streamlit

FAISS
