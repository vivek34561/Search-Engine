# 🔍 LangChain - Chat with Search

Welcome to **LangChain - Chat with Search**!
This interactive chatbot leverages the power of **LangChain** and integrates tools like **Wikipedia**, **ArXiv**, and **DuckDuckGo** search to offer real-time query handling and contextual information retrieval.

**Screensort** : 
![Screenshot 2025-05-17 000340](https://github.com/user-attachments/assets/d22126b8-b2bc-4da5-af29-4022cbfba532)
![Screenshot 2025-05-17 000359](https://github.com/user-attachments/assets/fdc0b7dd-e153-4f24-a74d-ee4d0a2e9212)


✨ **Live App**: [Streamlit Deployment](https://search-engine-a2gbospht9vvjmvfr3hoqb.streamlit.app/)

---

## 🚀 Features

* 🔎 **Web-Connected Chatbot**: Powered by LangChain tools to search and summarize from external sources like DuckDuckGo, Wikipedia, and ArXiv.
* 🔐 **Groq API Key Support**: Secure access with your Groq API key for enhanced LLM performance.
* 🌐 **Real-Time Query Resolution**: Ask questions on any topic, and the agent will search and respond using available online knowledge.
* 🧠 **LangChain Agents**: Utilizes LangChain’s agent architecture to route queries through appropriate tools automatically.
* 🎯 **Minimal UI**: Built using Streamlit for a fast, responsive, and easy-to-use interface.

---

## 🧩 Tech Stack

| Tool                            | Purpose                            |
| ------------------------------- | ---------------------------------- |
| 🐍 Python                       | Core programming                   |
| 🦜 LangChain                    | LLM and agent orchestration        |
| 🧪 Groq API                     | Fast and efficient model inference |
| 🌍 DuckDuckGo, Wikipedia, ArXiv | Information sources                |
| 📦 Streamlit                    | Web app UI and deployment          |

---

## 🔐 Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/langchain-chat-search.git
   cd langchain-chat-search
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. **Enter your Groq API Key** when prompted in the app UI.

---

## 📁 Folder Structure

```
langchain-chat-search/
├── app.py                # Streamlit app file
├── requirements.txt      # All dependencies
├── README.md             # Project overview (this file)
└── assets/               # (Optional) Icons, screenshots, etc.
```

---


## 📚 References

* [LangChain Documentation](https://docs.langchain.com/)
* [Streamlit Docs](https://docs.streamlit.io/)
* [DuckDuckGo Search API](https://duckduckgo.com/)
