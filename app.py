import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv()

# Streamlit App Title
st.title("🔎 LangChain - Chat with Search")

"""
This chatbot uses LangChain tools for Wikipedia, ArXiv, and DuckDuckGo search.
Try more LangChain 🤝 Streamlit Agent examples at 
[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Ensure API key is provided before proceeding
if not api_key:
    st.warning("Please enter a valid Groq API Key to proceed.")
    st.stop()

# Initialize Search Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    # Initialize LangChain Agent
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Process user query with the agent
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        try:
            # Ensure the agent receives a text input, not a list
            query_text = prompt  # Only pass the latest user input
            response = search_agent.run(query_text, callbacks=[st_cb])
            
            # Store response in chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
