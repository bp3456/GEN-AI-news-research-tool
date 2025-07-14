import streamlit as st
import os
import nltk
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain

# title
st.set_page_config(page_title="Gen AI News Research", layout="wide")

# NLTK setup
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

#Gemini API key
os.environ["GOOGLE_API_KEY"] = st.secrets["api_key"]

# UI
st.title("Gen AI: News Research Tool")
st.sidebar.title("🔗 Add News URLs")

# Session State
st.session_state.setdefault("URLS_INPUT", [])
st.session_state.setdefault("check", False)
st.session_state.setdefault("vectorindex_openai", None)
st.session_state.setdefault("docs_map", {})
st.session_state.setdefault("chat_history", [])  

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# Sidebar URL Input
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}")
    if url and url not in st.session_state.URLS_INPUT:
        st.session_state.URLS_INPUT.append(url)

#  Clearing  URLs data
if st.sidebar.button("🔄 Clear URLs data"):
    st.session_state.URLS_INPUT.clear()
    st.session_state.check = False
    st.session_state.vectorindex_openai = None
    st.session_state.docs_map = {}
    st.success("✅ Cleared all stored URLs data.")

# Process URLs
if st.sidebar.button("✅ Process URLs"):
    if not st.session_state.URLS_INPUT:
        st.warning("⚠️ Please enter at least one valid news article URL.")
    else:
        loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
        with st.spinner("📄 Loading and processing articles..."):
            data = loader.load()

        if not data:
            st.error("❌ Failed to load content from the given URLs.")
        else:
            splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","], chunk_size=1000)
            docs = splitter.split_documents(data)

            for doc, meta in zip(docs, data):
                doc.metadata["source"] = meta.metadata.get("source", "")

            url_doc_map = {}
            for doc in docs:
                url = doc.metadata.get("source", "")
                url_doc_map.setdefault(url, []).append(doc)

            st.session_state.docs_map = url_doc_map

            all_docs = [doc for sublist in url_doc_map.values() for doc in sublist]
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vectorindex_openai = FAISS.from_documents(all_docs, embeddings)
            st.session_state.vectorindex_openai.save_local("faiss_index")

            st.session_state.check = True
            st.success("✅ Articles processed successfully!")

# Main Interaction
if st.session_state.check:
    urls = list(st.session_state.docs_map.keys())
    selected_urls = st.multiselect("📚 Choose articles to analyze:", urls)

    if selected_urls:
        combined_docs = []
        for url in selected_urls:
            combined_docs.extend(st.session_state.docs_map[url])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(combined_docs, embeddings)

        query = st.text_input("💬 Ask a question or request a summary:")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            suggest = st.button("💡 Suggest Questions")
        with col2:
            clear_chat = st.button("🗑️ Clear Chat History")
        with col3:
            compare_button = st.button("📊 Compare Articles")

        if suggest:
            full_text = "\n".join([doc.page_content for doc in combined_docs])
            prompt = PromptTemplate(
                input_variables=["content"],
                template="Based on the following content, generate 5 insightful questions:\n\n{content}"
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            with st.spinner("🧠 Thinking..."):
                suggestions = chain.run({"content": full_text})
                st.markdown(suggestions)

        # compare button
        if compare_button:
            if len(selected_urls) < 2:
                st.warning("Please select at least two articles to compare.")
            else:
                all_texts = ["\n".join([doc.page_content for doc in st.session_state.docs_map[url]]) for url in selected_urls]
                compare_prompt = f"Compare the following articles:\n\nARTICLE 1:\n{all_texts[0]}\n\nARTICLE 2:\n{all_texts[1]}\n\nProvide a useful insight-based comparison:"
                with st.spinner("🔍 Comparing articles..."):
                    compare_result = llm.invoke(compare_prompt).content
                st.session_state.chat_history.append(("Compare Articles", compare_result, selected_urls))
                st.markdown("### 📊 Comparison Result")

                article_1_lines = []
                article_2_lines = []
                for line in compare_result.split("\n"):
                    if ':' in line:
                        parts = line.split(":", 1)
                        article_1_lines.append(parts[0].strip())
                        article_2_lines.append(parts[1].strip())
                    else:
                        article_1_lines.append(line.strip())
                        article_2_lines.append("")

                comparison_table = "<table style='width:100%;border-collapse:collapse;border:1px solid #ccc;'>"
                comparison_table += "<tr><th style='border:1px solid #ccc;padding:8px;'>Article 1</th><th style='border:1px solid #ccc;padding:8px;'>Article 2</th></tr>"
                for left, right in zip(article_1_lines, article_2_lines):
                    comparison_table += f"<tr><td style='border:1px solid #ccc;padding:8px;'>{left}</td><td style='border:1px solid #ccc;padding:8px;'>{right}</td></tr>"
                comparison_table += "</table>"
                st.markdown(comparison_table, unsafe_allow_html=True)

        # Answer query
        if query:
            lower_query = query.lower().strip()
            if "compare" in lower_query and len(selected_urls) >= 2:
                all_texts = ["\n".join([doc.page_content for doc in st.session_state.docs_map[url]]) for url in selected_urls]
                compare_prompt = f"Compare the following articles:\n\nARTICLE 1:\n{all_texts[0]}\n\nARTICLE 2:\n{all_texts[1]}\n\n{query}"
                with st.spinner("🔍 Comparing articles..."):
                    answer = llm.invoke(compare_prompt).content
                st.session_state.chat_history.append((query, answer, selected_urls))
            elif any(word in lower_query for word in ["summarize", "summary", "summarise"]):
                full_text = "\n".join([doc.page_content for doc in combined_docs])
                prompt = PromptTemplate(
                    input_variables=["content"],
                    template="Summarize the following content:\n\n{content}\n\nSummary:"
                )
                summarize_chain = LLMChain(llm=llm, prompt=prompt)
                with st.spinner("📋 Summarizing..."):
                    answer = summarize_chain.run({"content": full_text})
                    st.session_state.chat_history.append((query, answer, selected_urls))
            else:
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever()
                )
                with st.spinner("🔍 Searching..."):
                    response = chain({"question": query}, return_only_outputs=True)
                answer = response['answer']
                sources = response['sources'].split(', ') if response['sources'] else selected_urls
                st.session_state.chat_history.append((query, answer, sources))

# Live Chat History Display
if st.session_state.chat_history:
    st.markdown("### 📟 Chat History")
    for idx, (q, a, sources) in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"**🗨️ Q{len(st.session_state.chat_history)-idx}:** {q}")
            st.markdown(f"**✅ A{len(st.session_state.chat_history)-idx}:** {a}")
            if sources:
                st.markdown("**🔗 Sources:**")
                for src in sources:
                    st.markdown(f"- [{src}]({src})")
else:
    st.info("")

# Clear Chat History
if 'clear_chat' in locals() and clear_chat:
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
