"""
CodeMind AI - Conversational UI
A ChatGPT-style interface built with Streamlit for interacting with the 
autonomous multi-agent debugging system.
"""
import streamlit as st
import requests
import json
import time

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="CodeMind AI", layout="wide", page_icon="🧠")

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "repo_id" not in st.session_state:
    st.session_state.repo_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "debug"

def start_session():
    """Initializes a new session on the FastAPI backend."""
    try:
        res = requests.post(f"{BASE_URL}/start-session")
        if res.status_code == 200:
            st.session_state.session_id = res.json()["session_id"]
            return True
    except Exception as e:
        st.error(f"Failed to connect to backend: {e}")
    return False

def load_repo(url_or_path):
    """Triggers repository cloning and indexing."""
    payload = {}
    if url_or_path.startswith("http"):
        payload["repo_url"] = url_or_path
    else:
        payload["local_path"] = url_or_path
    
    try:
        with st.spinner("Cloning and Indexing Repository..."):
            res = requests.post(f"{BASE_URL}/load-repo", json=payload)
            if res.status_code == 200:
                data = res.json()
                st.session_state.repo_id = data["repo_id"]
                st.success(f"Loaded! {data['files_indexed']} files indexed.")
            else:
                st.error(f"Error: {res.json().get('detail')}")
    except Exception as e:
        st.error(f"API Error: {e}")

def run_query(query):
    """Sends a query to the selected pipeline mode (Debug/Explain/Impact)."""
    if not st.session_state.repo_id:
        st.warning("Please load a repository first!")
        return

    payload = {
        "session_id": st.session_state.session_id,
        "repo_id": st.session_state.repo_id,
        "mode": st.session_state.mode,
        "query": query
    }
    
    try:
        res = requests.post(f"{BASE_URL}/run", json=payload)
        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"Execution Error: {res.json().get('detail')}")
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

def upload_external_doc(name, content):
    """Attaches external documentation to the repository's knowledge base."""
    if not st.session_state.repo_id:
        st.warning("Load a repository before attaching documentation.")
        return
    payload = {
        "repo_id": st.session_state.repo_id,
        "name": name,
        "content": content
    }
    try:
        res = requests.post(f"{BASE_URL}/upload-doc", json=payload)
        if res.status_code == 200:
            st.success(f"Attached {name} to knowledge base!")
        else:
            st.error(f"Failed to attach doc: {res.json().get('detail')}")
    except Exception as e:
        st.error(f"API Error: {e}")

def fetch_history():
    """Retrieves session history from the backend to maintain state persistence."""
    if not st.session_state.session_id:
        return
    try:
        res = requests.get(f"{BASE_URL}/history/{st.session_state.session_id}")
        if res.status_code == 200:
            msgs = res.json()["messages"]
            formatted = []
            for m in msgs:
                content = m["content"]
                if m["role"] == "system":
                    try:
                        content = eval(content) 
                    except:
                        pass
                formatted.append({"role": m["role"], "content": content})
            st.session_state.messages = formatted
    except:
        pass

# --- UI LAYOUT (SIDEBAR) ---
with st.sidebar:
    st.title("🧠 CodeMind AI")
    st.markdown("---")
    
    if not st.session_state.session_id:
        if st.button("Initialize Session"):
            start_session()
            st.rerun()
    else:
        st.caption(f"Session: `{st.session_state.session_id[:8]}`")
        if st.button("New Session"):
            requests.delete(f"{BASE_URL}/session/{st.session_state.session_id}")
            st.session_state.session_id = None
            st.session_state.repo_id = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    
    st.subheader("Repository")
    repo_input = st.text_input("GitHub URL or Local Path", placeholder="https://github.com/...")
    if st.button("Load Repo"):
        load_repo(repo_input)

    if st.session_state.repo_id:
        st.caption(f"Active Repo: `{st.session_state.repo_id[:8]}`")
        st.markdown("---")
        st.subheader("Knowledge Base")
        uploaded_file = st.file_uploader("Attach Phase Docs / Styleguides", type=["md", "txt"])
        if uploaded_file is not None:
            doc_content = uploaded_file.read().decode("utf-8")
            if st.button("Index Documentation"):
                upload_external_doc(uploaded_file.name, doc_content)

    st.markdown("---")
    
    st.session_state.mode = st.selectbox(
        "InteractionMode", 
        ["debug", "explain", "impact"],
        index=["debug", "explain", "impact"].index(st.session_state.mode)
    )
    
    st.markdown("---")
    
    st.subheader("Recent Queries")
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.caption(f"Q: {msg['content'][:25]}...")

# --- UI LAYOUT (MAIN CHAT) ---
st.header(f"Mode: {st.session_state.mode.upper()}")

if not st.session_state.session_id:
    start_session()

if st.session_state.session_id and not st.session_state.messages:
    fetch_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        if message["role"] == "user":
            st.markdown(content)
        else:
            if isinstance(content, dict):
                mode = content.get("mode")
                
                if mode == "debug":
                    st.markdown(f"### 🔍 Root Cause\n{content.get('root_cause')}")
                    st.markdown(f"**📂 Suspected Files:** `{', '.join(content.get('suspected_files', []))}`")
                    st.markdown("### 🛠️ Proposed Fix")
                    st.code(content.get("fix"), language="python")
                    st.markdown(f"**📊 Confidence** ({int(content.get('confidence',0)*100)}%)")
                    st.progress(content.get("confidence", 0.0))
                    st.info(f"**🧠 Reasoning:** {content.get('reasoning')}")
                
                elif mode == "explain":
                    st.markdown(f"### 📖 Summary\n{content.get('summary')}")
                    st.markdown("### 🧩 Key Modules")
                    st.write(content.get("key_modules"))
                    st.markdown(f"### 🔄 Data Flow\n{content.get('data_flow')}")
                
                elif mode == "impact":
                    risk_colors = {"low": "green", "medium": "orange", "high": "red"}
                    risk = content.get("risk_level", "low")
                    st.markdown(f"### ⚠️ Impact Analysis")
                    st.markdown(f"**Risk Level:** :{risk_colors.get(risk)}[{risk.upper()}]")
                    st.markdown(f"**Affected Files:** {', '.join(content.get('affected_files', []))}")
                    st.markdown(f"**Dependent Functions:** {', '.join(content.get('dependent_functions', []))}")
                    st.info(f"**Explanation:** {content.get('explanation')}")
                else:
                    st.markdown(str(content))
            else:
                st.markdown(content)

if prompt := st.chat_input("What's the issue?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("CodeMind is thinking..."):
            response = run_query(prompt)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
