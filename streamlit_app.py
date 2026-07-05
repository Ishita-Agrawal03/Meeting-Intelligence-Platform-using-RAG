"""
Streamlit UI for the AI Meeting Intelligence Platform.

This is a SEPARATE process from your FastAPI backend — it doesn't
import any of your app code. It just calls your API over HTTP, the
same way curl or /docs does. That means:
  - Your FastAPI server (uvicorn) must ALREADY be running.
  - This file lives at the project root, not inside app/, because
    it's a frontend, not part of the backend package.

Run:
    (in one terminal) uvicorn app.main:app --reload
    (in a second terminal) streamlit run streamlit_app.py
"""
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Meeting Intelligence Platform", layout="wide")
st.title("🧠 AI Meeting Intelligence Platform")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}


def api_get(path):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}")
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the backend. Is `uvicorn app.main:app --reload` running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def api_post(path, json_body=None, files=None, data=None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the backend. Is `uvicorn app.main:app --reload` running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# ---------------- Sidebar: create meeting + upload ----------------
with st.sidebar:
    st.header("📁 Meetings")

    with st.expander("➕ Create a new meeting", expanded=False):
        new_title = st.text_input("Title", key="new_title")
        new_project = st.text_input("Project", key="new_project")
        if st.button("Create meeting"):
            if new_title.strip():
                result = api_post("/meetings/", json_body={"title": new_title, "project": new_project})
                if result:
                    st.success(f"Created meeting id={result.get('id')}")
                    st.rerun()
            else:
                st.warning("Title is required.")

    meetings = api_get("/meetings/") or []

    st.subheader("Upload a transcript")
    if meetings:
        meeting_options = {f"{m['id']} — {m['title']}": m["id"] for m in meetings}
        selected_label = st.selectbox("Choose a meeting", list(meeting_options.keys()))
        selected_id = meeting_options[selected_label]

        uploaded_file = st.file_uploader("Transcript file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if uploaded_file and st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            result = api_post(f"/meetings/{selected_id}/upload", files=files)
            if result:
                st.success(
                    f"Uploaded — {result.get('chunks_created', 0)} chunk(s) created, "
                    f"detected as '{result.get('source_type')}'"
                )
    else:
        st.info("No meetings yet — create one above first.")

    st.divider()
    st.subheader("All meetings")
    for m in meetings:
        st.caption(f"**{m['title']}** (id={m['id']}) — status: `{m.get('status')}`")


# ---------------- Main area: chat ----------------
st.header("💬 Ask about your meetings")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

query = st.chat_input("Ask something, e.g. 'Why did we choose PostgreSQL?'")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # send recent history (excluding the message we just added, which is `query` itself)
    history_payload = [
        {"role": t["role"], "content": t["content"]}
        for t in st.session_state.chat_history[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching meetings..."):
            result = api_post("/chat", json_body={"query": query, "chat_history": history_payload})

        if result:
            st.write(result["answer"])

            if result.get("citations"):
                with st.expander(f"📎 Sources ({len(result['citations'])})"):
                    for c in result["citations"]:
                        st.markdown(f"**{c['meeting_title']}** (chunk {c['chunk_id']})")
                        st.caption(c["source_text"])

            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})