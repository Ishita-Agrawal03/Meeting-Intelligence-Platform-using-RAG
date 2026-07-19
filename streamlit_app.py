"""
Streamlit UI for the AI Meeting Intelligence Platform.

Layout mirrors Claude/ChatGPT: a sidebar lists projects (like a chat
history list); the main area shows either a "start something new"
screen, or, once a project is selected, that project's full workspace
(Chat, Tasks, Decisions, Participants) scoped ONLY to that project's
meetings — no per-tab dropdowns, since the sidebar IS the navigation.

This is a SEPARATE process from your FastAPI backend — it doesn't
import any of your app code, just calls the API over HTTP.

Run:
    (in one terminal) uvicorn app.main:app --reload
    (in a second terminal) python -m streamlit run streamlit_app.py
"""
import requests
import streamlit as st

import os
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Meeting Intelligence Platform", layout="wide")

if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}  # {project_id: [{"role":..., "content":...}, ...]}


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


def api_post(path, json_body=None, files=None):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, files=files)
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


def api_delete(path):
    try:
        resp = requests.delete(f"{API_BASE_URL}{path}")
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


# ==================== SIDEBAR: project navigation ====================
with st.sidebar:
    st.header("🧠 Projects")

    projects = api_get("/projects/") or []

    for p in projects:
        is_active = p["id"] == st.session_state.active_project_id
        label = f"{'🟢 ' if is_active else ''}{p['name']} ({p['meeting_count']})"

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"proj_{p['id']}", use_container_width=True):
                st.session_state.active_project_id = p["id"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{p['id']}", help=f"Delete '{p['name']}'"):
                st.session_state.confirm_delete_id = p["id"]
                st.session_state.confirm_delete_name = p["name"]
                st.rerun()

    # Confirmation step — deleting a project also deletes every meeting,
    # chunk, task, decision, and participant inside it (cascade). This
    # is permanent, so a stray click on the trash icon shouldn't be
    # enough on its own to actually delete anything.
    if st.session_state.get("confirm_delete_id") is not None:
        st.warning(f"Delete **{st.session_state.confirm_delete_name}** and everything in it? This can't be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", type="primary", use_container_width=True):
                api_delete(f"/projects/{st.session_state.confirm_delete_id}")
                if st.session_state.active_project_id == st.session_state.confirm_delete_id:
                    st.session_state.active_project_id = None
                st.session_state.confirm_delete_id = None
                st.session_state.confirm_delete_name = None
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_delete_id = None
                st.session_state.confirm_delete_name = None
                st.rerun()

    st.divider()
    if st.button("➕ New project", use_container_width=True):
        st.session_state.active_project_id = None
        st.rerun()


# ==================== MAIN AREA ====================

# ---- No project selected: "start something new" screen ----
if st.session_state.active_project_id is None:
    st.title("🧠 AI Meeting Intelligence Platform")
    st.subheader("Start a new project")
    st.caption(
        "A project holds every meeting you upload to it — ask questions across "
        "all of them together, like picking up a chat thread and adding to it."
    )

    new_project_name = st.text_input("Project name", placeholder="e.g. Smart Customer Support AI Platform")
    new_file = st.file_uploader(
        "First meeting file (.txt, .pdf, .docx, .mp3, .wav, .mp4, .m4a)",
        type=["txt", "pdf", "docx", "mp3", "wav", "mp4", "m4a", "webm", "mpga", "mpeg"],
    )

    if st.button("Create project & upload", type="primary"):
        if not new_project_name.strip():
            st.warning("Project name is required.")
        elif not new_file:
            st.warning("Please choose a file to upload.")
        else:
            with st.spinner("Creating project..."):
                proj_result = api_post("/projects/", json_body={"name": new_project_name})
            if proj_result:
                project_id = proj_result["id"]
                with st.spinner("Uploading and processing..."):
                    files = {"file": (new_file.name, new_file.getvalue())}
                    upload_result = api_post(f"/projects/{project_id}/upload", files=files)
                if upload_result:
                    if upload_result.get("status") == "transcribing":
                        st.info("🎙️ Audio/video uploaded — transcribing in the background.")
                    else:
                        st.success(f"Uploaded — {upload_result.get('chunks_created', 0)} chunk(s) created.")
                    st.session_state.active_project_id = project_id
                    st.rerun()

# ---- A project is active: full scoped workspace ----
else:
    project_id = st.session_state.active_project_id
    project_meta = next((p for p in projects if p["id"] == project_id), None)
    project_name = project_meta["name"] if project_meta else f"Project {project_id}"

    st.title(f"🧠 {project_name}")

    with st.expander("➕ Upload another meeting to this project"):
        more_file = st.file_uploader(
            "File (.txt, .pdf, .docx, .mp3, .wav, .mp4, .m4a)",
            type=["txt", "pdf", "docx", "mp3", "wav", "mp4", "m4a", "webm", "mpga", "mpeg"],
            key="more_file_uploader",
        )
        if more_file and st.button("Upload"):
            with st.spinner("Uploading and processing..."):
                files = {"file": (more_file.name, more_file.getvalue())}
                result = api_post(f"/projects/{project_id}/upload", files=files)
            if result:
                if result.get("status") == "transcribing":
                    st.info("🎙️ Audio/video uploaded — transcribing in the background.")
                else:
                    st.success(f"Uploaded — {result.get('chunks_created', 0)} chunk(s) created.")
                st.rerun()

    tab_chat, tab_tasks, tab_decisions, tab_participants = st.tabs(
        ["💬 Chat", "✅ Tasks", "📌 Decisions", "👥 Participants"]
    )

    # ---- Chat tab ----
    with tab_chat:
        history = st.session_state.chat_histories.setdefault(project_id, [])

        for turn in history:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

        query = st.chat_input("Ask something about this project's meetings...")

        if query:
            history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            history_payload = [{"role": t["role"], "content": t["content"]} for t in history[:-1]]

            def _token_stream():
                payload = {"query": query, "chat_history": history_payload, "project_id": project_id}
                try:
                    with requests.post(
                        f"{API_BASE_URL}/chat/stream", json=payload, stream=True, timeout=60,
                    ) as resp:
                        resp.raise_for_status()
                        for piece in resp.iter_content(chunk_size=None, decode_unicode=True):
                            if piece:
                                yield piece
                except requests.exceptions.ConnectionError:
                    yield "Can't reach the backend. Is `uvicorn app.main:app --reload` running?"
                except Exception as e:
                    yield f"Request failed: {e}"

            with st.chat_message("assistant"):
                full_answer = st.write_stream(_token_stream())

            history.append({"role": "assistant", "content": full_answer})

    # ---- Tasks tab ----
    with tab_tasks:
        owner_filter = st.text_input("Filter by owner (optional)", key="task_owner_filter", placeholder="e.g. Rahul")
        params = f"?owner={owner_filter}" if owner_filter.strip() else ""
        tasks = api_get(f"/projects/{project_id}/tasks{params}") or []

        if not tasks:
            st.info("No tasks found for this project yet.")
        else:
            for t in tasks:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{t['task']}**")
                        st.caption(f"Owner: {t.get('owner') or 'Unassigned'}")
                    with col2:
                        if t.get("deadline"):
                            st.caption(f"📅 {t['deadline']}")
                        st.caption(f"Status: `{t.get('status', 'pending')}`")

    # ---- Decisions tab ----
    with tab_decisions:
        decisions = api_get(f"/projects/{project_id}/decisions") or []

        if not decisions:
            st.info("No decisions found for this project yet.")
        else:
            for d in decisions:
                st.markdown(f"- {d['decision']}")

    # ---- Participants tab ----
    with tab_participants:
        participants = api_get(f"/projects/{project_id}/participants") or []

        if not participants:
            st.info("No participants recorded for this project yet.")
        else:
            cols = st.columns(4)
            for i, p in enumerate(participants):
                with cols[i % 4]:
                    st.markdown(f"👤 **{p['person_name']}**")