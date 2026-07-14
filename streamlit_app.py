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
    (in a second terminal) python -m streamlit run streamlit_app.py
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
        new_participants = st.text_input(
            "Participants (comma-separated, optional)",
            key="new_participants",
            placeholder="e.g. Rahul, Ishita",
            help="For audio/video uploads, Whisper transcription has no speaker "
                 "labels, so participants won't auto-detect — add them here manually.",
        )
        if st.button("Create meeting"):
            if new_title.strip():
                participant_list = [p.strip() for p in new_participants.split(",") if p.strip()]
                result = api_post("/meetings/", json_body={
                    "title": new_title,
                    "project": new_project,
                    "participants": participant_list,
                })
                if result:
                    st.success(f"Created meeting id={result.get('id')}")
                    st.session_state["just_created_meeting_id"] = result.get("id")
                    st.rerun()
            else:
                st.warning("Title is required.")

    meetings = api_get("/meetings/") or []

    st.subheader("Upload a transcript")
    if meetings:
        meeting_options = {f"{m['id']} — {m['title']}": m["id"] for m in meetings}
        labels = list(meeting_options.keys())

        # If a meeting was just created, default the dropdown to it —
        # otherwise a file can silently get uploaded to whatever meeting
        # was previously selected instead of the new one.
        default_index = 0
        just_created_id = st.session_state.get("just_created_meeting_id")
        if just_created_id is not None:
            for i, label in enumerate(labels):
                if meeting_options[label] == just_created_id:
                    default_index = i
                    break

        selected_label = st.selectbox("Choose a meeting", labels, index=default_index)
        selected_id = meeting_options[selected_label]

        uploaded_file = st.file_uploader(
            "Transcript, notes, or audio/video (.txt, .pdf, .docx, .mp3, .wav, .mp4, .m4a)",
            type=["txt", "pdf", "docx", "mp3", "wav", "mp4", "m4a", "webm", "mpga", "mpeg"],
        )
        if uploaded_file and st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            result = api_post(f"/meetings/{selected_id}/upload", files=files)
            if result:
                if result.get("status") == "transcribing":
                    st.info(
                        "🎙️ Audio/video uploaded — transcribing in the background. "
                        "Check the meeting's status below; it'll move to `ready` once done "
                        "(refresh or ask a question to see updated status)."
                    )
                else:
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


# ---------------- Main area ----------------
meetings_by_id = {m["id"]: m for m in meetings}

tab_chat, tab_tasks, tab_decisions, tab_participants = st.tabs(
    ["💬 Chat", "✅ Tasks", "📌 Decisions", "👥 Participants"]
)

# ---------------- Tab: Chat ----------------
with tab_chat:
    st.header("Ask about your meetings")

    filter_options = {"All meetings (search everything)": None}
    filter_options.update({f"{m['id']} — {m['title']}": m["id"] for m in meetings})
    filter_label = st.selectbox(
        "Search scope", list(filter_options.keys()), key="chat_meeting_filter",
        help="Restrict this question to one meeting's content — useful for confirming "
             "a specific upload actually indexed, instead of always searching everything.",
    )
    filter_meeting_id = filter_options[filter_label]

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

        def _token_stream():
            """Yields plain-text pieces as they arrive from the backend,
            for st.write_stream to render incrementally instead of
            waiting for the full answer."""
            payload = {"query": query, "chat_history": history_payload}
            if filter_meeting_id is not None:
                payload["meeting_id"] = filter_meeting_id
            try:
                with requests.post(
                    f"{API_BASE_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=60,
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

        st.session_state.chat_history.append({"role": "assistant", "content": full_answer})

# ---------------- Tab: Tasks ----------------
with tab_tasks:
    st.header("Action items across all meetings")
    st.caption("Extracted automatically at upload time — no search involved, this is a direct database lookup.")

    owner_filter = st.text_input("Filter by owner (optional)", key="task_owner_filter", placeholder="e.g. Rahul")

    if st.button("Refresh tasks"):
        st.rerun()

    params = f"?owner={owner_filter}" if owner_filter.strip() else ""
    tasks = api_get(f"/meetings/tasks/all{params}") or []

    if not tasks:
        st.info(
            "No tasks found"
            + (f" for owner '{owner_filter}'." if owner_filter.strip() else " yet. Upload a meeting to extract some.")
        )
    else:
        for t in tasks:
            meeting_title = meetings_by_id.get(t["meeting_id"], {}).get("title", f"Meeting {t['meeting_id']}")
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{t['task']}**")
                    st.caption(f"Owner: {t.get('owner') or 'Unassigned'} · From: {meeting_title}")
                with col2:
                    if t.get("deadline"):
                        st.caption(f"📅 {t['deadline']}")
                    st.caption(f"Status: `{t.get('status', 'pending')}`")

# ---------------- Tab: Decisions ----------------
with tab_decisions:
    st.header("Decisions across all meetings")
    st.caption("Extracted automatically at upload time — no search involved, this is a direct database lookup.")

    if st.button("Refresh decisions"):
        st.rerun()

    decisions = api_get("/meetings/decisions/all") or []

    if not decisions:
        st.info("No decisions found yet. Upload a meeting to extract some.")
    else:
        # group by meeting so decisions from the same meeting sit together
        by_meeting = {}
        for d in decisions:
            by_meeting.setdefault(d["meeting_id"], []).append(d)

        for meeting_id, meeting_decisions in by_meeting.items():
            meeting_title = meetings_by_id.get(meeting_id, {}).get("title", f"Meeting {meeting_id}")
            st.subheader(meeting_title)
            for d in meeting_decisions:
                st.markdown(f"- {d['decision']}")
            st.divider()

# ---------------- Tab: Participants ----------------
with tab_participants:
    st.header("Participants by meeting")

    if not meetings:
        st.info("No meetings yet — create one in the sidebar first.")
    else:
        any_participants = False
        for m in meetings:
            participants = api_get(f"/meetings/{m['id']}/participants") or []
            if not participants:
                continue
            any_participants = True
            st.subheader(m["title"])
            cols = st.columns(4)
            for i, p in enumerate(participants):
                with cols[i % 4]:
                    st.markdown(f"👤 **{p['person_name']}**")
            st.divider()

        if not any_participants:
            st.info("No participants recorded yet across any meeting.")