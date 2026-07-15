from fastapi import FastAPI
from app.routers.meetings import router as meetings_router
from app.routers.projects import router as projects_router
from app.routers.chat import router as chat_router
from app.db.database import init_db

app = FastAPI()


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok"}


app.include_router(projects_router)
app.include_router(meetings_router)
app.include_router(chat_router)