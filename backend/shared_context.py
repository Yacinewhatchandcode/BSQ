from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import uvicorn
from datetime import datetime

app = FastAPI(title="Shared Context & Logs Service")
DB_PATH = "shared_context.db"

class LogEntry(BaseModel):
    agent: str
    event: str
    details: Optional[str] = None
    timestamp: Optional[str] = None

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent TEXT,
        event TEXT,
        details TEXT,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.post("/log")
def add_log(entry: LogEntry):
    ts = entry.timestamp or datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO logs (agent, event, details, timestamp) VALUES (?, ?, ?, ?)",
        (entry.agent, entry.event, entry.details, ts)
    )
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.get("/logs", response_model=List[LogEntry])
def get_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT agent, event, details, timestamp FROM logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [LogEntry(agent=row[0], event=row[1], details=row[2], timestamp=row[3]) for row in rows]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
