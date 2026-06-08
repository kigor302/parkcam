#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  plate TEXT NOT NULL,
  confidence REAL,
  image_path TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""

def connect(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn

def emit(obj):
    print(json.dumps(obj, separators=(",", ":")))

def cmd_init(args):
    with connect(args.db) as conn:
        conn.commit()
    emit({"ok": True})

def cmd_add(args):
    with connect(args.db) as conn:
        cur = conn.execute(
            "INSERT INTO events(ts, plate, confidence, image_path) VALUES(?,?,?,?)",
            (args.ts, args.plate or "Unknown", args.confidence, args.image),
        )
        conn.commit()
        emit({"ok": True, "id": cur.lastrowid})

def cmd_list(args):
    clauses = []
    params = []
    if args.start:
        clauses.append("ts >= ?")
        params.append(args.start)
    if args.end:
        clauses.append("ts <= ?")
        params.append(args.end)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    limit = max(1, min(args.limit, 500))
    offset = max(0, args.offset)
    with connect(args.db) as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM events" + where, params).fetchone()["c"]
        rows = conn.execute(
            "SELECT id,ts,plate,confidence,image_path FROM events" + where + " ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    emit({"ok": True, "total": total, "events": [dict(r) for r in rows]})

def cmd_get(args):
    with connect(args.db) as conn:
        row = conn.execute("SELECT id,ts,plate,confidence,image_path FROM events WHERE id=?", (args.id,)).fetchone()
    emit({"ok": bool(row), "event": dict(row) if row else None})

def cmd_update(args):
    with connect(args.db) as conn:
        cur = conn.execute(
            "UPDATE events SET plate=?, confidence=? WHERE id=?",
            (args.plate or "Unknown", args.confidence, args.id),
        )
        conn.commit()
    emit({"ok": cur.rowcount == 1, "updated": cur.rowcount})

def cmd_prune(args):
    cutoff = (datetime.now() - timedelta(days=max(1, args.days))).strftime("%Y-%m-%d %H:%M:%S")
    removed_files = 0
    with connect(args.db) as conn:
        rows = conn.execute("SELECT image_path FROM events WHERE ts < ?", (cutoff,)).fetchall()
        for row in rows:
            try:
                os.remove(row["image_path"])
                removed_files += 1
            except FileNotFoundError:
                pass
            except OSError:
                pass
        cur = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        conn.commit()
    emit({"ok": True, "deleted_events": cur.rowcount, "deleted_files": removed_files, "cutoff": cutoff})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("add")
    s.add_argument("--ts", required=True)
    s.add_argument("--plate", required=True)
    s.add_argument("--confidence", type=float)
    s.add_argument("--image", required=True)
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("list")
    s.add_argument("--start", default="")
    s.add_argument("--end", default="")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--offset", type=int, default=0)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("get")
    s.add_argument("--id", type=int, required=True)
    s.set_defaults(func=cmd_get)

    s = sub.add_parser("update")
    s.add_argument("--id", type=int, required=True)
    s.add_argument("--plate", required=True)
    s.add_argument("--confidence", type=float)
    s.set_defaults(func=cmd_update)

    s = sub.add_parser("prune")
    s.add_argument("--days", type=int, required=True)
    s.set_defaults(func=cmd_prune)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
