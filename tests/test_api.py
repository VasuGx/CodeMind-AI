import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import os

client = TestClient(app)

def test_session_lifecycle():
    # 1. Start Session
    response = client.post("/start-session")
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert session_id is not None

    # 2. Delete Session
    response = client.delete(f"/session/{session_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

def test_load_repo_local():
    # Create a tiny dummy repo for testing
    os.makedirs("test_api_repo/src", exist_ok=True)
    with open("test_api_repo/src/app.py", "w") as f:
        f.write("print('hello')")
        
    response = client.post("/load-repo", json={"local_path": "test_api_repo"})
    if response.status_code != 200:
        print(f"Error Detail: {response.json().get('detail')}")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] is not None
    assert data["files_indexed"] >= 1

def test_invalid_mode():
    session_id = client.post("/start-session").json()["session_id"]
    repo_id = client.post("/load-repo", json={"local_path": "test_api_repo"}).json()["repo_id"]
    
    response = client.post("/run", json={
        "session_id": session_id,
        "repo_id": repo_id,
        "mode": "invalid_mode",
        "query": "test"
    })
    # FastAPI returns 422 for pydantic Literal mismatches automatically
    assert response.status_code == 422

def test_chat_history():
    session_id = client.post("/start-session").json()["session_id"]
    
    # Check empty history
    response = client.get(f"/history/{session_id}")
    assert response.status_code == 200
    assert len(response.json()["messages"]) == 0
