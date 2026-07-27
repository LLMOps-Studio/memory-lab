from unittest.mock import patch

from fastapi.testclient import TestClient

from memory_lab.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "memory-lab"}


def test_chat_missing_message_returns_422():
    response = client.post("/chat", json={"session_id": "s1"})
    assert response.status_code == 422


@patch("memory_lab.api.memory_graph.invoke")
def test_chat_success(mock_invoke):
    # The API only reads .content off the last message, so a lightweight
    # stand-in is enough -- it doesn't need to be a real BaseMessage.
    ai_message = type("Msg", (), {"content": "Nice to meet you!"})()
    mock_invoke.return_value = {"messages": [ai_message]}

    response = client.post(
        "/chat", json={"session_id": "s1", "message": "Hi, I'm Alex."}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["response"] == "Nice to meet you!"
    assert data["session_id"] == "s1"


@patch("memory_lab.api.memory_graph.invoke")
def test_chat_pipeline_failure_returns_500(mock_invoke):
    mock_invoke.side_effect = RuntimeError("ollama connection refused")

    response = client.post("/chat", json={"session_id": "s1", "message": "Hi"})

    assert response.status_code == 500
    assert "ollama connection refused" in response.json()["detail"]


def test_metrics_endpoint_exposes_prometheus_data():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content
