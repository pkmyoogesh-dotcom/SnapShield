def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["ai_engine"] == "LOCAL" and j["cloud_ai_required"] is False

def test_docs(client):
    assert client.get("/docs").status_code == 200
