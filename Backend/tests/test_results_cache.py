async def test_results_cache_roundtrip(client):
    payload = {"transcript": "hello", "confidence": 0.9}
    put = await client.post("/results/whisper/abc123", json=payload)
    assert put.status_code == 200

    get = await client.get("/results/whisper/abc123")
    data = get.json()
    assert data["cached"] is True
    assert data["payload"] == payload
