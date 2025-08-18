async def test_queue_add_and_persist(client):
    # first call establishes cookie
    r0 = await client.get("/session")
    client.cookies.update(r0.cookies)
    # add two items
    await client.post("/queue/add", json={"filename": "a.wav"})
    await client.post("/queue/add", json={"filename": "b.wav"})

    r = await client.get("/queue")
    items = r.json()["items"]
    assert [i["filename"] for i in items] == ["a.wav", "b.wav"]

async def test_queue_progress(client):
    r0 = await client.get("/session"); client.cookies.update(r0.cookies)
    await client.patch("/queue/progress", json={"processing": {"pct": 40}})
    r = await client.get("/queue")
    assert r.json()["processing"]["pct"] == 40

async def test_queue_clear(client):
    r0 = await client.get("/session"); client.cookies.update(r0.cookies)
    await client.post("/queue/add", json={"filename":"x.wav"})
    await client.delete("/queue")
    r = await client.get("/queue")
    assert r.json() == {"items": [], "processing": None, "completed": []}
