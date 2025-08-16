async def test_queue_add_and_persist(client):
    # first call establishes cookie
    r0 = await client.get("/session")
    cookies = r0.cookies
    # add two items
    await client.post("/queue/add", json={"filename": "a.wav"}, cookies=cookies)
    await client.post("/queue/add", json={"filename": "b.wav"}, cookies=cookies)

    r = await client.get("/queue", cookies=cookies)
    items = r.json()["items"]
    assert [i["filename"] for i in items] == ["a.wav", "b.wav"]

async def test_queue_progress(client):
    r0 = await client.get("/session"); cookies = r0.cookies
    await client.patch("/queue/progress", json={"processing": {"pct": 40}}, cookies=cookies)
    r = await client.get("/queue", cookies=cookies)
    assert r.json()["processing"]["pct"] == 40

async def test_queue_clear(client):
    r0 = await client.get("/session"); cookies = r0.cookies
    await client.post("/queue/add", json={"filename":"x.wav"}, cookies=cookies)
    await client.delete("/queue", cookies=cookies)
    r = await client.get("/queue", cookies=cookies)
    assert r.json() == {"items": [], "processing": None, "completed": []}
