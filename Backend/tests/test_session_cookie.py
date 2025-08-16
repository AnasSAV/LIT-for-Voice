async def test_sets_cookie_and_returns_sid(client):
    r = await client.get("/session")
    assert r.status_code == 200
    sid = r.json()["sid"]
    # cookie present
    assert any(c.name == "sid" for c in r.cookies.jar)
    # same sid on next call
    r2 = await client.get("/session", cookies=r.cookies)
    assert r2.json()["sid"] == sid
