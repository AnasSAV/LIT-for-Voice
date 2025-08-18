from pathlib import Path
import shutil
import wave

# Tests rely on existing fixtures: client (AsyncClient) and redis_cleanup from tests/conftest.py

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"
RAVDESS_SUBSET = DATA_ROOT / "dev" / "ravdess_subset"


def _write_wav(path: Path, seconds: float = 1.0, framerate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(framerate * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x00" * nframes)


def _reset_dataset_dir():
    if RAVDESS_SUBSET.exists():
        shutil.rmtree(RAVDESS_SUBSET, ignore_errors=True)
    RAVDESS_SUBSET.mkdir(parents=True, exist_ok=True)


async def test_manifest_build_and_summary(client):
    # Arrange: create a few RAVDESS-style files with different labels
    _reset_dataset_dir()
    _write_wav(RAVDESS_SUBSET / "03-01-05-01-01-01-01.wav", seconds=0.5)  # angry
    _write_wav(RAVDESS_SUBSET / "03-01-04-01-01-01-01.wav", seconds=0.5)  # sad
    _write_wav(RAVDESS_SUBSET / "03-01-03-01-01-01-01.wav", seconds=0.5)  # happy

    # Establish session cookie
    r0 = await client.get("/session")
    client.cookies.update(r0.cookies)

    # Act: fetch files (triggers manifest build and sets active dataset automatically)
    r = await client.get("/datasets/files")
    assert r.status_code == 200
    payload = r.json()
    assert payload["active"] == "ravdess_subset"
    assert payload["total"] == 3
    assert len(payload["files"]) == 3

    labels = sorted([f.get("label") for f in payload["files"]])
    assert labels == ["angry", "happy", "sad"]
    for f in payload["files"]:
        # duration computed by wave module; should be >0
        assert f.get("duration") and f["duration"] > 0
        # stable join key present
        assert f.get("h") and isinstance(f["h"], str)

    # Summary endpoint should include label_counts
    rs = await client.get("/datasets/summary")
    assert rs.status_code == 200
    summary = rs.json()["summary"]
    assert summary["total"] == 3
    assert summary["total_bytes"] > 0
    assert summary["label_counts"]["angry"] == 1
    assert summary["label_counts"]["sad"] == 1
    assert summary["label_counts"]["happy"] == 1


async def test_pagination(client):
    # Arrange
    _reset_dataset_dir()
    for i, em in enumerate(["01", "02", "03", "04", "05"], start=1):
        _write_wav(RAVDESS_SUBSET / f"03-01-{em}-01-01-01-0{i}.wav", seconds=0.25)

    r0 = await client.get("/session"); client.cookies.update(r0.cookies)

    # Request a window
    r = await client.get("/datasets/files", params={"limit": 2, "offset": 2})
    assert r.status_code == 200
    payload = r.json()
    assert payload["total"] == 5
    assert len(payload["files"]) == 2


async def test_version_invalidation_on_change(client):
    # Arrange initial set
    _reset_dataset_dir()
    _write_wav(RAVDESS_SUBSET / "03-01-05-01-01-01-01.wav", seconds=0.5)

    r0 = await client.get("/session"); client.cookies.update(r0.cookies)

    r1 = await client.get("/datasets/files")
    assert r1.status_code == 200
    total1 = r1.json()["total"]
    assert total1 == 1

    # Add a new file -> version should change and manifest rebuild
    _write_wav(RAVDESS_SUBSET / "03-01-04-01-01-01-02.wav", seconds=0.5)

    r2 = await client.get("/datasets/files")
    assert r2.status_code == 200
    total2 = r2.json()["total"]
    assert total2 == 2


async def test_reindex_endpoint(client):
    _reset_dataset_dir()
    _write_wav(RAVDESS_SUBSET / "03-01-03-01-01-01-01.wav", seconds=0.5)

    r0 = await client.get("/session"); cookies = r0.cookies

    # Build once
    await client.get("/datasets/files")

    # Force rebuild
    rr = await client.post("/datasets/reindex", json={"id": "ravdess_subset"})
    assert rr.status_code == 200
    assert rr.json()["ok"] is True
