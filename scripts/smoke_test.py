"""Smoke test of a running genre-classifier app (local container or prod).

Waits until:
  1. GET /ready returns 200 (model loaded),
  2. GET /model-info reports the expected model version (and git SHA if given),
then sends a real POST /predict and checks the answer.

Exit code 0 if everything passes, 1 otherwise.

Usage:
  python scripts/smoke_test.py --url http://localhost:8000 --expected-version test
  python scripts/smoke_test.py --url $PROD_URL --expected-version 3 --expected-sha <sha> --timeout 900
"""
import argparse
import sys
import time

import requests

# kept here on purpose: this script must run without importing ml/ or app/
GENRES = {"classical", "country", "hip-hop", "jazz", "metal", "pop", "reggae", "techno"}
SAMPLE_INPUT = {
    "danceability": 0.62,
    "energy": 0.81,
    "loudness": -5.2,
    "speechiness": 0.05,
    "acousticness": 0.10,
    "instrumentalness": 0.0,
    "liveness": 0.12,
    "valence": 0.55,
    "tempo": 124.0,
}
POLL_SECONDS = 10
REQUEST_TIMEOUT = 30  # Render free tier can take a while to wake up


def log(msg):
    print(f"[smoke] {msg}", flush=True)


def fail(msg):
    log(f"FAIL: {msg}")
    sys.exit(1)


def wait_until(check, what, deadline):
    """Call check() until it returns (True, info) or the deadline passes."""
    last = "no answer yet"
    while time.monotonic() < deadline:
        try:
            ok, last = check()
            if ok:
                log(f"OK: {what} ({last})")
                return
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        log(f"waiting for {what}... last: {last}")
        time.sleep(POLL_SECONDS)
    fail(f"timeout waiting for {what}. Last: {last}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="base URL of the app")
    p.add_argument("--expected-version", required=True, help="model version the app must serve")
    p.add_argument("--expected-sha", help="git SHA the running image must report")
    p.add_argument("--timeout", type=int, default=600, help="seconds to wait in total")
    args = p.parse_args()

    base = args.url.rstrip("/")
    expected_version = str(args.expected_version).strip()
    deadline = time.monotonic() + args.timeout
    log(f"target={base} expected_version={expected_version} "
        f"expected_sha={args.expected_sha or '-'} timeout={args.timeout}s")

    # 1. readiness
    def ready():
        r = requests.get(f"{base}/ready", timeout=REQUEST_TIMEOUT)
        return r.status_code == 200, f"HTTP {r.status_code}"

    wait_until(ready, "/ready == 200", deadline)

    # 2. right model version (and right image, so we don't test the old deploy)
    def model_info():
        r = requests.get(f"{base}/model-info", timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        info = r.json()
        version = str(info.get("model_version"))
        sha = info.get("git_sha")
        ok = version == expected_version and (not args.expected_sha or sha == args.expected_sha)
        return ok, f"model_version={version} git_sha={sha}"

    wait_until(model_info, "/model-info with expected version/sha", deadline)

    # 3. a real prediction
    r = requests.post(f"{base}/predict", json=SAMPLE_INPUT, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        fail(f"/predict returned HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    top3 = body.get("top3")
    if not isinstance(top3, list) or len(top3) != 3:
        fail(f"/predict must return 3 genres, got: {top3}")
    for item in top3:
        if item.get("genre") not in GENRES:
            fail(f"unknown genre in prediction: {item}")
        proba = item.get("proba")
        if not isinstance(proba, (int, float)) or not 0 <= proba <= 1:
            fail(f"invalid probability: {item}")
    probas = [item["proba"] for item in top3]
    if probas != sorted(probas, reverse=True):
        fail(f"top3 is not sorted by probability: {top3}")
    if str(body.get("model_version")) != expected_version:
        fail(f"/predict model_version={body.get('model_version')}, expected {expected_version}")
    log(f"OK: /predict -> {[(i['genre'], round(i['proba'], 3)) for i in top3]}")

    log("all checks passed")


if __name__ == "__main__":
    main()
