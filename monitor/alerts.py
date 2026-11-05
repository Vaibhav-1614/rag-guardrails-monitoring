"""Alert delivery: a GitHub issue per detector (deduplicated) and an optional Discord/Slack webhook."""
from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)
LABEL = "drift-alert"


def format_alert(a: dict) -> str:
    return f"- **{a['detector']}** `{a['metric']}` = {a['value']} (threshold: {a['threshold']})"


def open_github_issues(alerts: list[dict], as_of: str, report_url: str | None = None,
                       repo: str | None = None, token: str | None = None,
                       transport: httpx.BaseTransport | None = None) -> list[str]:
    """One open issue per detector; later alerts for the same detector are added as comments."""
    repo = repo or os.environ.get("GITHUB_REPOSITORY")
    token = token or os.environ.get("GITHUB_TOKEN")
    if not (repo and token and alerts):
        return []
    by_detector: dict[str, list[dict]] = {}
    for a in alerts:
        by_detector.setdefault(a["detector"], []).append(a)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    urls = []
    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=30, transport=transport) as gh:
        existing = gh.get(f"/repos/{repo}/issues", params={"labels": LABEL, "state": "open", "per_page": 100})
        existing.raise_for_status()
        open_by_title = {i["title"]: i for i in existing.json()}
        for detector, items in by_detector.items():
            title = f"Drift alert: {detector}"
            body = "\n".join([f"Nightly monitoring run for **{as_of}** crossed a threshold:", "", *map(format_alert, items)]
                             + (["", f"Report: {report_url}"] if report_url else []))
            if title in open_by_title:
                r = gh.post(f"/repos/{repo}/issues/{open_by_title[title]['number']}/comments", json={"body": body})
                urls.append(open_by_title[title]["html_url"])
            else:
                r = gh.post(f"/repos/{repo}/issues", json={"title": title, "body": body, "labels": [LABEL]})
                if r.status_code < 300:
                    urls.append(r.json().get("html_url", ""))
            if r.status_code >= 300:
                logger.warning("GitHub issue call failed: %s %s", r.status_code, r.text[:200])
    return urls


def post_webhook(alerts: list[dict], as_of: str, url: str | None = None, transport: httpx.BaseTransport | None = None) -> bool:
    url = url or os.environ.get("DISCORD_WEBHOOK_URL") or os.environ.get("SLACK_WEBHOOK_URL")
    if not (url and alerts):
        return False
    text = "\n".join([f"Drift alerts for {as_of}:", *map(format_alert, alerts)])
    payload = {"content": text} if "discord" in url else {"text": text}
    with httpx.Client(timeout=15, transport=transport) as c:
        r = c.post(url, json=payload)
    return r.status_code < 300
