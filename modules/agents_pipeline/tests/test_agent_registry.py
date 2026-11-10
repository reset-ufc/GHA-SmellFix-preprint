"""Unit tests for AgentRegistry (#47).

Verifies the registry yields agents through the module-level make_*_agent
factories (so the monkeypatch seam the workflow tests rely on is honoured),
caches per (role, smell, scenario), and confines its cache per thread.
"""

from __future__ import annotations

import threading

import gha_agents.teams as teams
from gha_agents.agents import ValidSmells
from gha_agents.teams import AgentRegistry

_SMELL = next(iter(ValidSmells))


def test_each_role_builds_through_the_make_factories(monkeypatch):
    monkeypatch.setattr(teams, "make_detection_agent", lambda s, sc: ("det", s, sc))
    monkeypatch.setattr(teams, "make_refactor_agent", lambda s, sc: ("rep", s, sc))
    monkeypatch.setattr(teams, "make_combined_agent", lambda s: ("combined", s))

    reg = AgentRegistry()
    assert reg.detection_agent(_SMELL, "D2") == ("det", _SMELL, "D2")
    assert reg.refactor_agent(_SMELL, "D2") == ("rep", _SMELL, "D2")
    assert reg.combined_agent(_SMELL) == ("combined", _SMELL)


def test_caches_one_agent_per_key(monkeypatch):
    calls = {"n": 0}

    def make(_smell):
        calls["n"] += 1
        return object()

    monkeypatch.setattr(teams, "make_combined_agent", make)
    reg = AgentRegistry()
    first = reg.combined_agent(_SMELL)
    second = reg.combined_agent(_SMELL)
    assert first is second
    assert calls["n"] == 1


def test_distinct_roles_get_distinct_agents(monkeypatch):
    monkeypatch.setattr(teams, "make_detection_agent", lambda s, sc: object())
    monkeypatch.setattr(teams, "make_refactor_agent", lambda s, sc: object())
    reg = AgentRegistry()
    # Cache keys distinguish (role, smell, scenario), so different roles never alias.
    assert reg.detection_agent(_SMELL, "D2") is not reg.refactor_agent(_SMELL, "D2")


def test_cache_is_thread_confined(monkeypatch):
    monkeypatch.setattr(teams, "make_combined_agent", lambda s: object())
    reg = AgentRegistry()
    main_thread_agent = reg.combined_agent(_SMELL)

    captured: dict[str, object] = {}

    def worker():
        captured["agent"] = reg.combined_agent(_SMELL)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    # A separate thread builds into its own thread-local cache → distinct agent.
    assert captured["agent"] is not main_thread_agent
