"""Package imports and installed entry-point registration."""

from __future__ import annotations

import subprocess
import sys

import pytest
from relarena_core.registry import registry
from relarena_core.userdb import PredictiveQuery as ArenaQuery

from tabpfn_rel import PredictiveQuery, TabPFNRel, TabPFNRelLocalModel


@pytest.mark.parametrize("first", ["relarena_core", "tabpfn_rel"])
def test_import_orders_and_repeated_discovery(first: str) -> None:
    code = f"""
import sys
import {first}
import relarena_core
import tabpfn_rel
assert set(relarena_core.registry.names()) == {{
    'tabpfn-rel-local-2026-08-15', 'tabpfn-rel-client-2026-08-15',
    'tabpfn-rel-client-2026-09-18',
    'tabpfn-rel-client-2026-09-28',
    'tabpfn-rel-local-2026-09-28'
}}
assert 'relarena.models' not in sys.modules
for name in ('tabpfn', 'tabpfn_client', 'fastdfs'):
    assert name not in sys.modules, name
relarena_core.discover_models()
relarena_core.discover_models()
local = relarena_core.registry.get('tabpfn-rel-local-2026-08-15')
assert local is tabpfn_rel.TabPFNRelLocalModel
client = relarena_core.registry.get('tabpfn-rel-client-2026-08-15')
assert client is tabpfn_rel.TabPFNRelClientModel
assert 'tabpfn' not in sys.modules
assert 'tabpfn_client' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_decorator_registration_and_shared_rpi() -> None:
    assert registry.get("tabpfn-rel-local-2026-08-15") is TabPFNRelLocalModel
    assert PredictiveQuery is ArenaQuery


def test_wrapper_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="model must be"):
        TabPFNRel(model="unknown")


def test_wrapper_requires_fit_before_prediction() -> None:
    with pytest.raises(RuntimeError, match="Call fit"):
        TabPFNRel(model="client").predict(
            PredictiveQuery(entities="all", at_timestamp="test_timestamp")
        )


@pytest.mark.parametrize(
    "selector,registered",
    [
        ("client", "tabpfn-rel-client-latest"),
        ("local", "tabpfn-rel-local-latest"),
        ("local-2026-08-15", "tabpfn-rel-local-2026-08-15"),
        ("client-2026-08-15", "tabpfn-rel-client-2026-08-15"),
        ("client-2026-09-18", "tabpfn-rel-client-2026-09-18"),
        (
            "client-2026-09-28",
            "tabpfn-rel-client-2026-09-28",
        ),
        ("local-2026-09-28", "tabpfn-rel-local-2026-09-28"),
    ],
)
def test_wrapper_selects_registered_model(selector: str, registered: str) -> None:
    model = TabPFNRel(model=selector)
    assert model._model == registered
    assert registry.get(model._model) is registry.get(registered)


@pytest.mark.parametrize(
    "alias,dated",
    [
        ("tabpfn-rel-client-latest", "tabpfn-rel-client-2026-09-28"),
        ("tabpfn-rel-local-latest", "tabpfn-rel-local-2026-09-28"),
    ],
)
def test_latest_alias_resolves_to_the_dated_model(alias: str, dated: str) -> None:
    assert registry.resolve(alias) == dated
    assert registry.alias_for(dated) == alias
    assert registry.get(alias).name == dated
    assert alias not in registry.names()
