"""Structured abstract actions — the v2 (PennyPilot) policy's output vocabulary.

The policy emits ONE JSON action per turn (optionally wrapped in prose); it
never emits raw browser selectors or environment-specific commands. Environment
adapters translate these into catalog calls, WebShop actions, or Playwright
operations — which is what keeps the policy portable across environments and
keeps the browser a projection of decisions, not the decision surface.

Parsing is strict-but-recoverable: extract the first balanced JSON object that
contains an "action" key, validate it against the typed schema, and return
either the typed action or a machine-readable error. Invalid actions are the
env's problem to penalize (reward layer), not an exception path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class AskUser(BaseModel):
    action: Literal["ask_user"]
    question: str


class Search(BaseModel):
    action: Literal["search"]
    query: str


class InspectProduct(BaseModel):
    action: Literal["inspect_product"]
    product_id: str


class SelectProduct(BaseModel):
    action: Literal["select_product"]
    product_id: str
    reason: str = ""


class RequestCartPermission(BaseModel):
    action: Literal["request_cart_permission"]
    items: list[str]
    estimated_total: float


class AddToCart(BaseModel):
    action: Literal["add_to_cart"]
    product_id: str


AbstractAction = Annotated[
    Union[AskUser, Search, InspectProduct, SelectProduct,
          RequestCartPermission, AddToCart],
    Field(discriminator="action"),
]

_ADAPTER: TypeAdapter = TypeAdapter(AbstractAction)


@dataclass
class ParseResult:
    """Outcome of parsing one agent turn. Exactly one of action/error is set;
    `raw` carries the extracted dict when JSON parsed but validation failed
    (useful for error analysis of near-miss model outputs)."""
    action: object | None = None
    error: str | None = None
    raw: dict | None = None

    @property
    def ok(self) -> bool:
        return self.action is not None


def _first_json_object(text: str) -> dict | None:
    """First balanced JSON object in `text` containing an "action" key.
    Models often wrap the action in prose; we scan candidate '{' offsets and
    let the JSON decoder decide where the object ends."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text, i)
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except ValueError:
            pass
        i = text.find("{", i + 1)
    return None


def parse_agent_action(text: str) -> ParseResult:
    obj = _first_json_object(text or "")
    if obj is None:
        return ParseResult(error="no JSON action object found")
    try:
        return ParseResult(action=_ADAPTER.validate_python(obj), raw=obj)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        return ParseResult(error=f"invalid action: {loc}: {first['msg']}", raw=obj)


def action_to_json(action) -> str:
    """Canonical single-line JSON for transcripts / SFT targets."""
    return json.dumps(action.model_dump(), separators=(", ", ": "))
