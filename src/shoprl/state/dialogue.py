"""Structured dialogue state — environment-maintained, not model free-form.

The single source of truth for what the conversation has established. The
model's context window is a VIEW of this state, never the storage: corrections
update the state and invalidate stale plans, so "actually, my budget is $90"
cannot be silently forgotten by a model that stopped attending to turn 3.

The state is language-neutral: it stores extracted facts (via shoprl.lang),
whatever language they arrived in. Product names/SKUs are stored as written.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shoprl.lang import detect_language, english_gloss, extract_info

PermissionStatus = Literal["not_requested", "requested", "granted", "denied", "hold"]


class DialogueState(BaseModel):
    conversation_id: str = "conv-0"
    turn_number: int = 0

    # Last user message + language view.
    raw_user_message: str = ""
    detected_languages: list[str] = Field(default_factory=list)
    code_switched: bool = False
    normalized_english_intent: str = ""

    # Goal + constraints.
    shopping_goal: str = ""
    requested_categories: list[str] = Field(default_factory=list)
    hard_constraints: dict[str, float] = Field(default_factory=dict)
    soft_preferences: list[str] = Field(default_factory=list)
    known_constraint_keys: list[str] = Field(default_factory=list)
    items_user_already_owns: list[str] = Field(default_factory=list)
    forbidden_items: list[str] = Field(default_factory=list)

    # Budget + party.
    budget_total: float | None = None
    currency: str | None = None
    number_of_people: int | None = None
    number_of_children: int | None = None
    age_groups: list[str] = Field(default_factory=list)

    # Plan.
    candidate_products: list[str] = Field(default_factory=list)
    rejected_products: list[str] = Field(default_factory=list)
    selected_products: list[str] = Field(default_factory=list)
    estimated_total: float | None = None
    estimated_savings: float | None = None
    baseline_total: float | None = None    # savings are vs THIS, never claimed without it
    plan_stale: bool = False               # a correction invalidated the current plan

    # Safety + execution.
    permission_status: PermissionStatus = "not_requested"
    permission_items: list[str] = Field(default_factory=list)
    cart_contents: list[str] = Field(default_factory=list)
    browser_state: str = "closed"
    termination_reason: str | None = None

    @property
    def budget_remaining(self) -> float | None:
        if self.budget_total is None:
            return None
        return round(self.budget_total - (self.estimated_total or 0.0), 2)

    # -- user-turn ingestion ------------------------------------------------
    def observe_user_message(self, text: str) -> None:
        """Ingest one user message: detect language, extract facts, merge with
        correction semantics (new facts override; anything that contradicts the
        current plan invalidates it rather than coexisting with it)."""
        self.turn_number += 1
        self.raw_user_message = text
        det = detect_language(text)
        self.detected_languages = det.languages
        self.code_switched = det.code_switched
        info = extract_info(text)
        self.normalized_english_intent = english_gloss(info)

        if info.budget_total is not None:
            if self.budget_total is not None and info.budget_total != self.budget_total:
                self._invalidate_plan()          # budget correction
            self.budget_total = info.budget_total
            self.currency = info.currency or self.currency
        if info.number_of_children is not None:
            if (self.number_of_children is not None
                    and info.number_of_children != self.number_of_children):
                self._invalidate_plan()          # party-size correction
            self.number_of_children = info.number_of_children
        if info.number_of_people is not None:
            self.number_of_people = info.number_of_people

        for cat in info.required_categories:
            if cat not in self.requested_categories and cat not in self.items_user_already_owns:
                self.requested_categories.append(cat)
        for k, v in info.hard_constraints.items():
            if self.hard_constraints.get(k) not in (None, v):
                self._invalidate_plan()          # constraint correction
            self.hard_constraints[k] = v
            if k not in self.known_constraint_keys:
                self.known_constraint_keys.append(k)

        for item in info.owned_items:
            if item not in self.items_user_already_owns:
                self.items_user_already_owns.append(item)
            if item in self.requested_categories:
                self.requested_categories.remove(item)
            self._drop_from_plan(item)
        for item in info.forbidden_items:
            if item not in self.forbidden_items:
                self.forbidden_items.append(item)
            if item in self.requested_categories:
                self.requested_categories.remove(item)
            self._drop_from_plan(item)
        for item in info.removed_items:
            self._drop_from_plan(item)

        if info.hold_permission:
            self.permission_status = "hold"

    # -- plan bookkeeping (driven by the env) --------------------------------
    def record_candidates(self, skus: list[str]) -> None:
        self.candidate_products = list(skus)

    def select_product(self, sku: str, estimated_total: float | None = None) -> None:
        if sku not in self.selected_products:
            self.selected_products.append(sku)
        if estimated_total is not None:
            self.estimated_total = estimated_total
        self.plan_stale = False

    def request_permission(self, items: list[str]) -> None:
        # A user hold stands until the user lifts it; asking doesn't override.
        if self.permission_status != "hold":
            self.permission_status = "requested"
        self.permission_items = list(items)

    def resolve_permission(self, granted: bool) -> None:
        if self.permission_status == "hold":
            return
        self.permission_status = "granted" if granted else "denied"

    def add_to_cart(self, sku: str) -> bool:
        """True iff the add is permitted: explicit grant covering this SKU."""
        ok = self.permission_status == "granted" and sku in self.permission_items
        if ok:
            self.cart_contents.append(sku)
        return ok

    # -- internals ------------------------------------------------------------
    def _invalidate_plan(self) -> None:
        if self.selected_products or self.estimated_total is not None:
            self.plan_stale = True
        self.selected_products = []
        self.estimated_total = None
        self.estimated_savings = None
        if self.permission_status in ("requested", "granted"):
            self.permission_status = "not_requested"   # stale plan ⇒ stale approval
            self.permission_items = []

    def _drop_from_plan(self, item: str) -> None:
        low = item.lower()
        kept = [s for s in self.selected_products if low not in s.lower()]
        if kept != self.selected_products:
            self.selected_products = kept
            self.plan_stale = True
            self.estimated_total = None
