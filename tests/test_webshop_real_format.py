"""Adapter hardening (Block 3 step 14): parse the REAL WebShop observation
formats — both `text` (simple [SEP]) and `text_rich` (button-markup), plus
pagination and options-page chrome — verified against
web_agent_text_env.convert_html_to_text (princeton-nlp/WebShop, read 2026-07-23).
CPU, no install needed: these are the actual token conventions the source emits.
"""
from shoprl.env.webshop_env import (has_next_page, parse_results_page,
                                    strip_markup)

# `text` mode — pure [SEP], the format the RL baselines actually use.
TEXT_MODE = (
    "Instruction: [SEP] i want a gluten free whole bean coffee, and price "
    "lower than 40 [SEP] Back to Search [SEP] Page 1 (Total results: 50) "
    "[SEP] Next > [SEP] B078GWRC1J [SEP] Bright Whole Bean Coffee, 2 lb "
    "[SEP] $34.99 [SEP] B07Salbum1 [SEP] not-an-asin junk [SEP] "
    "B08KBVJ4XN [SEP] Dark Roast Whole Bean [SEP] $15.30")

# `text_rich` mode — same content wrapped in the button markers the source
# emits for clickable elements.
TEXT_RICH_MODE = (
    "Instruction: [SEP] gluten free coffee [SEP] [button] Back to Search "
    "[button_] [SEP] Page 1 (Total results: 50) [SEP] [button] Next > "
    "[button_] [SEP] [clicked button] B078GWRC1J [clicked button_] [SEP] "
    "Bright Whole Bean Coffee, 2 lb [SEP] $34.99 [SEP] [button] B08KBVJ4XN "
    "[button_] [SEP] Dark Roast Whole Bean [SEP] $15.30")


def test_text_mode_parses_and_ignores_chrome():
    items = parse_results_page(TEXT_MODE)
    assert [it.asin for it in items] == ["B078GWRC1J", "B08KBVJ4XN"]
    assert items[0].price == 34.99 and items[1].price == 15.30
    # "Next >", "Back to Search", "Page 1", junk lines all excluded as products
    assert all(it.title not in ("Next >", "Back to Search") for it in items)


def test_text_rich_mode_button_markup_stripped():
    assert "[button]" not in strip_markup(TEXT_RICH_MODE)
    items = parse_results_page(TEXT_RICH_MODE)
    assert [it.asin for it in items] == ["B078GWRC1J", "B08KBVJ4XN"]
    assert items[1].price == 15.30          # survives button wrapping


def test_both_modes_yield_identical_products():
    a = [(i.asin, i.price) for i in parse_results_page(TEXT_MODE)]
    b = [(i.asin, i.price) for i in parse_results_page(TEXT_RICH_MODE)]
    assert a == b                           # the decoupling boundary is mode-agnostic


def test_pagination_detected():
    assert has_next_page(TEXT_MODE) is True
    assert has_next_page(TEXT_RICH_MODE) is True
    assert has_next_page("Instruction: [SEP] x [SEP] no more pages") is False


def test_price_scan_tolerates_interleaved_tokens():
    # a rating token can sit between title and price on some pages
    obs = ("B01ABCDEFG [SEP] Some Product Title [SEP] Rating: 4.5 [SEP] "
           "$12.34 [SEP] Prime")
    items = parse_results_page(obs)
    assert len(items) == 1 and items[0].price == 12.34


def test_dedup_across_page_chrome():
    obs = ("B01ABCDEFG [SEP] Widget [SEP] $9.99 [SEP] "
           "B01ABCDEFG [SEP] Widget [SEP] $9.99")   # repeated across chrome
    assert len(parse_results_page(obs)) == 1
