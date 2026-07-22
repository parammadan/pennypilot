"""Browser demo (replay path): storefront generation + observation parsing.
Model-free and Playwright-free — the browser itself is exercised by
scripts/demo_browser.py (headless run with screenshot artifacts)."""
from shoprl.data.catalog import generate_catalog
from shoprl.env.browser_demo import render_storefront_html, skus_in


def test_storefront_embeds_catalog_and_hooks():
    catalog = generate_catalog(n=25, seed=0)
    html = render_storefront_html(catalog)
    for p in catalog[:5]:
        assert p.sku in html
    for hook in ("window.pennymart", "results(skus)", "permission(items",
                 "addToCart(sku)", 'id="search"', 'id="modal"', "SIMULATED",
                 "laptopSVG", "Add to Cart"):
        assert hook in html
    assert html.count("<script>") == 1          # self-contained, no external JS


def test_skus_in_parses_search_observation():
    obs = ("Matching products (cheapest first):\n"
           "- LAP-0110: $1047, 16GB RAM, 2.9lbs, 10hrs, Apple\n"
           "- LAP-0101: $1168, 16GB RAM, 2.6lbs, 12hrs, Lenovo\n"
           "- LAP-0110: duplicate line")
    assert skus_in(obs) == ["LAP-0110", "LAP-0101"]   # ordered, de-duplicated
    assert skus_in("No matching products found.") == []
