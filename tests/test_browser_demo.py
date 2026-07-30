"""Browser demo (replay path): storefront generation + observation parsing.
Model-free and Playwright-free — the browser itself is exercised by
scripts/demo_browser.py (headless run with screenshot artifacts)."""
from shoprl.data.catalog import generate_catalog
from shoprl.env.browser_demo import render_storefront_html, skus_in


def test_storefront_embeds_catalog_and_hooks():
    """Render-agnostic contract: the storefront (React build or legacy
    fallback) embeds the catalog and exposes the driver interface."""
    from shoprl.data.catalog import generate_catalog
    from shoprl.env.browser_demo import render_storefront_html
    cat = generate_catalog(n=5, seed=0)
    html = render_storefront_html(cat)
    assert "__PRODUCTS__" not in html          # catalog actually injected
    assert cat[0].sku in html
    for hook in ("pennymart", "__feedback", "__fbN", "__human", "__uiev",
                 "👍", "👎", "SIMULATED"):
        assert hook in html, hook

def test_skus_in_parses_search_observation():
    obs = ("Matching products (cheapest first):\n"
           "- LAP-0110: $1047, 16GB RAM, 2.9lbs, 10hrs, Apple\n"
           "- LAP-0101: $1168, 16GB RAM, 2.6lbs, 12hrs, Lenovo\n"
           "- LAP-0110: duplicate line")
    assert skus_in(obs) == ["LAP-0110", "LAP-0101"]   # ordered, de-duplicated
    assert skus_in("No matching products found.") == []


def test_agent_bubbles_carry_feedback_thumbs():
    # the JS is exercised for real in the headless smoke; here we pin the
    # contract the driver depends on: agent bubbles register __feedback slots
    from shoprl.data.catalog import generate_catalog
    from shoprl.env.browser_demo import render_storefront_html
    html = render_storefront_html(generate_catalog(n=5, seed=0))
    assert "window.__feedback" in html and "__fbN" in html
    assert html.count("👍") >= 1 and html.count("👎") >= 1
