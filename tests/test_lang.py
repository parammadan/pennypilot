"""Language layer: detection + bilingual extraction (constraints must survive
any input language). Includes the spec's canonical Spanglish example."""
from shoprl.lang import detect_language, english_gloss, extract_info


def test_spec_canonical_spanglish_example():
    msg = "Voy con dos niños, budget is $100, and necesito sunscreen SPF 50."
    d = detect_language(msg)
    assert d.code_switched and set(d.languages) == {"english", "spanish"}
    info = extract_info(msg)
    assert info.number_of_children == 2
    assert info.budget_total == 100.0
    assert info.currency == "USD"
    assert info.required_categories == ["sunscreen"]
    assert info.hard_constraints == {"spf_minimum": 50.0}


def test_detect_pure_english_and_spanish():
    assert detect_language("I need a new laptop for work.").primary == "english"
    assert not detect_language("I need a new laptop.").code_switched
    d = detect_language("Necesito una computadora nueva para mi hijo.")
    assert d.primary == "spanish" and not d.code_switched


def test_budget_forms():
    assert extract_info("My budget is about $90").budget_total == 90.0
    assert extract_info("presupuesto de 250 dólares").budget_total == 250.0
    assert extract_info("keep it under 60 dollars").budget_total == 60.0


def test_owned_items_not_required():
    info = extract_info("I already have sunscreen, but necesito una sombrilla.")
    assert info.owned_items == ["sunscreen"]
    assert info.required_categories == ["umbrella"]


def test_forbidden_spanglish():
    info = extract_info("No quiero una umbrella.")
    assert info.forbidden_items == ["umbrella"]
    assert info.required_categories == []


def test_removal_and_hold():
    info = extract_info("Remove the waterproof pouch. Do not add anything yet.")
    assert info.removed_items == ["waterproof pouch"]
    assert info.hold_permission


def test_people_counts():
    assert extract_info("somos cuatro").number_of_people == 4
    assert extract_info("a party of 6").number_of_people == 6
    assert extract_info("make that three children, not two").number_of_children == 3


def test_laptop_constraints_bilingual():
    info = extract_info("Busco una laptop con 16GB de memoria y 10 horas de batería.")
    assert info.required_categories == ["laptop"]
    assert info.hard_constraints["min_ram"] == 16.0
    assert info.hard_constraints["min_battery"] == 10.0


def test_gloss_is_compact_and_faithful():
    g = english_gloss(extract_info(
        "Voy con dos niños, budget is $100, and necesito sunscreen SPF 50."))
    assert g.startswith("[interpreted: ") and "2 children" in g
    assert "budget $100 USD" in g and "sunscreen" in g and "spf_minimum=50" in g
    assert english_gloss(extract_info("hola!")) == ""
