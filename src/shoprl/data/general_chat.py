"""General-chat data for the H3 forgetting/rehearsal experiment.

Two disjoint pools:
  - `rehearsal_questions()` — a template-generated pool of everyday questions
    (EN + ES). Answered by the BASE 7B (self-distillation) and mixed into Arm B's
    SFT so the model rehearses general conversation while learning to shop.
  - `GENERAL_EVAL` — a FIXED, hand-written held-out set (topics disjoint from the
    rehearsal templates) used to score general-chat RETENTION before/after
    training. Never used as training data.

Retention scoring (`answered_generally`): a response counts as retained general
chat iff it is real prose and emits NO shopping-action JSON — i.e., the model
did NOT collapse a general question into a store action (the forgetting failure
mode from CHALLENGES #27).
"""
from __future__ import annotations

import random

# --- rehearsal pool (templated, EN + ES) -------------------------------------
_CAPITAL_PLACES = ["France", "Japan", "Brazil", "Egypt", "Canada", "Italy",
                   "Kenya", "Norway", "Peru", "Thailand", "Greece", "Mexico",
                   "India", "Portugal", "Chile", "Vietnam", "Poland", "Ireland",
                   "Morocco", "Sweden"]
_CONCEPTS = ["photosynthesis", "gravity", "inflation", "evaporation",
             "why the sky is blue", "how vaccines work", "the water cycle",
             "what causes tides", "how a rainbow forms", "why leaves change color",
             "how the internet works", "what DNA is", "how magnets work",
             "why we dream", "how bread rises", "what causes earthquakes"]
_HOWTO = ["boil an egg", "tie a tie", "study for an exam", "start running",
          "save money each month", "write a cover letter", "make coffee",
          "get better sleep", "stay focused while working", "keep a plant alive",
          "make friends in a new city", "learn a language", "fix a slow laptop",
          "pack light for a trip", "reduce screen time", "start journaling"]
_PEOPLE = ["Ada Lovelace", "Marie Curie", "Nelson Mandela", "Leonardo da Vinci",
           "Alan Turing", "Frida Kahlo", "Isaac Newton", "Cleopatra",
           "Charles Darwin", "Maya Angelou"]
_DEFINE = ["a metaphor", "photosphere", "an algorithm", "compound interest",
           "an ecosystem", "a synonym", "GDP", "osmosis", "a palindrome",
           "artificial intelligence", "a prime number", "empathy"]
_DIFF = [("a crocodile", "an alligator"), ("weather", "climate"),
         ("a comet", "an asteroid"), ("jam", "jelly"), ("a lake", "a pond"),
         ("stalactites", "stalagmites"), ("HTTP", "HTTPS")]
_RECS = ["a beginner-friendly board game", "a relaxing weekend activity",
         "a classic novel to read", "a podcast about science",
         "a low-effort healthy breakfast", "a hobby to reduce stress"]
_SMALL = ["How are you today?", "What can you help me with?",
          "Tell me a short joke.", "What's a fun fact?",
          "Recommend a good book.", "What's a nice weekend activity?",
          "Cheer me up, please.", "What should I cook tonight?",
          "What's your favorite color?", "Say something encouraging."]
_ES = ["¿Cuál es la capital de España?", "Cuéntame un chiste corto.",
       "¿Cómo estás hoy?", "Explícame la fotosíntesis en una frase.",
       "¿Qué puedo cocinar para la cena?", "Dame un dato curioso.",
       "¿Cómo se dice 'thank you' en francés?", "Recomiéndame una película.",
       "¿Cuánto es 15 por 12?", "¿Qué es un sinónimo?",
       "¿Cuál es el planeta más grande?", "¿Por qué llueve?",
       "¿Cómo se hace café?", "Dame un consejo para estudiar mejor.",
       "¿Quién pintó Las Meninas?", "¿Qué es un algoritmo?",
       "¿Cuántos días tiene febrero?", "Cuéntame algo sobre los delfines.",
       "¿Cómo puedo relajarme?", "¿Qué significa 'resiliencia'?"]


def rehearsal_questions(n: int = 180, seed: int = 0) -> list[dict]:
    """Return up to `n` {q, lang} general questions from the templated pool."""
    rng = random.Random(f"reh-{seed}")
    pool: list[dict] = []
    for p in _CAPITAL_PLACES:
        pool.append({"q": f"What's the capital of {p}?", "lang": "en"})
    for c in _CONCEPTS:
        pool.append({"q": f"Explain {c} in one or two sentences.", "lang": "en"})
    for h in _HOWTO:
        pool.append({"q": f"How do I {h}?", "lang": "en"})
    for pe in _PEOPLE:
        pool.append({"q": f"Who was {pe}?", "lang": "en"})
    for d in _DEFINE:
        pool.append({"q": f"What is {d}?", "lang": "en"})
    for x, y in _DIFF:
        pool.append({"q": f"What's the difference between {x} and {y}?", "lang": "en"})
    for r in _RECS:
        pool.append({"q": f"Recommend {r}.", "lang": "en"})
    for a, b in [(23, 47), (128, 9), (15, 12), (360, 6), (7, 8), (99, 3),
                 (44, 5), (17, 17), (250, 4)]:
        pool.append({"q": f"What is {a} times {b}?", "lang": "en"})
    pool += [{"q": s, "lang": "en"} for s in _SMALL]
    pool += [{"q": s, "lang": "es"} for s in _ES]
    eval_qs = {x["q"] for x in GENERAL_EVAL}   # never train on a held-out question
    pool = [x for x in pool if x["q"] not in eval_qs]
    rng.shuffle(pool)
    return pool[:n]


# --- held-out retention eval (fixed, disjoint from the rehearsal templates) ---
GENERAL_EVAL: list[dict] = [
    {"q": "What's the capital of Australia?", "lang": "en"},
    {"q": "Why do we have seasons?", "lang": "en"},
    {"q": "How do I make scrambled eggs?", "lang": "en"},
    {"q": "Who wrote Romeo and Juliet?", "lang": "en"},
    {"q": "What does 'ephemeral' mean?", "lang": "en"},
    {"q": "What is 34 plus 58?", "lang": "en"},
    {"q": "Tell me a fun fact about octopuses.", "lang": "en"},
    {"q": "What's a good icebreaker question?", "lang": "en"},
    {"q": "Explain what a black hole is, simply.", "lang": "en"},
    {"q": "How many continents are there?", "lang": "en"},
    {"q": "What's the difference between weather and climate?", "lang": "en"},
    {"q": "Give me a tip for public speaking.", "lang": "en"},
    {"q": "What language is spoken in Brazil?", "lang": "en"},
    {"q": "How does an airplane stay in the air?", "lang": "en"},
    {"q": "What's a synonym for 'happy'?", "lang": "en"},
    {"q": "Who painted the Mona Lisa?", "lang": "en"},
    {"q": "What is the boiling point of water in Celsius?", "lang": "en"},
    {"q": "Recommend a relaxing hobby.", "lang": "en"},
    {"q": "What's the tallest mountain on Earth?", "lang": "en"},
    {"q": "Explain the difference between a virus and bacteria.", "lang": "en"},
    {"q": "How do I stay motivated?", "lang": "en"},
    {"q": "What year did World War II end?", "lang": "en"},
    {"q": "What's a haiku?", "lang": "en"},
    {"q": "Convert 10 kilometers to miles, roughly.", "lang": "en"},
    {"q": "What's your favorite season and why?", "lang": "en"},
    {"q": "How are you doing today?", "lang": "en"},
    {"q": "Tell me a short joke.", "lang": "en"},
    {"q": "¿Cuál es la capital de Argentina?", "lang": "es"},
    {"q": "¿Por qué el cielo es azul?", "lang": "es"},
    {"q": "¿Cómo se hace un té?", "lang": "es"},
    {"q": "¿Cuánto es 40 más 25?", "lang": "es"},
    {"q": "Cuéntame algo interesante sobre el espacio.", "lang": "es"},
    {"q": "¿Qué significa 'efímero'?", "lang": "es"},
    {"q": "Dame un consejo para dormir mejor.", "lang": "es"},
    {"q": "¿Quién escribió Don Quijote?", "lang": "es"},
    {"q": "¿Cómo estás hoy?", "lang": "es"},
    {"q": "Recomiéndame un pasatiempo relajante.", "lang": "es"},
    {"q": "¿Cuántos planetas hay en el sistema solar?", "lang": "es"},
    {"q": "Explica qué es la gravedad, de forma simple.", "lang": "es"},
    {"q": "¿Cuál es el río más largo del mundo?", "lang": "es"},
]


def answered_generally(text: str) -> bool:
    """True iff `text` is a real general-chat answer (prose, NO shopping action).
    Emitting a shopping-action JSON for a general question = forgetting."""
    from shoprl.actions import parse_agent_action
    if parse_agent_action(text).ok:          # collapsed into a store action
        return False
    prose = text.split("{", 1)[0].strip()    # drop any stray JSON-ish tail
    return len(prose) >= 8                    # non-trivial natural-language reply
