#!/usr/bin/env python3
"""
hookscore.py - score the first line of a Reel on the five things strong hooks
have in common, and rank a batch of them against each other.

What this is:  five local heuristics, computed on your machine from the text
alone. They measure properties that hooks which hold attention tend to share -
a length you can say in under three seconds, a concrete marker, something at stake,
the payload at the front rather than the back, and a viewer to aim it at.

What this is NOT:  a view predictor. It was tested against 74 real short-form
hooks, transcribed from the first three seconds of the top eight and bottom
eight performers on five channels. Separating a real hook from a deliberately
bad one it does well: AUC 0.83, and 9 of 10 written-to-be-bad hooks scored
below the real median. Separating a good creator's hits from that same
creator's misses it barely does at all: AUC 0.56, where 0.50 is a coin flip.

So use it for what it measured well. It catches greetings, preambles, hooks
with nothing concrete in them and hooks that take five seconds to say. It will
not tell you which of two decent hooks will travel, and nothing that reads text
can, because that is decided by your face, your edit, your audio and who
Instagram shows it to. Trust the retention graph over this script.

Each check returns 0 to 100. Higher is better.

Usage
  python3 hookscore.py hooks.txt              # one hook per line, ranked
  python3 hookscore.py --hook "I lost $18,000 on one missing contract."
  pbpaste | python3 hookscore.py -
  python3 hookscore.py hooks.txt --json
"""

import argparse
import json
import re
import statistics
import sys

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9$%'’-]+")
NUMBER_RE = re.compile(
    r"\$\s?\d[\d,.]*"                                # money, whole
    r"|\b\d[\d,]*(?:\.\d+)?\s?"                      # a figure, with or
    r"(?:%|k\b|x\b|hrs?\b|hours?\b|mins?\b|minutes?\b"  # without a unit
    r"|days?\b|weeks?\b|months?\b|years?\b"
    r"|horas?\b|minutos?\b|segundos?\b|d[ií]as?\b|semanas?\b|mes(?:es)?\b"
    r"|a[ñn]os?\b|mil\b|millones\b|lucas?\b|uf\b)?",
    re.IGNORECASE)
PROPER_RE = re.compile(r"(?<!^)\b[A-ZÁÉÍÓÚÑ][a-záéíóúüñ]{2,}\b")
HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

# Spoken hooks say their numbers out loud. "Zero dollars" and "twenty grand"
# are as concrete as "$0" and "$20,000", and counting only digits missed them.
# "one" and "first" are deliberately absent: they are filler far more often
# than they are a quantity.
SPOKEN_NUMBERS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen", "half", "twice", "triple",
}
MONEY_WORDS = {
    "dollars", "dollar", "bucks", "grand", "percent", "cents",
    "millionaire", "billionaire", "revenue", "profit", "salary", "rent",
}

# Words that put something on the line. A hook with none of these is a
# statement; a hook with one is a reason to keep watching.
STAKES = {
    "stop", "never", "wrong", "mistake", "mistakes", "lost", "lose", "losing",
    "cost", "costs", "broke", "broken", "failed", "failure", "fail", "nobody",
    "no", "not", "don't", "dont", "doesn't", "didn't", "can't", "won't",
    "quit", "quitting", "fired", "deleted", "delete", "killed", "kills", "kill",
    "replaced", "replaces", "cut", "beat", "free", "paid", "charged", "hired",
    "saved", "first",
    "banned", "illegal", "worst", "hate", "hated", "wasted", "waste", "scam",
    "lie", "lied", "lying", "truth", "secret", "hidden", "stole", "stolen",
    "before", "until", "instead", "but", "except", "unless", "problem",
    "risk", "danger", "warning", "regret", "wish", "should", "shouldn't",
    "still", "already", "only", "without", "versus", "vs", "actually",
}

# Openers that spend the first second saying nothing.
WEAK_OPENERS = [
    "so", "ok", "okay", "hey", "hi", "hello", "guys", "yo", "alright",
    "welcome", "today", "basically", "honestly", "look", "listen", "um",
    "just", "let", "lets", "let's", "i wanted", "i want", "one of",
    "have you", "did you", "do you", "are you", "in this", "in today",
    "the thing", "a lot", "there is", "there are", "this is", "it is",
    "as a", "when it", "if you've", "you know",
]

# Imperatives that earn the front position.
IMPERATIVES = {
    "stop", "steal", "copy", "delete", "try", "watch", "read", "save",
    "use", "build", "make", "write", "send", "take", "start", "quit",
    "never", "always", "don't", "dont", "do", "put", "run", "check",
}

DEALBREAKERS = [
    (re.compile(r"(?i)^\s*(?:stop scrolling|don'?t scroll)"),
     "Opens with \"stop scrolling\". Asking for attention proves you have not earned it."),
    (re.compile(r"(?i)\b(?:in (?:this|today'?s) (?:video|reel)|i'?m going to show you|i'?ll show you how)\b"),
     "Video preamble. Delete it and open on the payoff."),
    (re.compile(r"(?i)^\s*(?:hey |hi |what'?s up |welcome )"),
     "Greeting. Nobody came to the feed to be greeted."),
    (HASHTAG_RE,
     "Hashtag in the hook. Hashtags belong at the bottom of the caption, if anywhere."),
    (EMOJI_RE,
     "Emoji in the hook. On-screen text at hook size has room for words or for an emoji, not both."),
]

# --- Spanish -----------------------------------------------------------------
# Same five properties, Spanish vocabulary. Kept as separate sets so English
# scoring is unchanged: "yo" is a weak opener in English and a strong one in
# Spanish ("Yo perdí 2 millones..."), and "no" is an imperative only in Spanish.

ES_SPOKEN_NUMBERS = {
    "cero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "quince", "veinte", "treinta", "cuarenta",
    "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien", "ciento",
    "cientos", "mil", "miles", "millón", "millon", "millones", "docena",
    "mitad", "doble", "triple",
}
ES_MONEY_WORDS = {
    "pesos", "peso", "dólares", "dolares", "dólar", "dolar", "lucas", "luca",
    "palos", "uf", "porciento", "centavos", "sueldo", "arriendo", "ventas",
    "ganancia", "ganancias", "facturé", "facture", "factura", "boleta",
    "millonario", "precio", "plata", "gratis",
}
ES_STAKES = {
    "para", "deja", "dejes", "nunca", "jamás", "jamas", "mal", "error",
    "errores", "perdí", "perdi", "perder", "pierdes", "perdiendo", "perdiste",
    "cuesta", "costó", "costo", "caro", "quiebra", "quebré", "falla", "fallo",
    "falló", "fracaso", "fracasé", "nadie", "no", "ni", "sin", "renuncié",
    "renuncia", "despedido", "despidieron", "borré", "borra", "borrar",
    "elimina", "eliminé", "mata", "mató", "reemplazó", "reemplaza",
    "reemplazar", "gratis", "pagué", "pague", "pagas", "pagando", "cobran",
    "cobré", "cobra", "ahorré", "ahorra", "ahorrar", "primer", "primera",
    "prohibido", "ilegal", "peor", "odio", "odié", "botaste", "botando",
    "estafa", "mentira", "mentí", "verdad", "secreto", "oculto", "robaron",
    "robo", "antes", "hasta", "pero", "excepto", "salvo", "problema",
    "riesgo", "peligro", "ojo", "cuidado", "alerta", "arrepiento",
    "debería", "deberia", "deberías", "todavía", "todavia", "aún", "ya",
    "solo", "sólo", "vs", "versus", "realmente", "cayó", "cae", "caído",
    "caída", "hackearon", "hackeo", "vencido", "vence", "bloquearon",
}
ES_WEAK_OPENERS = [
    "bueno", "ok", "okey", "hola", "chicos", "chiquillos", "amigos", "buenas",
    "bienvenidos", "bienvenidas", "hoy", "básicamente", "basicamente",
    "honestamente", "mira", "miren", "oye", "eh", "este", "entonces", "solo",
    "quería", "queria", "quiero", "una de", "uno de", "has visto",
    "sabías", "sabias", "sabes", "en este", "en el video", "en el día",
    "lo que", "la cosa", "hay", "esto es", "este es", "como", "cuando se",
]
ES_IMPERATIVES = {
    "deja", "para", "copia", "roba", "borra", "elimina", "prueba", "mira",
    "lee", "guarda", "usa", "arma", "haz", "escribe", "manda", "envía",
    "toma", "empieza", "revisa", "nunca", "siempre", "no", "pon", "cambia",
    "olvida", "evita", "aprende", "compra", "cobra", "sube", "baja",
    "activa", "desactiva", "configura", "instala", "renueva",
}
ES_ADDRESS_RE = re.compile(
    r"\b(tú|tu|tus|te|ti|contigo|usted|ustedes|vos|tienes|sabes|puedes|"
    r"necesitas|quieres|estás|estas|eres|haces|pagas|pierdes|vendes|"
    r"cobras|tenés|sabés|podés)\b")
ES_FIRST_RE = re.compile(r"\b(yo|mi|mis|me|nosotros|nos|nuestro|nuestra|nuestros|nuestras)\b")

ES_DEALBREAKERS = [
    (re.compile(r"(?i)^\s*(?:deja de (?:scrollear|deslizar|pasar)|no (?:sigas|pases|scrollees))"),
     "Abre con \"deja de scrollear\". Pedir atención demuestra que no te la ganaste."),
    (re.compile(r"(?i)\b(?:en (?:este|el) (?:video|reel) de hoy|en este (?:video|reel)|"
                r"(?:hoy )?te voy a (?:mostrar|enseñar)|les voy a (?:mostrar|enseñar))\b"),
     "Preámbulo de video. Bórralo y abre con lo que importa."),
    (re.compile(r"(?i)^\s*(?:hola|buenas|qué tal|que tal|bienvenid[oa]s)\b"),
     "Saludo. Nadie entró al feed para que lo saludaran."),
]

ES_HINTS = {
    "el", "la", "los", "las", "un", "una", "que", "de", "del", "y", "en",
    "es", "por", "para", "con", "tu", "te", "mi", "lo", "se", "su", "pero",
    "nadie", "cómo", "como", "qué", "porque", "sin", "más", "mas", "yo",
    "esto", "este", "esta", "tus", "muy", "hay", "ya", "todo",
}
EN_HINTS = {
    "the", "a", "an", "and", "of", "to", "is", "in", "for", "with", "you",
    "your", "my", "it", "this", "that", "on", "but", "nobody", "how", "what",
    "i", "me", "be", "are", "was", "not", "don't", "just", "do",
}


def lang_of(text):
    """'es' or 'en', from function words and Spanish-only characters."""
    low = [x.lower().strip("'’") for x in words(text)]
    es = sum(1 for x in low if x in ES_HINTS) + 2 * len(re.findall(r"[ñ¿¡áéíóú]", text.lower()))
    en = sum(1 for x in low if x in EN_HINTS)
    return "es" if es > en else "en"


def vocab(lang):
    if lang == "es":
        return {
            "spoken": SPOKEN_NUMBERS | ES_SPOKEN_NUMBERS,
            "money": MONEY_WORDS | ES_MONEY_WORDS,
            "stakes": ES_STAKES,
            "weak": ES_WEAK_OPENERS,
            "imperatives": ES_IMPERATIVES,
            "dealbreakers": DEALBREAKERS + ES_DEALBREAKERS,
        }
    return {
        "spoken": SPOKEN_NUMBERS, "money": MONEY_WORDS, "stakes": STAKES,
        "weak": WEAK_OPENERS, "imperatives": IMPERATIVES,
        "dealbreakers": DEALBREAKERS,
    }


def clamp(n):
    return max(0.0, min(100.0, n))


def words(text):
    # "$18,000" is one word when it is spoken, so it is one word here too.
    # Spanish writes thousands with a dot: "$18.000" is one word too.
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    return WORD_RE.findall(re.sub(r"(?<=\d),(?=\d)", "", text))


def check_length(text):
    """A hook has to land before the thumb moves. Roughly two seconds."""
    w = words(text)
    n = len(w)
    secs = n / 2.75                      # ~165 words per minute, spoken
    chars = len(text.strip())
    if 5 <= n <= 12:
        score = 100.0
    elif n < 5:
        score = clamp(100 - (5 - n) * 20)
    else:
        score = clamp(100 - (n - 12) * 11)
    if chars > 60:                       # two lines of big on-screen text
        score -= 12
    return clamp(score), f"{n} words, {chars} chars, ~{secs:.1f}s spoken (want 5-12 words)"


def check_specificity(text, v):
    """One concrete thing beats three abstract ones."""
    nums = [n.strip() for n in NUMBER_RE.findall(text) if n.strip()]
    propers = set(PROPER_RE.findall(text))
    low = [w.lower().strip("'’") for w in words(text)]
    spoken = [w for w in low if w in v["spoken"] or w in v["money"]]
    hits = len(nums) + len(propers) + len(spoken)
    score = 15.0 if hits == 0 else clamp(45 + hits * 30)
    found = ", ".join(nums[:2] + sorted(propers)[:2] + spoken[:2])
    return score, (f"{hits} concrete marker(s)" + (f": {found}" if found else
                   " - no number, no name, nothing checkable"))


def check_stakes(text, v):
    """Tension, cost, negation. Something the viewer might lose."""
    w = [x.lower().strip("'’") for x in words(text)]
    hits = [x for x in w if x in v["stakes"]]
    markers = sorted(set(hits))
    if re.search(r"\$\s?\d", text):
        markers.append("a price")
    n = len(markers)
    score = {0: 20.0, 1: 70.0}.get(n, 100.0)
    detail = f"{n} tension marker(s)" + (f": {', '.join(markers[:4])}" if markers else
                                         " - nothing is at stake in this line")
    return clamp(score), detail


def check_frontload(text, v):
    """The interesting word cannot be in position nine."""
    w = words(text)
    if not w:
        return 0.0, "empty"
    low = [x.lower().strip("'’") for x in w]
    opener = " ".join(low[:2])
    penalty = 0
    hit_opener = None
    for weak in v["weak"]:
        if opener.startswith(weak) or low[0] == weak:
            penalty, hit_opener = 30, weak
            break
    payload = None
    for i, token in enumerate(low):
        if (token in v["stakes"] or token in v["spoken"] or token in v["money"]
                or NUMBER_RE.match(w[i]) or (i and PROPER_RE.match(w[i]))):
            payload = i
            break
    if payload is None:
        base = 30.0
        where = "no payload word anywhere in the line"
    elif payload <= 3:
        base = 100.0
        where = f"payload at word {payload + 1}"
    elif payload <= 6:
        base = 70.0
        where = f"payload at word {payload + 1}, could move forward"
    else:
        base = 40.0
        where = f"payload at word {payload + 1}, too late"
    detail = where + (f"; weak opener \"{hit_opener}\"" if hit_opener else "")
    return clamp(base - penalty), detail


def check_address(text, v, lang="en"):
    """Aimed at one viewer, or floating in the air."""
    low = text.lower()
    w = [x.lower().strip("'’") for x in words(text)]
    if re.search(r"\b(you|your|you're|youre|yourself)\b", low) or (
            lang == "es" and ES_ADDRESS_RE.search(low)):
        return 100.0, "speaks to the viewer"
    if w and w[0] in v["imperatives"]:
        return 90.0, f"imperative opener (\"{w[0]}\")"
    if re.search(r"\b(i|my|me|we|our)\b", low) or (
            lang == "es" and ES_FIRST_RE.search(low)):
        return 70.0, "first person, no viewer named"
    return 35.0, "third person, nobody in the room"


CHECKS = ["LENGTH", "SPECIFICITY", "STAKES", "FRONTLOAD", "ADDRESS"]


def run(text, lang=None):
    lang = lang or lang_of(text)
    v = vocab(lang)
    results = {
        "LENGTH": check_length(text),
        "SPECIFICITY": check_specificity(text, v),
        "STAKES": check_stakes(text, v),
        "FRONTLOAD": check_frontload(text, v),
        "ADDRESS": check_address(text, v, lang),
    }
    flags = [msg for pattern, msg in v["dealbreakers"] if pattern.search(text)]
    scores = [results[c][0] for c in CHECKS]
    # The weakest property caps the hook, same logic as detect.py: one bad
    # property is enough for the thumb to keep moving.
    overall = statistics.mean(scores) * 0.6 + min(scores) * 0.4 - len(flags) * 15
    overall = clamp(overall)
    verdict = "STRONG" if overall >= 70 and min(scores) >= 55 and not flags else (
        "OK" if overall >= 50 else "WEAK")
    return results, overall, verdict, flags


def bar(score, width=24):
    filled = round(score / 100 * width)
    return "#" * filled + "." * (width - filled)


def render_one(text, results, overall, verdict, flags, out=sys.stdout):
    print("\nHOOK SCORE", file=out)
    print("=" * 62, file=out)
    print(f"  \"{text.strip()}\"\n", file=out)
    for name in CHECKS:
        score, detail = results[name]
        print(f"  {name:<13} {bar(score)} {score:5.1f}", file=out)
        print(f"  {'':<13} {detail}", file=out)
    print("-" * 62, file=out)
    print(f"  {'HOOK SCORE':<13} {bar(overall)} {overall:5.1f}   {verdict}", file=out)
    for f in flags:
        print(f"\n  DEALBREAKER  {f}", file=out)
    if verdict != "STRONG":
        weakest = min(CHECKS, key=lambda c: results[c][0])
        print(f"\n  Weakest property: {weakest}. Fix that one and re-run.", file=out)
    print("", file=out)


def render_table(rows, out=sys.stdout):
    print("\nHOOK RANKING\n" + "=" * 78, file=out)
    for i, r in enumerate(rows, 1):
        mark = "->" if i == 1 else "  "
        hook = r["hook"] if len(r["hook"]) <= 62 else r["hook"][:59] + "..."
        print(f"{mark} {r['score']:5.1f} {r['verdict']:<7} {hook}", file=out)
        print(f"        weakest: {r['weakest']} ({r['checks'][r['weakest']]['score']:.0f})", file=out)
        for f in r["flags"]:
            print(f"        dealbreaker: {f}", file=out)
    print("\nShoot the top one. If the top one is under 50, none of these are the hook.\n",
          file=out)


def main():
    ap = argparse.ArgumentParser(description="Score a Reel hook on five properties.")
    ap.add_argument("input", nargs="?", default="-", help="file with one hook per line, or -")
    ap.add_argument("--hook", help="score a single hook given on the command line")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lang", choices=["auto", "en", "es"], default="auto",
                    help="vocabulary to score with (default: detect per hook)")
    args = ap.parse_args()
    lang = None if args.lang == "auto" else args.lang

    if args.hook:
        lines = [args.hook]
    else:
        raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        print("nothing to score", file=sys.stderr)
        sys.exit(2)

    payload = []
    for line in lines:
        results, overall, verdict, flags = run(line, lang)
        payload.append({
            "hook": line,
            "checks": {k: {"score": round(v[0], 1), "detail": v[1]} for k, v in results.items()},
            "weakest": min(CHECKS, key=lambda c: results[c][0]),
            "flags": flags,
            "score": round(overall, 1),
            "verdict": verdict,
        })

    if args.json:
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2, ensure_ascii=False))
        return

    if len(payload) == 1:
        results, overall, verdict, flags = run(lines[0], lang)
        render_one(lines[0], results, overall, verdict, flags)
    else:
        render_table(sorted(payload, key=lambda r: -r["score"]))

    sys.exit(0 if max(p["score"] for p in payload) >= 70 else 1)


if __name__ == "__main__":
    main()
