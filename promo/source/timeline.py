"""Shared timeline for the promo video: cue times used by both the visuals
(written to cues.js) and the synthesized soundtrack."""
import json

BPM = 120
BEAT = 60 / BPM          # 0.5 s
BAR = 4 * BEAT           # 2 s
DURATION = 60.0
FPS = 30

# (word, language) - "book" in many languages, common first, rarer later.
WORDS = [
    ("Book", "English"), ("Livre", "French"), ("书", "Chinese"), ("كتاب", "Arabic"),
    ("本", "Japanese"), ("Libro", "Spanish"), ("책", "Korean"), ("Книга", "Russian"),
    ("पुस्तक", "Hindi"), ("Buch", "German"), ("Βιβλίο", "Greek"), ("ספר", "Hebrew"),
    ("Livro", "Portuguese"), ("หนังสือ", "Thai"), ("Kitap", "Turkish"), ("বই", "Bengali"),
    ("Sách", "Vietnamese"), ("புத்தகம்", "Tamil"), ("Książka", "Polish"), ("წიგნი", "Georgian"),
    ("Kitabu", "Swahili"), ("Գիրք", "Armenian"), ("Könyv", "Hungarian"), ("መጽሐፍ", "Amharic"),
    ("Kirja", "Finnish"), ("សៀវភៅ", "Khmer"), ("Bók", "Icelandic"), ("Ном", "Mongolian"),
    ("Llyfr", "Welsh"), ("စာအုပ်", "Burmese"), ("Liburua", "Basque"), ("ᐅᖃᓕᒫᒐᒃ", "Inuktitut"),
    ("Leabhar", "Irish"), ("පොත", "Sinhala"), ("Naaltsoos", "Navajo"), ("ປຶ້ມ", "Lao"),
    ("Pukapuka", "Māori"), ("పుస్తకం", "Telugu"), ("Ìwé", "Yoruba"),
    ("ಪುಸ್ತಕ", "Kannada"), ("Incwadi", "Zulu"), ("പുസ്തകം", "Malayalam"), ("Boky", "Malagasy"),
    ("પુસ્તક", "Gujarati"), ("Téere", "Wolof"), ("ਕਿਤਾਬ", "Punjabi"),
    ("Tusi", "Samoan"), ("בוך", "Yiddish"), ("Pirtûk", "Kurdish"), ("Liber", "Latin"),
    ("Girji", "Sámi"), ("Buug", "Somali"), ("Grāmata", "Latvian"), ("Кітап", "Kazakh"),
    ("Ktieb", "Maltese"), ("Levr", "Breton"), ("Atuagaq", "Greenlandic"), ("Littafi", "Hausa"),
    ("Aklat", "Tagalog"), ("Raamat", "Estonian"), ("Llibre", "Catalan"), ("Ivola", "Fijian"),
    ("Kitob", "Uzbek"), ("Knyga", "Lithuanian"), ("Libër", "Albanian"), ("Tohi", "Tongan"),
    ("መጽሓፍ", "Tigrinya"), ("Kniha", "Czech"), ("Boek", "Dutch"), ("Bok", "Swedish"),
]


def _grid(segments):
    out = []
    for start, end, step in segments:
        n = round((end - start) / step)
        out += [start + i * step for i in range(n)]
    return out


# Intro cascade: accelerates on the musical grid (quarter -> 32nd notes).
INTRO_TIMES = [1.0] + _grid([
    (3.0, 5.0, 0.5), (5.0, 7.0, 0.25), (7.0, 9.0, 1 / 6),
    (9.0, 10.5, 0.125), (10.5, 11.5, 0.0625),
])
INTRO_END = 11.5
# Final dark crescendo cascade.
OUTRO_TIMES = _grid([(52.0, 53.0, 0.125), (53.0, 53.75, 0.0625)])
OUTRO_END = 53.75


def cues():
    intro = [{"t": t, "w": WORDS[i % len(WORDS)][0], "l": WORDS[i % len(WORDS)][1]}
             for i, t in enumerate(INTRO_TIMES)]
    # Outro starts deep in the list so it shows other scripts.
    off = 20
    outro = [{"t": t, "w": WORDS[(off + i * 3) % len(WORDS)][0], "l": WORDS[(off + i * 3) % len(WORDS)][1]}
             for i, t in enumerate(OUTRO_TIMES)]
    return {
        "intro": intro, "introEnd": INTRO_END,
        "outro": outro, "outroEnd": OUTRO_END,
        "chips": [42.5 + i * 0.25 for i in range(16)],
    }


if __name__ == "__main__":
    c = cues()
    with open("cues.js", "w", encoding="utf-8") as f:
        f.write("window.CUES = " + json.dumps(c, ensure_ascii=False) + ";\n")
    print(len(c["intro"]), "intro words,", len(c["outro"]), "outro words,", len(WORDS), "words total")
