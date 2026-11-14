"""
Output format examples for translation prompts.

These examples show expected output format in different target languages.
Uses the same sentence: "In this quiet night, the moonlight bathed the entire valley."
"""

from typing import Dict
from .constants import TAG0, TAG1

# Single sentence translated into all supported languages
# Format: "In this quiet night, the [TAG0]moonlight[TAG1] bathed the entire valley."
OUTPUT_FORMAT_EXAMPLES: Dict[str, Dict[str, str]] = {
    # ============================================================
    # GERMANIC LANGUAGES
    # ============================================================
    "english": {
        "plain": "In this quiet night, the moonlight bathed the entire valley.",
        "standard": f"In this quiet night, the {TAG0}moonlight{TAG1} bathed the entire valley.",
    },
    "german": {
        "plain": "In dieser ruhigen Nacht badete das Mondlicht das ganze Tal.",
        "standard": f"In dieser ruhigen Nacht badete {TAG0}das Mondlicht{TAG1} das ganze Tal.",
    },
    "dutch": {
        "plain": "In deze rustige nacht baadde het maanlicht de hele vallei.",
        "standard": f"In deze rustige nacht baadde {TAG0}het maanlicht{TAG1} de hele vallei.",
    },
    "swedish": {
        "plain": "I denna tysta natt badade månljuset hela dalen.",
        "standard": f"I denna tysta natt badade {TAG0}månljuset{TAG1} hela dalen.",
    },
    "norwegian": {
        "plain": "I denne stille natten badet månelyset hele dalen.",
        "standard": f"I denne stille natten badet {TAG0}månelyset{TAG1} hele dalen.",
    },
    "danish": {
        "plain": "I denne stille nat badede månelyset hele dalen.",
        "standard": f"I denne stille nat badede {TAG0}månelyset{TAG1} hele dalen.",
    },

    # ============================================================
    # ROMANCE LANGUAGES
    # ============================================================
    "spanish": {
        "plain": "En esta noche tranquila, la luz de la luna bañaba todo el valle.",
        "standard": f"En esta noche tranquila, {TAG0}la luz de la luna{TAG1} bañaba todo el valle.",
    },
    "french": {
        "plain": "Dans cette nuit paisible, le clair de lune baignait toute la vallée.",
        "standard": f"Dans cette nuit paisible, {TAG0}le clair de lune{TAG1} baignait toute la vallée.",
    },
    "portuguese": {
        "plain": "Nesta noite tranquila, o luar banhava todo o vale.",
        "standard": f"Nesta noite tranquila, {TAG0}o luar{TAG1} banhava todo o vale.",
    },
    "italian": {
        "plain": "In questa notte tranquilla, il chiaro di luna bagnava tutta la valle.",
        "standard": f"In questa notte tranquilla, {TAG0}il chiaro di luna{TAG1} bagnava tutta la valle.",
    },
    "romanian": {
        "plain": "În această noapte liniștită, lumina lunii scălda întreaga vale.",
        "standard": f"În această noapte liniștită, {TAG0}lumina lunii{TAG1} scălda întreaga vale.",
    },
    "catalan": {
        "plain": "En aquesta nit tranquil·la, la llum de la lluna banyava tota la vall.",
        "standard": f"En aquesta nit tranquil·la, {TAG0}la llum de la lluna{TAG1} banyava tota la vall.",
    },

    # ============================================================
    # SLAVIC LANGUAGES
    # ============================================================
    "russian": {
        "plain": "В эту тихую ночь лунный свет заливал всю долину.",
        "standard": f"В эту тихую ночь {TAG0}лунный свет{TAG1} заливал всю долину.",
    },
    "ukrainian": {
        "plain": "У цю тиху ніч місячне світло заливало всю долину.",
        "standard": f"У цю тиху ніч {TAG0}місячне світло{TAG1} заливало всю долину.",
    },
    "polish": {
        "plain": "W tę cichą noc światło księżyca zalało całą dolinę.",
        "standard": f"W tę cichą noc {TAG0}światło księżyca{TAG1} zalało całą dolinę.",
    },
    "czech": {
        "plain": "V tuto tichou noc měsíční světlo zalilo celé údolí.",
        "standard": f"V tuto tichou noc {TAG0}měsíční světlo{TAG1} zalilo celé údolí.",
    },
    "slovak": {
        "plain": "V túto tichú noc mesačné svetlo zalievalo celé údolie.",
        "standard": f"V túto tichú noc {TAG0}mesačné svetlo{TAG1} zalievalo celé údolie.",
    },
    "serbian": {
        "plain": "U ovoj tihoj noći mesečina je okupala celu dolinu.",
        "standard": f"U ovoj tihoj noći {TAG0}mesečina{TAG1} je okupala celu dolinu.",
    },
    "croatian": {
        "plain": "U ovoj tihoj noći mjesečina je okupala cijelu dolinu.",
        "standard": f"U ovoj tihoj noći {TAG0}mjesečina{TAG1} je okupala cijelu dolinu.",
    },
    "bulgarian": {
        "plain": "В тази тиха нощ лунната светлина обливаше цялата долина.",
        "standard": f"В тази тиха нощ {TAG0}лунната светлина{TAG1} обливаше цялата долина.",
    },

    # ============================================================
    # EAST ASIAN LANGUAGES
    # ============================================================
    "chinese": {
        "plain": "在这个宁静的夜晚，月光洒满了整个山谷。",
        "standard": f"在这个宁静的夜晚，{TAG0}月光{TAG1}洒满了整个山谷。",
    },
    "japanese": {
        "plain": "この静かな夜、月光が谷全体を照らしていました。",
        "standard": f"この静かな夜、{TAG0}月光{TAG1}が谷全体を照らしていました。",
    },
    "korean": {
        "plain": "이 고요한 밤, 달빛이 온 계곡을 비추고 있었습니다.",
        "standard": f"이 고요한 밤, {TAG0}달빛{TAG1}이 온 계곡을 비추고 있었습니다.",
    },

    # ============================================================
    # SOUTH ASIAN LANGUAGES
    # ============================================================
    "hindi": {
        "plain": "इस शांत रात में, चांदनी पूरी घाटी को नहला रही थी।",
        "standard": f"इस शांत रात में, {TAG0}चांदनी{TAG1} पूरी घाटी को नहला रही थी।",
    },
    "bengali": {
        "plain": "এই শান্ত রাতে, চাঁদের আলো পুরো উপত্যকা জুড়ে ছড়িয়ে পড়েছিল।",
        "standard": f"এই শান্ত রাতে, {TAG0}চাঁদের আলো{TAG1} পুরো উপত্যকা জুড়ে ছড়িয়ে পড়েছিল।",
    },
    "urdu": {
        "plain": "اس پرسکون رات میں، چاندنی نے پوری وادی کو نہلا دیا۔",
        "standard": f"اس پرسکون رات میں، {TAG0}چاندنی{TAG1} نے پوری وادی کو نہلا دیا۔",
    },
    "punjabi": {
        "plain": "ਇਸ ਸ਼ਾਂਤ ਰਾਤ ਵਿੱਚ, ਚੰਦਰਮਾ ਦੀ ਰੌਸ਼ਨੀ ਨੇ ਸਾਰੀ ਵਾਦੀ ਨੂੰ ਨਹਾ ਦਿੱਤਾ।",
        "standard": f"ਇਸ ਸ਼ਾਂਤ ਰਾਤ ਵਿੱਚ, {TAG0}ਚੰਦਰਮਾ ਦੀ ਰੌਸ਼ਨੀ{TAG1} ਨੇ ਸਾਰੀ ਵਾਦੀ ਨੂੰ ਨਹਾ ਦਿੱਤਾ।",
    },
    "tamil": {
        "plain": "இந்த அமைதியான இரவில், நிலவொளி முழு பள்ளத்தாக்கையும் குளிப்பாட்டியது.",
        "standard": f"இந்த அமைதியான இரவில், {TAG0}நிலவொளி{TAG1} முழு பள்ளத்தாக்கையும் குளிப்பாட்டியது.",
    },
    "telugu": {
        "plain": "ఈ ప్రశాంతమైన రాత్రిలో, వెన్నెల మొత్తం లోయను స్నానం చేయించింది.",
        "standard": f"ఈ ప్రశాంతమైన రాత్రిలో, {TAG0}వెన్నెల{TAG1} మొత్తం లోయను స్నానం చేయించింది.",
    },
    "marathi": {
        "plain": "या शांत रात्री, चंद्रप्रकाशाने संपूर्ण दरी न्हाऊन निघाली.",
        "standard": f"या शांत रात्री, {TAG0}चंद्रप्रकाशाने{TAG1} संपूर्ण दरी न्हाऊन निघाली.",
    },
    "gujarati": {
        "plain": "આ શાંત રાત્રે, ચંદ્રપ્રકાશ આખી ખીણને નવડાવી રહ્યો હતો.",
        "standard": f"આ શાંત રાત્રે, {TAG0}ચંદ્રપ્રકાશ{TAG1} આખી ખીણને નવડાવી રહ્યો હતો.",
    },

    # ============================================================
    # SOUTHEAST ASIAN LANGUAGES
    # ============================================================
    "vietnamese": {
        "plain": "Trong đêm yên tĩnh này, ánh trăng tràn ngập khắp thung lũng.",
        "standard": f"Trong đêm yên tĩnh này, {TAG0}ánh trăng{TAG1} tràn ngập khắp thung lũng.",
    },
    "thai": {
        "plain": "ในคืนที่เงียบสงบนี้ แสงจันทร์ส่องสว่างไปทั่วหุบเขา",
        "standard": f"ในคืนที่เงียบสงบนี้ {TAG0}แสงจันทร์{TAG1}ส่องสว่างไปทั่วหุบเขา",
    },
    "indonesian": {
        "plain": "Di malam yang tenang ini, cahaya bulan membasahi seluruh lembah.",
        "standard": f"Di malam yang tenang ini, {TAG0}cahaya bulan{TAG1} membasahi seluruh lembah.",
    },
    "malay": {
        "plain": "Pada malam yang tenang ini, cahaya bulan membasahi seluruh lembah.",
        "standard": f"Pada malam yang tenang ini, {TAG0}cahaya bulan{TAG1} membasahi seluruh lembah.",
    },
    "tagalog": {
        "plain": "Sa tahimik na gabing ito, niligo ng liwanag ng buwan ang buong lambak.",
        "standard": f"Sa tahimik na gabing ito, niligo ng {TAG0}liwanag ng buwan{TAG1} ang buong lambak.",
    },
    "burmese": {
        "plain": "ဤတိတ်ဆိတ်သောညတွင် လရောင်သည် တောင်ကြားတစ်ခုလုံးကို ရေချိုးပေးခဲ့သည်။",
        "standard": f"ဤတိတ်ဆိတ်သောညတွင် {TAG0}လရောင်{TAG1}သည် တောင်ကြားတစ်ခုလုံးကို ရေချိုးပေးခဲ့သည်။",
    },

    # ============================================================
    # MIDDLE EASTERN LANGUAGES
    # ============================================================
    "arabic": {
        "plain": "في هذه الليلة الهادئة، كان ضوء القمر يغمر الوادي بأكمله.",
        "standard": f"في هذه الليلة الهادئة، كان {TAG0}ضوء القمر{TAG1} يغمر الوادي بأكمله.",
    },
    "persian": {
        "plain": "در این شب آرام، نور ماه تمام دره را غرق کرده بود.",
        "standard": f"در این شب آرام، {TAG0}نور ماه{TAG1} تمام دره را غرق کرده بود.",
    },
    "turkish": {
        "plain": "Bu sakin gecede, ay ışığı tüm vadiyi aydınlatıyordu.",
        "standard": f"Bu sakin gecede, {TAG0}ay ışığı{TAG1} tüm vadiyi aydınlatıyordu.",
    },
    "hebrew": {
        "plain": "בלילה השקט הזה, אור הירח הציף את כל העמק.",
        "standard": f"בלילה השקט הזה, {TAG0}אור הירח{TAG1} הציף את כל העמק.",
    },

    # ============================================================
    # OTHER EUROPEAN LANGUAGES
    # ============================================================
    "greek": {
        "plain": "Σε αυτή την ήσυχη νύχτα, το φως του φεγγαριού λούζε ολόκληρη την κοιλάδα.",
        "standard": f"Σε αυτή την ήσυχη νύχτα, {TAG0}το φως του φεγγαριού{TAG1} λούζε ολόκληρη την κοιλάδα.",
    },
    "hungarian": {
        "plain": "Ezen a csendes éjszakán a holdfény fürdette az egész völgyet.",
        "standard": f"Ezen a csendes éjszakán {TAG0}a holdfény{TAG1} fürdette az egész völgyet.",
    },
    "finnish": {
        "plain": "Tänä hiljaisena yönä kuunvalo kylpi koko laakson.",
        "standard": f"Tänä hiljaisena yönä {TAG0}kuunvalo{TAG1} kylpi koko laakson.",
    },

    # ============================================================
    # AFRICAN LANGUAGES
    # ============================================================
    "swahili": {
        "plain": "Katika usiku huu wa utulivu, mwanga wa mwezi uliiosha bonde lote.",
        "standard": f"Katika usiku huu wa utulivu, {TAG0}mwanga wa mwezi{TAG1} uliiosha bonde lote.",
    },
    "amharic": {
        "plain": "በዚህ ጸጥ ባለ ምሽት፣ የጨረቃ ብርሃን መላውን ሸለቆ አጠበ።",
        "standard": f"በዚህ ጸጥ ባለ ምሽት፣ {TAG0}የጨረቃ ብርሃን{TAG1} መላውን ሸለቆ አጠበ።",
    },
}


# ============================================================
# EXAMPLE LINE SHOWN IN THE "CORRECT format" BLOCK OF A PROMPT
# ============================================================
# These are the short placeholder lines the translation and refinement prompts
# put between the output tags to show the expected shape of an answer. They are
# deliberately generic ("your text here") rather than the moonlight sentence
# above, which the prompts have never used: changing them changes model output.

# Translation prompt: "plain" is used in Plain Text Mode (no placeholders),
# "standard" when the input carries placeholders.
TRANSLATION_OUTPUT_EXAMPLES: Dict[str, Dict[str, str]] = {
    "chinese": {
        "plain": "您翻译的文本在这里",
        "standard": f"您翻译的文本在这里，所有{TAG0}标记都精确保留",
    },
    "french": {
        "plain": "Votre texte traduit ici",
        "standard": f"Votre texte traduit ici, tous les marqueurs {TAG0} sont préservés exactement",
    },
    "spanish": {
        "plain": "Su texto traducido aquí",
        "standard": f"Su texto traducido aquí, todos los marcadores {TAG0} se preservan exactamente",
    },
    "german": {
        "plain": "Ihr übersetzter Text hier",
        "standard": f"Ihr übersetzter Text hier, alle {TAG0}-Markierungen werden genau beibehalten",
    },
    "japanese": {
        "plain": "翻訳されたテキストはこちら",
        "standard": f"翻訳されたテキストはこちら、すべての{TAG0}マーカーは正確に保持されます",
    },
    "italian": {
        "plain": "Il tuo testo tradotto qui",
        "standard": f"Il tuo testo tradotto qui, tutti i marcatori {TAG0} sono conservati esattamente",
    },
    "portuguese": {
        "plain": "Seu texto traduzido aqui",
        "standard": f"Seu texto traduzido aqui, todos os marcadores {TAG0} são preservados exatamente",
    },
    "russian": {
        "plain": "Ваш переведенный текст здесь",
        "standard": f"Ваш переведенный текст здесь, все маркеры {TAG0} сохранены точно",
    },
    "korean": {
        "plain": "번역된 텍스트는 여기에",
        "standard": f"번역된 텍스트는 여기에, 모든 {TAG0} 마커는 정확히 보존됩니다",
    },
}

# Fallback for a target language with no entry above (both modes).
TRANSLATION_OUTPUT_FALLBACK = "Your translated text here"

# Refinement prompt (same line whether or not placeholders are present).
REFINEMENT_OUTPUT_EXAMPLES: Dict[str, str] = {
    "chinese": "您润色后的文本在这里",
    "french": "Votre texte affiné ici",
    "spanish": "Su texto refinado aquí",
    "german": "Ihr verfeinerter Text hier",
    "japanese": "洗練されたテキストはこちら",
    "italian": "Il tuo testo raffinato qui",
    "portuguese": "Seu texto refinado aqui",
    "russian": "Ваш улучшенный текст здесь",
    "korean": "다듬어진 텍스트는 여기에",
}

REFINEMENT_OUTPUT_FALLBACK = "Your refined text here"
