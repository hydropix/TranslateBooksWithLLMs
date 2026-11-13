"""
Subtitle format examples for SRT translation and refinement.

These examples show the expected format for translated subtitles, written in
the target language so the model sees the language it must answer in.
"""

from typing import Dict

SUBTITLE_EXAMPLES: Dict[str, str] = {
    "chinese": "[1]第一行翻译文本\n[2]第二行翻译文本",
    "french": "[1]Première ligne traduite\n[2]Deuxième ligne traduite",
    "spanish": "[1]Primera línea traducida\n[2]Segunda línea traducida",
    "german": "[1]Erste übersetzte Zeile\n[2]Zweite übersetzte Zeile",
    "italian": "[1]Prima riga tradotta\n[2]Seconda riga tradotta",
    "portuguese": "[1]Primeira linha traduzida\n[2]Segunda linha traduzida",
    "russian": "[1]Первая переведенная строка\n[2]Вторая переведенная строка",
    "japanese": "[1]最初の翻訳行\n[2]2番目の翻訳行",
    "korean": "[1]첫 번째 번역 줄\n[2]두 번째 번역 줄",
    "arabic": "[1]السطر الأول المترجم\n[2]السطر الثاني المترجم",
    "dutch": "[1]Eerste vertaalde regel\n[2]Tweede vertaalde regel",
    "polish": "[1]Pierwsza przetłumaczona linia\n[2]Druga przetłumaczona linia",
    "turkish": "[1]İlk çevrilmiş satır\n[2]İkinci çevrilmiş satır",
    "vietnamese": "[1]Dòng dịch đầu tiên\n[2]Dòng dịch thứ hai",
    "thai": "[1]บรรทัดแปลแรก\n[2]บรรทัดแปลที่สอง",
    "indonesian": "[1]Baris terjemahan pertama\n[2]Baris terjemahan kedua",
    "hindi": "[1]पहली अनुवादित पंक्ति\n[2]दूसरी अनुवादित पंक्ति",
}

# Neutral fallback when the target language has no entry above.
SUBTITLE_EXAMPLE_FALLBACK = "[1]First translated line\n[2]Second translated line"

# Refinement examples. The refine prompt numbers a block's subtitles from 0
# (local indices), so these start at [0], unlike the translation examples.
SUBTITLE_REFINEMENT_EXAMPLES: Dict[str, str] = {
    "chinese": "[0]第一行润色文本\n[1]第二行润色文本",
    "french": "[0]Première ligne affinée\n[1]Deuxième ligne affinée",
    "spanish": "[0]Primera línea refinada\n[1]Segunda línea refinada",
    "german": "[0]Erste überarbeitete Zeile\n[1]Zweite überarbeitete Zeile",
    "italian": "[0]Prima riga rifinita\n[1]Seconda riga rifinita",
    "portuguese": "[0]Primeira linha refinada\n[1]Segunda linha refinada",
    "russian": "[0]Первая отредактированная строка\n[1]Вторая отредактированная строка",
    "japanese": "[0]洗練された最初の行\n[1]洗練された2番目の行",
    "korean": "[0]다듬어진 첫 번째 줄\n[1]다듬어진 두 번째 줄",
    "arabic": "[0]السطر الأول المنقح\n[1]السطر الثاني المنقح",
    "dutch": "[0]Eerste verfijnde regel\n[1]Tweede verfijnde regel",
    "polish": "[0]Pierwsza dopracowana linia\n[1]Druga dopracowana linia",
    "turkish": "[0]İlk düzeltilmiş satır\n[1]İkinci düzeltilmiş satır",
    "vietnamese": "[0]Dòng đã trau chuốt đầu tiên\n[1]Dòng đã trau chuốt thứ hai",
    "thai": "[0]บรรทัดแรกที่ขัดเกลาแล้ว\n[1]บรรทัดที่สองที่ขัดเกลาแล้ว",
    "indonesian": "[0]Baris pertama yang diperhalus\n[1]Baris kedua yang diperhalus",
    "hindi": "[0]पहली परिष्कृत पंक्ति\n[1]दूसरी परिष्कृत पंक्ति",
}

# Neutral fallback when the target language has no refinement entry above.
SUBTITLE_REFINEMENT_EXAMPLE_FALLBACK = "[0]First refined line\n[1]Second refined line"
