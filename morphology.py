# -*- coding: utf-8 -*-
"""
Морфологический помощник: склонение русских слов по падежам программно,
через pymorphy3, вместо ручного перебора словоформ в шаблонах.

Зачем: три раунда подряд правки data_generator.py показали потолок ручного
подхода — даже когда добавляешь конкретную недостающую формулировку
("покоящимся" vs "покоящееся"), тексты слишком разнообразны по падежным
формам, чтобы перечислить их все руками. pymorphy3 умеет склонять слово в
любой из 6 падежей programmatically — это даёт комбинаторное разнообразие
словоформ без необходимости печатать каждую вручную.

ВАЖНО: pymorphy3 недоступен в песочнице (нет интернета на pip install).
Функция decline() имеет безопасный откат — если библиотека не установлена
или слово не распозналось, возвращает слово как есть, без изменений (не
ломает генерацию, просто не даёт разнообразия). Саму работу склонения
нужно проверить на твоей машине.

Установка:
    pip install pymorphy3 pymorphy3-dicts-ru
"""

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    HAS_MORPHOLOGY = True
except ImportError:
    _morph = None
    HAS_MORPHOLOGY = False

# Коды падежей pymorphy3 (OpenCorpora tagset)
NOMN = "nomn"  # именительный (кто? что?)
GENT = "gent"  # родительный (кого? чего?)
DATV = "datv"  # дательный (кому? чему?)
ACCS = "accs"  # винительный (кого? что?)
ABLT = "ablt"  # творительный (кем? чем?)
LOCT = "loct"  # предложный (о ком? о чём?)

ALL_CASES = [NOMN, GENT, DATV, ACCS, ABLT, LOCT]


def decline(word: str, case: str, plural: bool = False) -> str:
    """
    Склоняет слово в заданный падеж, сохраняя число/род по умолчанию.
    Безопасный откат: если pymorphy3 недоступен или слово не распознано —
    возвращает исходное слово без изменений (генерация не ломается, просто
    не получает дополнительного разнообразия словоформ).
    """
    if not HAS_MORPHOLOGY:
        return word
    parsed = _morph.parse(word)
    if not parsed:
        return word
    grammemes = {case}
    if plural:
        grammemes.add("plur")
    inflected = parsed[0].inflect(grammemes)
    return inflected.word if inflected is not None else word


def decline_phrase(words: list, case: str, plural: bool = False) -> str:
    """Склоняет согласованное словосочетание (прилагательное/причастие +
    существительное) в один и тот же падеж — важно для согласования
    ("покоящееся тело" -> "покоящимся телом", оба слова меняются вместе)."""
    return " ".join(decline(w, case, plural) for w in words)


if __name__ == "__main__":
    # Быстрая самопроверка при наличии pymorphy3
    if not HAS_MORPHOLOGY:
        print("pymorphy3 не установлен — decline() работает в безопасном "
              "режиме отката (возвращает слово без изменений).")
    else:
        print("Склонение 'покоящееся тело' по падежам:")
        for case in ALL_CASES:
            print(f"  {case}: {decline_phrase(['покоящееся', 'тело'], case)}")
        print("\nСклонение 'заряд' по падежам (мн. число):")
        for case in ALL_CASES:
            print(f"  {case}: {decline('заряд', case, plural=True)}")