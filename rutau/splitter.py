"""Разбиваем текст нужным образом
"""
from typing import List
from razdel import sentenize


def get_stead_sent_pairs(text: str) -> List[str]:
    """Разбиваем текст на пары последовательных предложений (первое и второе)

    Args:
        text (str): Входной текст

    Returns:
        List[str]: Список пар предложений
    """
    sents = [sent.text for sent in sentenize(text)]
    sent_pairs = []
    for i in range(0, len(sents)):
        if i+1 < len(sents):
            sent_pairs.append(sents[i] + ' ' + sents[i+1])
    return sent_pairs

def get_diff_sent_pairs(text: str) -> List[str]:
    """Разбиваем текст на пары предложений: соединяем все со всеми последующими

    Args:
        text (str): Входной текст

    Returns:
        List[str]: Список пар предложений
    """
    sents = [sent.text for sent in sentenize(text)]
    sent_pairs = []
    for posA, sentA in enumerate(sents):
        for posB, sentB in enumerate(sents):
            if (sentA != sentB) and (posA < posB):
                sent_pairs.append(sentA + ' ' + sentB)
    return sent_pairs