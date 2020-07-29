"""Заменяем PER-сущности местоимениями
"""
import os
from typing import List, Dict
import pymorphy2
from slovnet import NER
from . import models_path, load_ner
from .splitter import get_stead_sent_pairs, get_diff_sent_pairs


ner = load_ner(models_path)
morph = pymorphy2.MorphAnalyzer()

def get_pronoun(word: str) -> str:
    """Получаем местоимение из имени

    Args:
        word (str): Имя собственное, которое нужно преобразовать в местоимение

    Returns:
        str: местоимение
    """
    tag = morph.parse(word)[0].tag
    pronoun = morph.parse('она')[0] if tag.gender == 'femn' else morph.parse('он')[0]
    tagset = set()
    if tag.case is not None:
        tagset.add(tag.case)
    # if tag.number is not None:
    #     tagset.add(tag.number)
    if 'Af-p' in tag:
        tagset.add('Af-p')
    return pronoun.inflect(tagset).word

def anaphorate_names(sentence: str) -> List[Dict]:
    """Получем новые тексты путём замены имён местоимениями

    Args:
        text (str): Входной текст

    Returns:
        List: Результирующий список, состоящий из антецедента, анафора и нового текста

    Пример возвращаемого списка:
    
    ```
    [
    {'anaphor':
        {
        'end': 25,
        'start': 23,
        'text': 'Он'},
    'antecedent':
        {
        'end': 7,
        'start': 0,
        'text': 'Василий'},
    'text': 'Василий купил ботинки. Он продал ботинки.'},
    {'anaphor': 
        {
        'end': 2,
        'start': 0, 
        'text': 'Он'},
    'antecedent':
        {
        'end': 34,
        'start': 18,
        'text': 'Василий Иванович'},
    'text': 'Он купил ботинки. Василий Иванович продал ботинки.'}]
    ```

    """
    new_texts: List = []
    per_items: List = []
    ner_tokens = ner(sentence).spans
    for span in ner_tokens:
        if span.type == 'PER':
            per_items.append(span)
    if len(per_items) >= 2:
        # выбираем всех кандидатов
        candidates = []
        for item in per_items:
            candidates.append({
                'text': sentence[item.start:item.stop],
                'lemma': morph.parse(sentence[item.start:item.stop])[0].normal_form,
                'type': item.type,
                'start': item.start,
                'stop': item.stop,
            })
        # оставляем только нужные пары
        candidates_selected = []
        for itemA in candidates:
            for itemB in candidates:
                if (itemA['lemma'] in itemB['lemma'] or itemB['lemma'] in itemA['lemma']) \
                and (itemA['start'] != itemB['start']):
                    candidates_selected.append({
                        'itemA': itemA,
                        'itemB': {
                            'lemma': itemB['lemma'],
                            'start': itemB['start'],
                            'stop': itemB['stop'],
                            'text': itemB['text'],
                            'type': itemB['type'],
                            'anaphor': get_pronoun(itemB['text'].split(' ')[0]),
                        },
                    })
        # Формируем сэмплы из отобранных пар кандидатов: новый текст, антецедент, анафор
        for candidate in candidates_selected:
            if sentence[candidate['itemB']['start'] - 2] in '.!?' or candidate['itemB']['start'] == 0:
                candidate['itemB']['anaphor'] = candidate['itemB']['anaphor'].capitalize()
            new_text = sentence[:candidate['itemB']['start']] + candidate['itemB']['anaphor'] + sentence[candidate['itemB']['stop']:]
            if candidate['itemA']['start'] < candidate['itemB']['start']:
                # Текст начинается с антецедента, смещение не нужно:
                antecedent = {
                    'text': candidate['itemA']['text'],
                    'start': candidate['itemA']['start'],
                    'end': candidate['itemA']['stop'],
                }
                anaphor = {
                    'text': candidate['itemB']['anaphor'],
                    'start': candidate['itemB']['start'],
                    'end': candidate['itemB']['start'] + len(candidate['itemB']['anaphor']),
                }
            else:
                # Текст начинается с анафора, значит нужно смещение антецедента
                anaphor = {
                    'text': candidate['itemB']['anaphor'],
                    'start': candidate['itemB']['start'],
                    'end': candidate['itemB']['start'] + len(candidate['itemB']['anaphor']),
                }
                offset = len(candidate['itemB']['text']) - len(candidate['itemB']['anaphor'])
                antecedent = {
                    'text': candidate['itemA']['text'],
                    'start': candidate['itemA']['start'] - offset,
                    'end': candidate['itemA']['stop'] - offset,
                }
            new_texts.append({
                'text': new_text,
                'antecedent': antecedent,
                'anaphor': anaphor,
            })
    return new_texts

def anaphorate(text: str, sent_splitting: str, anaph_type: List[str]) -> List[Dict]:
    """Метод создаёт из текста корпус текстов для разрешения анафоры.

    Args:
        text (str): Входной текст

        sent_splitting (str): Метод создания пар предложений:
            * `steadily` - пары последовательных предложений (первое и второе)
            * `differently` - пары предложений: соединяем все со всеми последующими

        anaph_type (list): Из каких сущностей создавать корпус.
            * `PER` - имена людей (есть)
            * `LOC` - названия локаций (будет)
            * `ORG` - наименования организаций (будет)

    Returns:
        List: Результирующий список, состоящий из антецедента, анафора и нового текста
    """
    corpus: List = []
    if sent_splitting == 'steadily':
        sentences = get_stead_sent_pairs(text)
    if sent_splitting == 'differently':
        sentences = get_diff_sent_pairs(text)
    
    for sentence in sentences:
        if 'LOC' in anaph_type:
            pass
        if 'ORG' in anaph_type:
            pass
        if 'PER' in anaph_type:
            corpus += anaphorate_names(sentence)
    return corpus
