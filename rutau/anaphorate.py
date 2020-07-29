"""Заменяем PER-сущности местоимениями
"""
import os
from typing import List, Dict
import pickle
import random
import pymorphy2
from razdel import tokenize
from slovnet import NER
from . import models_path, data_path, load_ner
from .splitter import get_stead_sent_pairs, get_diff_sent_pairs


ner = load_ner(models_path)
morph = pymorphy2.MorphAnalyzer()

with open(os.path.join(data_path, 'surnames.pickle'), 'rb') as fp:
    surnames = pickle.load(fp)
with open(os.path.join(data_path, 'names.pickle'), 'rb') as fp:
    names = pickle.load(fp)

def get_nomen(count: int, gender: str, type: str) -> List[str]:
    """Генерация случайных имён и фамилий

    Имена и фамилии были взяты из проекта Ивана Бегтина:
    https://github.com/datacoon/russiannames

    Args:
        count (int): Количество имён, которое необходимо сгенерировать.

        gender (str): Род имён (женский или мужской). Варианты: `f`, `m`.

        type (str): Тип генерируемого имени - имя собственное, либо фамилия. Варианты:
        `name`, `surname`.

    Returns:
        List[str]: Список сгенерированных имён.
    """
    lst = list(surnames.items()) if type == 'surname' else list(names.items())
    selected_items = []
    while len(selected_items) < count:
        item = random.choice(lst)
        if item[1] == gender and item[0] not in selected_items:
            selected_items.append(item[0])
    return selected_items

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

def find_pronoun_pairs(sentence: str) -> List[Dict]:
    """Метод находит в тексте 2 местоимения и связывает их вместе.

    Args:
        sentence (str): Предложение, которе следует преобразовать.

    Returns:
        List: Размеченный список (если предложение подходит под условия).

    **Пример:**

    ```
    text = 'Я пошёл на реку. Там меня за хвост поймал сом.'

    find_pronoun_pairs(text)

    > [{'anaphor': {'end': 25, 'start': 21, 'text': 'меня'},
    > 'antecedent': {'end': 1, 'start': 0, 'text': 'Я'},
    > 'text': 'Я пошёл на реку. Там меня за хвост поймал сом.'}]
    ```

    """
    if len(sentence) < 3:
        return []
    
    pronoun_pairs: List[List[str]] = [
        ['я', 'меня'], ['я', 'мой'], ['я', 'моя'], ['я', 'мои'], ['я', 'моё'],
        ['я', 'мое'], ['ты', 'тебя'],['ты', 'твой'], ['ты', 'твоя'], ['ты', 'твои'],
        ['ты', 'твоё'], ['ты', 'твое'], ['вы', 'вас'], ['вы', 'ваш'], ['вы', 'ваша'],
        ['вы', 'ваши'], ['вы', 'ваше'], ['он', 'его'], ['она', 'её'], ['она', 'ее'],
        ['они', 'их'], ['они', 'ихний'], ['они', 'ихняя'], ['они', 'ихние'],
        ['они', 'ихнее'], ['оно', 'его'],]
    tokens = list(tokenize(sentence))

    for pos, pair in enumerate(pronoun_pairs):
        found, antecedent_found, anaphor_found = False, False, False
        if set(pair).issubset(set([_.text.lower() for _ in tokens])):
            found = True
            break
        
    if found == True:
        antecedent = pronoun_pairs[pos][0]
        anaphor = pronoun_pairs[pos][1]
        for item in tokens:
            if antecedent_found == False and (item.text.lower() == antecedent):
                antecedent_found = True
                antecedent = item.text
                antecedent_start = item.start
                antecedent_end = item.stop
            if anaphor_found == False and (item.text.lower() == anaphor):
                anaphor_found = True
                anaphor = item.text
                anaphor_start = item.start
                anaphor_end = item.stop
            if antecedent_found == True and anaphor_found == True:
                return [{
                    'text': sentence,
                    'anaphor': {
                        'text': anaphor,
                        'start': anaphor_start,
                        'end': anaphor_end,
                    },
                    'antecedent': {
                        'text': antecedent,
                        'start': antecedent_start,
                        'end': antecedent_end,
                    }
                }]
    return []

def rename_antecedent(samples: List[Dict], surname: bool, count: int) -> List[Dict]:
    """Метод обогащает корпус путём замены антецедента случайными именами

    Args:
        samples (List[Dict]): Cписок уже размеченных в нашем формате данных.

        surname (bool): Использовать ли фамилию, или заменять только именами.

        count (int): Количество, которое нужно сгенерировать.

    Returns:
        List: Результирующий список, состоящий из антецедента, анафора и нового текста
    """
    if len(samples) < 1:
        return []
    
    new_texts: List = []
    for sample in samples:
        tokens = list(tokenize(sample['text']))
        # define gender
        gender = morph.parse(sample['antecedent']['text'])[0].tag.gender
        if gender == 'masc':
            gender = 'm'
        elif gender == 'neut':
            gender = 'm'
        elif gender == 'femn':
            gender = 'f'
        else:
            pointer = ''
            for token in tokens:
                if token.start == sample['antecedent']['end']+1:
                    pointer = token.text
            if len(pointer) > 0:
                gender = 'f' if morph.parse(pointer)[0].tag.gender == 'femn' else 'm'
        # get a list of names
        new_antecedents = get_nomen(count=count, gender=gender, type='name')
        if surname == True:
            new_antecedents = [item + ' ' + get_nomen(
                count=1,
                gender=gender,
                type='surname')[0] for item in new_antecedents]
        # replace antecedents with new names
        for item in new_antecedents:
            offset = len(sample['antecedent']['text']) - len(item)
            if sample['anaphor']['start'] > sample['antecedent']['start']:
                #needs offset
                new_anaphor_start = sample['anaphor']['start'] - offset
                new_anaphor_end =  sample['anaphor']['end'] - offset
            else:
                new_anaphor_start = sample['anaphor']['start']
                new_anaphor_end =  sample['anaphor']['end']
            new_antecedent_text = item
            new_text = sample['text'][:sample['antecedent']['start']] \
            + item + sample['text'][sample['antecedent']['end']:]
            new_antecedent_end = sample['antecedent']['end'] - offset
            new_antecedent_start = sample['antecedent']['start']

            new_texts.append({
                'text': new_text,
                'antecedent': {
                    'text': new_antecedent_text,
                    'start': new_antecedent_start,
                    'end': new_antecedent_end,
                },
                'anaphor': {
                    'text': sample['anaphor']['text'],
                    'start': new_anaphor_start,
                    'end': new_anaphor_end,
                },
            })
    return new_texts