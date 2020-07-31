"""Аугментируем тексты
"""
import os
from typing import List, Dict
import pymorphy2
from razdel import tokenize
import pickle
import random
from . import models_path, data_path, load_w2v, load_ner

with open(os.path.join(data_path, 'surnames.pickle'), 'rb') as fp:
    surnames = pickle.load(fp)
with open(os.path.join(data_path, 'names.pickle'), 'rb') as fp:
    names = pickle.load(fp)

ner = load_ner(models_path)
morph = pymorphy2.MorphAnalyzer()
w2v_model = load_w2v(models_path)

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

def select_by_pos(word_list: List[str], pos: str) -> List[str]:
    """Вспомогательный метод: выбирает из списка слов только слова 
    с определённой частью речи

    Args:
        word_list (List[str]): Список слов, из которых нужно отобрать слова с определённой частью речи

        pos (str): Часть речи (Например, 'NOUN', 'ADJF' или 'VERB').

    Returns:
        List[str]: Список слов, соответствующих данной части речи

    """
    new_list = []
    for word in word_list:
        word_tag = morph.parse(word)[0].tag
        if pos == word_tag.POS:
            new_list.append(word)
    return new_list

def transform_word(word_from: str, word_to: str) -> str:
    """Преобразуем слово, чтобы оно было похоже на пример

    Args:
        word_from (str): Слово, на которое должно быть похоже преобразуемое.

        word_to (str): Слово, которое нужно преобразовать.

    Returns:
        str: Преобразованное в нужную форму слово.
    """
    tag = morph.parse(word_from)[0].tag
    word_to = morph.parse(word_to)[0]
    tagset = set()
    try:
        if tag.gender is not None:
            tagset.add(tag.gender)
        if tag.person is not None:
            tagset.add(tag.person)
        if tag.case is not None:
            tagset.add(tag.case)
        if tag.number is not None:
            tagset.add(tag.number)
        if 'Af-p' in tag:
            tagset.add('Af-p')
        result = word_to.inflect(tagset).word
    except:
        result = word_to.word
    return result

def get_similars(word: str, type: str) -> List[str]:
    """Поиск похожих слов по word2vec

    Args:
        word (str): Слово-шаблон, для которого осуществляется поиск "синонимов".

        type (str): Часть речи слова. Например, 'NOUN', 'ADJF' или 'VERB'.

    Returns:
        List[str]: Список похожих слов.
    """
    normal_form = morph.parse(word)[0].normal_form
    try:
        similars = w2v_model.most_similar(normal_form + f'_{type}')
        sim_list = [item[0].split(f'_{type}')[0] for item in similars if type in item[0]]
    except:
        sim_list = []
    return sim_list

def synonimize_text(text: str, type: List[str]) -> List[str]:
    """Метод синонимизирует текст на основе word2vec.

    Метод получает "синонимы" определённых слов и создаёт новые тексты, заменяя слова оригинального
    текста синонимами.

    Args:
        text (str): Оригинальный текст, который нужно аугментировать.

        type (List[str]): Часть речи слов, которые подвергнутся замене. Список: ['NOUN', 'ADJF', 'VERB'].

    Returns:
        List[str]: Список аугментированных текстов.
    """
    selected_words: List[str] = []
    if len(text) < 1:
        # Слишком короткий текст
        return []
    
    tokens = list(tokenize(text))
    tokens = [_.text for _ in tokens]

    new_list: List[str] = []
    counter: int = 0
    mean_len: int = 0

    # Выбираем все слова для заданной части речи
    if 'NOUN' in type:
        selected_words = select_by_pos(word_list=tokens, pos='NOUN')
        for word in selected_words:
            if 'Name' not in morph.parse(word)[0].tag:
                sim_list = get_similars(word=word, type='NOUN')
                if len(sim_list) > 0:
                    sim_list = [transform_word(word_from=word, word_to=sim) for sim in sim_list]
                    new_list.append([word, sim_list])
                    counter += 1
                    mean_len += len(sim_list)
    if 'ADJF' in type:
        selected_words = select_by_pos(word_list=tokens, pos='ADJF')
        for word in selected_words:
            sim_list = get_similars(word=word, type='ADJF')
            if len(sim_list) > 0:
                sim_list = [transform_word(word_from=word, word_to=sim) for sim in sim_list]
                new_list.append([word, sim_list])
                counter += 1
                mean_len += len(sim_list)
    if 'VERB' in type:
        selected_words = select_by_pos(word_list=tokens, pos='VERB')
        for word in selected_words:
            sim_list = get_similars(word=word, type='VERB')
            if len(sim_list) > 0:
                sim_list = [transform_word(word_from=word, word_to=sim) for sim in sim_list]
                new_list.append([word, sim_list])
                counter += 1
                mean_len += len(sim_list)
    
    if mean_len == 0:
        # Синонимов не найдено
        return []
    mean_len = int(mean_len / counter)
    
    # Генерируем новые тексты
    new_texts = [text] * mean_len
    for item in new_list:
        for num in range(0, mean_len):
            word_from = item[0]
            if len(item[1]) >= mean_len:
                word_to = item[1][num]
            else:
                word_to = random.choice(item[1])
            if word_from.istitle():
                word_to = word_to.capitalize()
            if word_from.isupper():
                word_to = word_to.upper()
            new_texts[num] = new_texts[num].replace(word_from, word_to)
    return new_texts
