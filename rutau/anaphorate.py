from typing import List, Dict
import pymorphy2
from razdel import tokenize
from . import models_path, load_ner
from .splitter import get_stead_sent_pairs, get_diff_sent_pairs
from .synonimizers import Synonimizer

class Anaphorate:
    """Набор методов для формирования синтетического размеченного корпуса текстов
    для разрешения местоимённой анафоры для русского языка.
    """
    def __init__(self):
        self.ner = load_ner(models_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.Synonimizer = Synonimizer()
    
    
    def get_pronoun(self, word: str) -> str:
        """Получаем местоимение из имени
    
        Args:
            word (str): Имя собственное, которое нужно преобразовать в местоимение
    
        Returns:
            str: местоимение
        """
        tag = self.morph.parse(word)[0].tag
        pronoun = self.morph.parse('она')[0] if tag.gender == 'femn' else self.morph.parse('он')[0]
        tagset = set()
        if tag.case is not None:
            tagset.add(tag.case)
        # if tag.number is not None:
        #     tagset.add(tag.number)
        if 'Af-p' in tag:
            tagset.add('Af-p')
        return pronoun.inflect(tagset).word
    
    def anaphorate_sentence(self, sentence: str, ner_type: str) -> List[Dict]:
        """Получем новые тексты путём замены выявленных NER-сущностей местоимениями
    
        Args:
            text (str): Входной текст
    
            ner_type (str): Тип извлекаемой сущности: PER, LOC, ORG
    
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
        ner_tokens = self.ner(sentence).spans
        for span in ner_tokens:
            if span.type == ner_type:
                per_items.append(span)
        if len(per_items) >= 2:
            # выбираем всех кандидатов
            candidates = []
            for item in per_items:
                candidates.append({
                    'text': sentence[item.start:item.stop],
                    'lemma': self.morph.parse(sentence[item.start:item.stop])[0].normal_form,
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
                                'anaphor': self.get_pronoun(itemB['text'].split(' ')[0]),
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
    
    def anaphorate(self, text: str, sent_splitting: str, anaph_type: List[str]) -> List[Dict]:
        """Метод создаёт из текста корпус текстов для разрешения анафоры.
    
        Args:
            text (str): Входной текст
    
            sent_splitting (str): Метод создания пар предложений:
                * `steadily` - пары последовательных предложений (первое и второе)
                * `differently` - пары предложений: соединяем все со всеми последующими
    
            anaph_type (list): Из каких сущностей создавать корпус.
                * `PER` - имена людей
                * `LOC` - названия локаций
                * `ORG` - наименования организаций
    
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
                corpus += self.anaphorate_sentence(sentence, ner_type='LOC')
            if 'ORG' in anaph_type:
                corpus += self.anaphorate_sentence(sentence, ner_type='ORG')
            if 'PER' in anaph_type:
                corpus += self.anaphorate_sentence(sentence, ner_type='PER')
        return corpus
    
    def find_pronoun_pairs(self, sentence) -> List[Dict]:
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
    
        found, antecedent_found, anaphor_found = False, False, False
        for pos, pair in enumerate(pronoun_pairs):
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
    
    def rename_antecedent(self, samples: List[Dict], surname: bool, count: int) -> List[Dict]:
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
            gender = self.morph.parse(sample['antecedent']['text'])[0].tag.gender
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
                        break
                gender = 'f' if self.morph.parse(pointer)[0].tag.gender == 'femn' else 'm'
            # get a list of names
            new_antecedents = self.Synonimizer.get_nomen(count=count, gender=gender, type='name')
            if surname == True:
                new_antecedents = [item + ' ' + self.Synonimizer.get_nomen(
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
