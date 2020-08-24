from typing import List, Dict
import pymorphy2
from razdel import tokenize
import os
import random
from dataclasses import dataclass
import pymorphy2
from natasha import Doc, Segmenter, MorphVocab
from natasha import NewsEmbedding, NewsMorphTagger, NewsNERTagger


@dataclass
class CorefItem:
    token: str
    lemma: str
    ner: str
    start: int = 0
    stop: int = 0
    coref: int = -100

class Coref:
    """Находит несколько кореферентностей в тексте
    """
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
    def select_corefs(self, text):
        '''Метод извлекает из текста кореферентности на основе NER.
        '''
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        doc.tag_ner(self.ner_tagger)

        # Извлекаем леммы и ищем встречающиеся NER-сущности
        extracted_lemmas = {}
        for span in doc.spans:
            for token in span.tokens:
                if token.lemma in extracted_lemmas:
                    extracted_lemmas[token.lemma] += 1
                else:
                    extracted_lemmas[token.lemma] = 1
        selected_items = [item for item in extracted_lemmas if extracted_lemmas[item] > 1]

        # Выбираем антецеденты и упоминания
        coref_sequence = []
        for item in selected_items:
            antecedent_found = -100
            for span in doc.spans:
                for token in span.tokens:
                    if token.lemma == item:
                        if antecedent_found == -100:
                            antecedent_found = span.start
                            coref_sequence.append(CorefItem(
                                span.text,
                                token.lemma,
                                span.type,
                                span.start,
                                span.stop))
                        else:
                            coref_sequence.append(CorefItem(
                                span.text,
                                token.lemma,
                                span.type,
                                span.start,
                                span.stop, antecedent_found))

        # Обзначаем индексы токенов
        sequence = [token for token in doc.tokens]
        indexes = {}
        for item in coref_sequence:
            for i, token in enumerate(doc.tokens):
                if item.start == token.start:
                    indexes[item.start] = i
                    item.start = i
                if item.stop == token.stop:
                    item.stop = i

        for item in coref_sequence:
            if item.coref != -100:
                item.coref = indexes[item.coref]
        return sequence, coref_sequence

    def get_pronoun(self, doc_token):
        '''Преобразует слово в местоимение
        '''
        # https://pymorphy2.readthedocs.io/en/latest/user/grammemes.html#grammeme-docs
        cases = {
            'Acc': 'accs',
            'Dat': 'datv',
            'Gen': 'gent',
            'Ins': 'ablt', 
            'Loc': 'loct', 
            'Nom': 'nomn',
        }

        feats = doc_token.feats

        try:
            if 'Number' in feats and feats['Number'] == 'Plur':
                pronoun = 'они'
            elif feats['Gender'] == 'Masc':
                pronoun = 'он'
            elif feats['Gender'] == 'Fem':
                pronoun = 'она'
            else:
                pronoun = 'оно'

            tagset = set()
            tagset.add(cases['Gen'])

            pronoun = self.morph.parse(pronoun)[0]
            pronoun = pronoun.inflect(tagset).word
        except:
            pronoun = 'он'
        return pronoun
    
    def replace_with_pronouns(self, sequence, coref_sequence, shift=True):
        '''Заменяем упоминания местоимениями
        '''
        new_sequence = [token.text for token in sequence]
        start, stop, offset = 0, 0, 0
        for mention in coref_sequence:
            if mention.coref == -100:
                pass
            else:
                if mention.start == mention.stop:
                    current_start = mention.start
                    current_stop = mention.stop+1
                    pronoun = self.get_pronoun(sequence[mention.start])
                    mention.token = pronoun
                    if shift == True:
                        mention.start = mention.start - 1
                        mention.stop = mention.stop - 1
                        new_sequence = new_sequence[:current_start-1] + [pronoun] \
                        + new_sequence[current_start-1:current_start] + new_sequence[current_start+1:]
                    else:
                        new_sequence = new_sequence[:current_start] + [pronoun] + new_sequence[current_stop:]

        return new_sequence, coref_sequence
    
    def coref_to_dict(self, coref_sequence):
        ''' CorefItem -> list
        '''
        ants, ments = [], []
        coreferense = []
        for itm in coref_sequence:
            if itm.coref == -100:
                ants.append({
                    'token': itm.token,
                    'lemma': itm.lemma,
                    'start': itm.start,
                    'end': itm.stop,
                })
            else:
                ments.append({
                    'token': itm.token,
                    'lemma': itm.lemma,
                    'start': itm.start,
                    'end': itm.stop,
                    'coref': itm.coref,
                })
        for a in ants:
            coreferense.append({
                'antecedent': a,
                'mentions': [m for m in ments if m['coref'] == a['start']]
            })
        return coreferense
    
    def anaphoras_to_corpus(self, anaphoras):
        '''Конвертируем найденные анафорические связи в корпус
        '''
        for item in anaphoras:
            corpus = []
            sequence = list(tokenize(item['text']))
            for i, s in enumerate(sequence):
                if s.start == item['antecedent']['start']:
                    antecedent = {
                        'token': item['antecedent']['text'],
                        'lemma': item['antecedent']['text'],
                        'start': i,
                        'end': i,
                    }
                    break
            for i, s in enumerate(sequence):
                if s.start == item['anaphor']['start']:
                    mentions = [{
                        'token': item['anaphor']['text'],
                        'lemma': item['anaphor']['text'],
                        'start': i,
                        'end': i,
                        'coref': antecedent['start'],
                    }]
                    break
            corpus.append({
                'text': [s.text for s in sequence],
                'coreferences': [{
                    'antecedent': antecedent,
                    'mentions': mentions,
                }],
            })
        return corpus