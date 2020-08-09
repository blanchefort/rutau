import os
import shutil
import wget
from navec import Navec
from slovnet import NER
from gensim.models import KeyedVectors

__version__ = '0.1.4'

basedir = os.path.abspath(os.path.dirname(__file__))

models_path = os.environ.get('MODELS_PATH') or os.path.join(basedir, 'models')
data_path = os.path.join(basedir, 'datafiles')

def load_ner(models_path: str) -> NER:
    """Загружаем и инициализируем NER-модель

    Args:
        models_path (str): Папка, в которой расположены необходимые для работы модели

    Returns:
        slovnet.NER: Объект slovnet.NER
    """
    os.makedirs(models_path, exist_ok=True)
    if not os.path.isfile(os.path.join(models_path, 'navec_news_v1_1B_250K_300d_100q.tar')):
        wget.download('https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar',
                      os.path.join(models_path, 'navec_news_v1_1B_250K_300d_100q.tar'))
    if not os.path.isfile(os.path.join(models_path, 'slovnet_ner_news_v1.tar')):
        wget.download('https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar',
                      os.path.join(models_path, 'slovnet_ner_news_v1.tar'))
    navec = Navec.load(os.path.join(models_path, 'navec_news_v1_1B_250K_300d_100q.tar'))
    ner = NER.load(os.path.join(models_path, 'slovnet_ner_news_v1.tar'))
    ner.navec(navec)
    return ner

def load_w2v(models_path: str) -> KeyedVectors:
    """Загрузка модели word2vec

    https://rusvectores.org/ru/models/
    """
    os.makedirs(models_path, exist_ok=True)
    if not os.path.isfile(os.path.join(models_path, 'ruwikiruscorpora_upos_skipgram_300_2_2019.w2v')):
        wget.download('http://vectors.nlpl.eu/repository/20/182.zip',
                      os.path.join(models_path, '182.zip'))
        shutil.unpack_archive(
            os.path.join(models_path, '182.zip'),
            os.path.join(models_path, 'w2v'), 'zip')
        shutil.copy(
            os.path.join(models_path, 'w2v', 'model.bin'),
            os.path.join(models_path, 'ruwikiruscorpora_upos_skipgram_300_2_2019.w2v'))
        shutil.rmtree(os.path.join(models_path, 'w2v'))
        os.remove(os.path.join(models_path, '182.zip'))
    w2v_model = KeyedVectors.load_word2vec_format(
        os.path.join(models_path, 'ruwikiruscorpora_upos_skipgram_300_2_2019.w2v'),
        binary=True)
    w2v_model.init_sims(replace=True)
    return w2v_model
