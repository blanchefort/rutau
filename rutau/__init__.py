import os
import wget
from navec import Navec
from slovnet import NER

__version__ = '0.1.0'

basedir = os.path.abspath(os.path.dirname(__file__))

models_path = os.environ.get('MODELS_PATH') or os.path.join(basedir, 'models')

os.makedirs(models_path, exist_ok=True)

def load_ner(models_path: str) -> NER:
    """Загружаем и инициализируем NER-модель

    Args:
        models_path (str): Папка, в которой расположены необходимые для работы модели

    Returns:
        slovnet.NER: Объект slovnet.NER
    """
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