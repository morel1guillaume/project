from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
import spacy


def train_nlu(data, config, model_dir):
    spacy.load('fr_core_news_md')
    training_data = load_data(data)
    trainer = Trainer(RasaNLUConfig(config))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = 'chatbot2')

def run_nlu():
    interpreter= Interpreter.load('./models/nlu/default/chatbot2',RasaNLUConfig('config_chatbot.json'))
    print(interpreter.parse("commment se connecter au wifi interne"))
    print(interpreter.parse("wifi"))
    print(interpreter.parse("code puk"))
    print(interpreter.parse("guest"))
    print("interne")

if __name__ == '__main__':
    train_nlu('./data/data2.md', 'config_chatbot.json', './models/nlu')
    run_nlu()
    

