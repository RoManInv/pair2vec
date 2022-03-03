from webbrowser import get
from endtasks.util import get_pair2vec
import torch
import pickle

from embeddings.util import *
from embeddings.representation import *

__MODEL_PATH__ = 'experiments/pair2vec/best.pt'
__CONFIG_PATH__ = 'experiments/pair2vec/saved_config.json'
__VALID_CONFIG_PATH__ = 'experiments/pair2vec/saved_config_utf8.json'
__DICT_PATH__ = 'embeddings/pair_to_index.pkl'

def main():
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    with open(__CONFIG_PATH__, 'rb') as fin:
        with open(__VALID_CONFIG_PATH__, 'wb') as fout:
            content = fin.read()
            fout.write(content.decode().encode('utf-8'))
    model = get_pair2vec(__VALID_CONFIG_PATH__, __MODEL_PATH__)
    config = get_config(__VALID_CONFIG_PATH__)
    # print(model(['germany', 'france']))
    print(SpanRepresentation(config, 300, 'germany'))
    # with open(__DICT_PATI__, 'rb') as f:
    #     pair2index = pickle.load(f)
    # print(pair2index)


if __name__ == '__main__':
    main()