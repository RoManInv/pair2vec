from webbrowser import get
from endtasks.util import get_pair2vec
import torch
import pickle

from embeddings.util import *
from embeddings.representation import *
from embeddings.vocab import *

__MODEL_PATH__ = 'experiments/pair2vec/best.pt'
__CONFIG_PATH__ = 'experiments/pair2vec/saved_config.json'
__VALID_CONFIG_PATH__ = 'experiments/pair2vec/saved_config_utf8.json'
__DICT_PATH__ = 'embeddings/pair_to_index.pkl'
__VECTOR_PATH__ = 'data/wiki-vector.vec'
__VOCAB_PATH__ = 'data/pair2vec_tokens.txt'
__VOCAB_VALID_PATH__ = 'data/pair2vec_tokens_utf8.txt'
__VECTOR_CACHE__ = 'data'
__VECTOR_NAME__ = 'wiki-vector.vec'

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
    vectors = Vectors(__VECTOR_PATH__, __VECTOR_CACHE__)
    wordlist = list()
    # with open(__VOCAB_PATH__, 'rb') as fin:
    #     with open(__VOCAB_VALID_PATH__, 'wb') as fout:
    #         content = fin.read()
    #         fout.write(content.decode().encode('utf-8'))
    with open(__VOCAB_VALID_PATH__, 'r', errors='ignore') as fin:
        for line in fin:
            wordlist.append(line.strip())
    vocab = Vocab(wordlist, vectors = vectors, vectors_cache = __VECTOR_CACHE__)
    # vectors = vocab.load_vectors('fasttext.en.300d')
    # print(vectors)
    # vocab = Vocab(vectors)
    vec1 = vectors.__getitem__('germany')
    vec2 = vectors.__getitem__('berlin')
    print(model([vec1, vec2]))
    # print(vec1)
    # print(model(['germany', 'france']))
    # print(SpanRepresentation(config, 300, vec1))
    # with open(__DICT_PATI__, 'rb') as f:
    #     pair2index = pickle.load(f)
    # print(pair2index)


if __name__ == '__main__':
    main()