import numpy as np

def load_feature_dict(filename):
    word2id = {}
    id2word = []
    with open(filename, 'rt', encoding='utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip('\r\n')
            fs = line.split(' ')
            word = fs[0]
            word2id[word] = idx
            id2word.append(word)
            idx += 1
    return word2id, id2word


def load_label_dict(filename, min_count=200):
    word2id = {}
    id2word = []
    with open(filename, 'rt', encoding='utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip('\r\n')
            fs = line.split('\t')
            word, count = fs[0], fs[1]
            count = int(count)
            if count < min_count:
                continue
            word2id[word] = idx
            id2word.append(word)
            idx += 1
    return word2id, id2word


class Util(object):
    def __init__(self, feature_file, label_file):
        self.fea2id, self.id2fea = load_feature_dict(feature_file)
        self.label2id, self.id2label = load_label_dict(label_file)
    
    def parse_feature(self, poi, pad=True, max_len=31):
        sentence_ids = []
        sentence_id = [self.fea2id[w] if w in self.fea2id else 0 for w in poi.lower()]
        if pad and max_len:
            sen_len = len(sentence_id)
            if sen_len > max_len:
                sentence_id = sentence_id[:max_len]
            else:
                sentence_id = sentence_id + [0] * (max_len - sen_len)
        sentence_ids.append(sentence_id)
        return np.array(sentence_ids)

    def index2label(self, index):
        return self.id2label[index]

