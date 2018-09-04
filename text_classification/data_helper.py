import numpy as np

def load_feature_dict_1(filename, encode=True):
    word2id = {}
    id2word = []
    with open(filename, 'rt', encoding='utf-8') as f:
        idx = 0
        for line in f:
            line = line.strip('\r\n')
            fs = line.split(' ')
            if encode:
                u_word = fs[0].encode('utf-8')
            else:
                u_word = fs[0]
            word2id[u_word] = idx
            id2word.append(u_word)
            idx += 1
    return word2id, id2word


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


def load_raw_data(filename, mode):
    sentences, labels = [], []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\r\n')
            fs = line.split('\t')
            sentences.append(fs[0].lower())
            if mode == "train":
                labels.append(fs[1])
            else:
                labels.append("")
    return sentences, labels


def build_input_data(raw_file, feature_file, label_file, pad=True, max_len=31, mode="train"):
    sentences, labels = load_raw_data(raw_file, mode)
    fea2id, id2fea = load_feature_dict(feature_file)
    label2id, id2label = load_label_dict(label_file)

    if pad and max_len:
        label_ids = []
        sentence_ids = []
        for i in range(len(sentences)):
            label_id = [0 for i in range(len(label2id))]
            if mode == "train":
                label = labels[i]
                if label not in label2id:
                    continue
                label_id[label2id[label]] = 1
            label_ids.append(label_id)
            #sentence_id = [fea2id[w.encode('utf-8')] if w.encode('utf-8') in fea2id  else 0 for w in sentences[i]]
            sentence_id = [fea2id[w] if w in fea2id  else 0 for w in sentences[i]]
            sen_len = len(sentence_id)
            if sen_len > max_len:
                sentence_id = sentence_id[:max_len]
            else:
                sentence_id = sentence_id + [0] * (max_len - sen_len)
            sentence_ids.append(sentence_id)

    x = np.array(sentence_ids)
    y = np.array(label_ids)
    return x, y, fea2id, id2fea, label2id, id2label 


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
