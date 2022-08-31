from collections import Counter
from pathlib import Path
import numpy as np
import json


def build_vocab(min_count=1):
    """
    Creates a vocabulary of unique tokens given in <dir>/{}.words.txt
    and extracts IOB tags from <dir>/{}.tags.txt. Format should be
    arranged before calling this. Extracted files are saved to
    the DATA_DIR directory given in data.json file. Tag and word files
    for test and train should be provided in the directory given in
    DATASET_DIR of data.json file.

    :param min_count   : (Int) Number of same tokens to count, default 1
    :return: None
    """

    print("="*50)
    save_dir = ''
    dataset_dir = ''
    with open('data.json') as json_file:
        data = json.load(json_file)
        save_dir = data['DATA_DIR']
        dataset_dir = data['DATASET_DIR']

    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return str(Path(dataset_dir, '{}.words.txt')).format(name)

    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'test']:
        with Path(words(n)).open(encoding="utf-8") as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= min_count}

    with Path(save_dir, 'vocab.words.txt').open('w', encoding="utf-8") as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path(save_dir, 'vocab.chars.txt').open('w', encoding="utf-8") as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set
    def tags(name):
        return str(Path(dataset_dir, '{}.tags.txt')).format(name)

    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train')).open(encoding="utf-8") as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path(save_dir, 'vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))
    print("=" * 50)


def build_embeddings(size=300, glove='glove.vectors.txt', m2v='glove.morph2vec.txt'):
    """
    Generates glove and morph2vec embeddings from trained vectors given as parameters.
    Saves embeddings to data file given in data.json file with .npz extensions
    :param size : (Int) Vector size
    :param glove: (String) Name of glove vectors
    :param m2v  : (String) Name of morph2vec vectors
    :return: None
    """
    print("=" * 50)
    data_dir = ''
    vector_dir = ''
    with open('data.json') as json_file:
        data = json.load(json_file)
        data_dir = data['DATA_DIR']
        vector_dir = data['VECTOR_DIR']

    # Load vocab
    with Path(data_dir, 'vocab.words.txt').open(encoding="utf-8") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)
    # Array of zeros
    embeddings = np.zeros((size_vocab, size))
    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    try:
        with Path(vector_dir, glove).open(encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % 10000 == 0:
                    print('- At line {}'.format(line_idx))
                line = line.strip().split()
                if len(line) != size + 1:
                    continue
                word = line[0]
                embedding = line[1:]
                if word in word_to_idx:
                    found += 1
                    word_idx = word_to_idx[word]
                    embeddings[word_idx] = embedding
        print('- done. Found {} vectors for {} words'.format(found, size_vocab))
        np.savez_compressed(str(Path(data_dir, 'glove.npz')), embeddings=embeddings)
    except:
        print("Glove vectors could not be found")
        pass

    try:
        # Array of zeros
        m2v_embeddings = np.zeros((size_vocab, size))
        # Get relevant morph2vec vectors
        found = 0
        print('Reading Morph2Vec file (may take a while)')
        with Path(vector_dir, m2v).open(encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % 10000 == 0:
                    print('- At line {}'.format(line_idx))
                line = line.strip().split()
                if len(line) != size + 1:
                    continue
                word = line[0]
                m2v_embedding = line[1:]
                if word in word_to_idx:
                    found += 1
                    word_idx = word_to_idx[word]
                    m2v_embeddings[word_idx] = m2v_embedding
        print('- done. Found {} vectors for {} words'.format(found, size_vocab))
        np.savez_compressed(str(Path(data_dir, 'morph2vec.npz')), embeddings=m2v_embeddings)
    except:
        print("Morph2Vec vectors could not be found")
        pass
    print("=" * 50)
