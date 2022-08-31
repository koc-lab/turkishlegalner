from sklearn.model_selection import KFold
import json
import shutil
import functools
import time
from pathlib import Path
import tensorflow as tf
import numpy as np
import conlleval as conll


timestr = time.strftime("%Y-%m-%d_%H.%M")


def train_and_evaluate(model):
    # Delete previous model if exists
    try:
        shutil.rmtree(Path(model.results_dir, 'model'))
        shutil.rmtree(Path(model.results_dir, 'outputs'))
    except OSError as e:
        pass
    # Estimator, train and evaluate
    train_inpf = functools.partial(model.input_fn,
                                   model.get_datapath('train', 'words'),
                                   model.get_datapath('train', 'tags'),
                                   model.params,
                                   shuffle_and_repeat=True)
    eval_inpf = functools.partial(model.input_fn,
                                  model.get_datapath('test', 'words'),
                                  model.get_datapath('test', 'tags'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model.model_fn, str(Path(model.results_dir, 'model')), cfg)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, 'f1',
                                                              500, min_steps=8000,
                                                              run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def write_predictions(name):
        Path(model.results_dir, 'outputs').mkdir(parents=True, exist_ok=True)
        with Path(model.results_dir, 'outputs/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(model.input_fn,
                                          model.get_datapath(name, 'words'),
                                          model.get_datapath(name, 'tags'))
            golds_gen = model.generator_fn(model.get_datapath(name, 'words'),
                                           model.get_datapath(name, 'tags'))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                if str(model) == "LSTM_CRF":
                    (words, _), tags = golds
                else:
                    ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'test']:
        write_predictions(name)

    # conll evaluation
    input_file = str(Path(model.results_dir, 'outputs/test.preds.txt'))
    output_file = str(Path(model.results_dir, 'score.test.metrics.{}.txt'.format(timestr)))
    result = conll.conlleval(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
# ======================================================================================================================
def k_fold_train(model, settings):
    data_tags = ""
    data_words = ""
    k = 10
    with open(settings) as json_file:
        data = json.load(json_file)
        data_tags = data["KFOLD_TAGS"]
        data_words = data["KFOLD_WORDS"]
        k = data["KFOLD_COUNT"]
    k_fold_dir = str(Path(model.results_dir, "k_fold{}".format(k)))
    # Delete previous model if exists
    try:
        shutil.rmtree(Path(k_fold_dir))
    except OSError as e:
        pass
    Path(k_fold_dir).mkdir(parents=True, exist_ok=True)
    # combining dataset
    dataset = []
    with Path(data_words).open('r', encoding="utf-8") as w, Path(data_tags).open('r', encoding="utf-8") as t:
        dataset.extend(list(zip(w, t)))

    k_fold = KFold(n_splits=k, shuffle=True)

    def write_dataset(name, type, data):
        with Path(k_fold_dir, 'partitions/{}.{}.txt'.format(name, type)).open('w', encoding="utf-8") as file:
            for idx, line in enumerate(data):
                if idx == len(data)-1:
                    file.write(line.strip())
                else:
                    file.write(line)

    # begin training k times
    k = 1
    for train_index, test_index in k_fold.split(dataset):
        try:
            shutil.rmtree(Path(k_fold_dir, "model"))
        except OSError as e:
            pass
        Path(k_fold_dir, 'partitions').mkdir(parents=True, exist_ok=True)
        write_dataset('train', 'words', [item[0] for item in np.take(dataset, train_index, axis=0)])
        write_dataset('train', 'tags',  [item[1] for item in np.take(dataset, train_index, axis=0)])
        write_dataset('test', 'words', [item[0] for item in np.take(dataset, test_index, axis=0)])
        write_dataset('test', 'tags',  [item[1] for item in np.take(dataset, test_index, axis=0)])
        # Estimator, train and evaluate
        train_inpf = functools.partial(model.input_fn,
                                       str(Path(k_fold_dir, 'partitions/train.words.txt')),
                                       str(Path(k_fold_dir, 'partitions/train.tags.txt')),
                                       model.params,
                                       shuffle_and_repeat=True)
        eval_inpf = functools.partial(model.input_fn,
                                      str(Path(k_fold_dir, 'partitions/test.words.txt')),
                                      str(Path(k_fold_dir, 'partitions/test.tags.txt')))

        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(model.model_fn, str(Path(k_fold_dir, 'model')), cfg)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, 'f1',
                                                                  500, min_steps=8000,
                                                                  run_every_secs=120)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        preds_gen = estimator.predict(eval_inpf)
        golds_gen = model.generator_fn(str(Path(k_fold_dir, 'partitions/test.words.txt')),
                                       str(Path(k_fold_dir, 'partitions/test.tags.txt')))

        Path(k_fold_dir, 'outputs').mkdir(parents=True, exist_ok=True)
        with Path(k_fold_dir, 'outputs/test.preds.txt').open('wb') as f:
            for golds, preds in zip(golds_gen, preds_gen):
                if str(model) == "LSTM_CRF":
                    (words, _), tags = golds
                else:
                    ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

        # conll evaluation
        input_file = str(Path(k_fold_dir, 'outputs/test.preds.txt'))
        output_file = str(Path(k_fold_dir, 'score{}.test.metrics.{}.txt'.format(k, timestr)))
        result = conll.conlleval(input_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        k += 1


# ======================================================================================================================
def evaluate_model(model, name):
    # Delete previous outputs if exist
    try:
        shutil.rmtree(str(Path(model.results_dir, 'outputs')))
    except OSError as e:
        pass

    eval_inpf = functools.partial(model.input_fn,
                                  model.get_datapath(name, 'words'),
                                  model.get_datapath(name, 'tags'))
    golds_gen = model.generator_fn(model.get_datapath(name, 'words'),
                                   model.get_datapath(name, 'tags'))
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model.model_fn, str(Path(model.results_dir, 'model')), cfg)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    preds_gen = estimator.predict(eval_inpf)
    with Path(model.results_dir, 'outputs/{}.preds.txt'.format(name)).open('wb') as f:
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), (_, _)), tags = golds
            for word, tag, tag_pred in zip(words, tags, preds['tags']):
                f.write(b' '.join([word, tag, tag_pred]) + b'\n')
            f.write(b'\n')

    # conll evaluation
    input_file = str(Path(model.results_dir, 'outputs/{}.preds.txt'.format(name)))
    output_file = str(Path(model.results_dir, 'score.{}.metrics.{}.txt'.format(name, timestr)))
    result = conll.conlleval(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)


# ======================================================================================================================
def use_model(model, data_dir):
    # Delete previous outputs if exist
    try:
        shutil.rmtree(str(Path(model.results_dir, 'outputs')))
    except OSError as e:
        pass
    estimator = tf.estimator.Estimator(model.model_fn, str(Path(model.results_dir, 'model')), params=model.params)
    predict_inpf = functools.partial(model.predict_input_fn, data_dir)
    Path(model.results_dir, 'outputs').mkdir(parents=True, exist_ok=True)
    with Path(model.results_dir, 'outputs/input.preds.txt').open('wb') as f:
        preds_gen = estimator.predict(predict_inpf)
        golds_gen = model.predict_input_fn(data_dir)
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), (_, _)), _ = golds
            for word, tag_pred in zip(words, preds['tags']):
                f.write(b' '.join([word, tag_pred]) + b'\n')
            f.write(b'\n')

