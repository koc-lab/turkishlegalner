from pathlib import Path
import sys
import getopt
import json

import model_fcn
import preprocessing
import LSTM_CNN_CRF as LCC
import LSTM_CRF as LC
import LSTM_LSTM_CRF as LLC


# Models
models = (LC.LSTM_CRF, LCC.LSTM_CNN_CRF, LLC.LSTM_LSTM_CRF)
Path('../data').mkdir(exist_ok=True)
settings = 'data.json'


def main(argv):
    model = models[1]
    pp_flag = False
    mode = "train"
    embed = "glove"
    try:
        opts, args = getopt.getopt(argv, "hm:e:pa:", ["model=", "embed=", "prep=", "mode="])
    except getopt.GetoptError:
        # Some error code
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(' main.py -m <model> -e <embed> -p (preprocess flag) -a <mode>' +
                  '\n models: \n\t  lc -> LSTM-CRF (default) \n\t lcc -> LSTM-CNN-CRF \n\t llc -> LSTM-LSTM-CRF ' +
                  '\n embeddings: \n\t glove (default) \n\t m2v  \n\t hybrid' +
                  '\n -p: preprocess flag, default = False' +
                  '\n -a: mode of operation -> train, k_fold, test or use. Default = train')
            sys.exit()
        elif opt in ("-m", "--model"):
            if arg == "lc":
                model = models[0]
            elif arg == "lcc":
                model = models[1]
            elif arg == "llc":
                model = models[2]
        elif opt in ("-e", "--embed"):
            if arg == 'glove' or 'm2v' or 'hybrid':
                embed = arg
        elif opt == '-p':
            pp_flag = True
        elif opt in ("-a", "--mode"):
            if arg == 'train' or 'test' or 'use' or 'k_fold':
                mode = arg

    model = model(embed)
    # preprocessing
    if pp_flag:
        preprocessing.build_vocab()
        preprocessing.build_embeddings()

    # train and evaluate
    if mode == "train":
        model_fcn.train_and_evaluate(model)
    elif mode == "k_fold":
        model_fcn.k_fold_train(model, settings=settings)
    elif mode == "test":
        with open(settings) as json_file:
            data = json.load(json_file)
            model_fcn.evaluate_model(model, data['TEST_FILE'])
    elif mode == "use":
        with open(settings) as json_file:
            data = json.load(json_file)
            model_fcn.use_model(model, data['INPUT_FILE'])


if __name__ == "__main__":
    main(sys.argv[1:])
