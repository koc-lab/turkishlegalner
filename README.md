# LegalNER: Named Entity Recognition in Turkish Legal Text

## Data and Results

This repository is generated based on [Berkay Yazıcıoğlu](https://github.com/BerkayYazicioglu)'s [LegalNER](https://github.com/BerkayYazicioglu/LegalNER) implementation. However, due to GitHub LFS storage and bandwidth restrictions, some data and result files are moved to [Google Drive](https://drive.google.com/drive/folders/142Us0GfeEP_nXfZR90c1q_H3eoMwen50?usp=sharing) storage. The files under `data/vectors/*` and `src/*_results/*` are available under this [Google Drive link](https://drive.google.com/drive/folders/142Us0GfeEP_nXfZR90c1q_H3eoMwen50?usp=sharing). To get the full version of the repository, download and place necessary files from [Google Drive](https://drive.google.com/drive/folders/142Us0GfeEP_nXfZR90c1q_H3eoMwen50?usp=sharing), and place regarding the directory hierarchy.

## Data Format

Follow the [`data/train_and_test`]

1. For `name` in `{train, test}`, create files `{name}.words.txt` and `{name}.tags.txt` that contain one sentence per line, each word / tag separated by space using IOBES tagging scheme.
2. Create files `vocab.words.txt`, `vocab.tags.txt` and `vocab.chars.txt` that contain one token per line. This can be automatized using the corresponding function in [`src/preprocessing.py`] and altering the fields **DATASET_DIR** to point to the locations of the file in Step 1 where **DATA_DIR** pointing to the output directory.
3. Create a `glove.*.npz` file containing one array `embeddings` of shape `(size_vocab_words, 300)` using [GloVe 840B vectors](https://nlp.stanford.edu/projects/glove/). This can be built by using the corresponding function in [`src/preprocessing.py`] after completing Step 2 and altering the field **VECTOR_DIR** to point desired output directory.

## Get Started

Tensorflow 1.15 should be used, other versions are untested. Remaining packages should be operational without a specified version.

Once produced all the required data files, use the `main.py` with specifying correct parameters. These are:

1. **model (-m):** Three base architectures; `lc`for LSTM-CRF, `llc` for LSTM-LSTM-CRF and `lcc` for LSTM-CRF-CRF.
2. **embeddings (-e):** Three embeddings to pair with a base architecture; `glove` for GloVe, `m2v` for Morph2Vec and `hybrid` for their combination. Make sure that correct `.npz` files are present in the `data` folder and call preprocessing each time a different embedding is used.
3. **preprocessing (-p):** Flag to use preprocessing scripts, non-mandatory. Must be called in between new embedding selections.
4. **mode (-a):** Four modes, directories are used as stated in [`src/data.json`]; `train` to train a model from scratch, `k_fold` to perform cross-validation (default=5), `test` for testing and validating a specific input (default=None) and `use` for generating an output from a specified file using a trained model (default=None).

```
python main.py -m <model> -e <embed> -p (preprocess flag) -a <mode>
```

If multiple tests are aimed to be performed with slight changes on the parameters, check out [`src/multiple_run.sh`] script.
