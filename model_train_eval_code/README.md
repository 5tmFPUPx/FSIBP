Note that data preprocessing needs to be performed before training the model.

## Training

To train the model, run train.py with the following command:

```bash
python train.py -trp modbus dnp3 -tep s7comm
```

Arguments:

- `-trp`, `--trainprotocol`: protocols used for training

- `-tep`, `--testprotocol`: protocols used for testing

Although the data of the protocols used for testing will not be used to train the model, the protocols used for testing need to be specified for preparing the data for later testing.

We use a pre-trained NLP model named `all-MiniLM-L12-v2`. It can be changed to other pre-trained models supported by [sentence-transformer](https://github.com/UKPLab/sentence-transformers). For example,  use `all-MiniLM-L6-v2` by:

```bash
python train.py -m all-MiniLM-L6-v2 -trp modbus dnp3 -tep s7comm
```

Arguments:

* `-m`, `--model`: the pre-trained NLP model

The `sentence-transformer` will automatically download the specified pre-trained model.

## Testing

To train the model, run train.py with the following command:

```bash
python eval.py -tep s7comm
```

Arguments:

* `-tep`, `--testprotocol`: protocols used for testing
