# FSIBP

This is the code of FSIBP.

The *model_train_eval_code* folder contains code for training and testing models.
The *data_preprocess_code* folder contains code related to preprocessing data.
The *data* folder is used to place the data.
The *all-MiniLM-L12-v2* folder is the pre-training NLP models.

The overall process of using the code is to first place the pcap file of the protocol used for training or testing in *. /data/pcap/*, then execute the code in the *data_preprocess_code* folder for data preprocessing, after which execute the code in the *model_train_eval_code* folder to train the FSIBP model and test it.

It is required that **tshark** has been installed and added to the environment variables.

See the README in each folder for how to use the code.


