## Data preprocessing

Before data preprocessing, placing the pcap file in `../data/pcap/`, creating a subfolder for each protocol. For example, suppose there are three protocols: dnp3, s7comm, modbus, then placethe pcap file of dnp3 in `../data/pcap/dnp3/`, the pcap file of s7comm in `../data/pcap/s7comm/`, the pcap file of modbus in `../data/pcap/modbus/`.

Packets of different message types for each protocol are stored in different pcap files, in other words, each pcap file contains packets of one message type.

We place several simple pcap files in the `../data/pcap/` folder for demonstration purposes, and after cloning the repository, the following preprocessing code can be executed directly. After the preprocessing is complete, the training and test model code can also be executed.

To perform data preprocessing, run `data_preprocessing.py` with the following command:

```bash
python train.py -p dnp3 s7comm modbus
```

Arguments:

* `-p`, `--protocol`: protocols for preprocessing
  
  

Data preprocessing consists of four steps.
First, the field names and descriptions of the protocols are crawled from wireshark ([Wireshark · Display Filter Reference: Index](https://www.wireshark.org/docs/dfref/)), the file stored in `../data/description/` and `../data/field-name-description/`

Second, tshark is used to extract the data of each packet in the pcap file, and the results are saved in json format, the json file stored in `../data/json/`
Third, the json file is processed and converted to a csv file. A packet corresponds to a row in the csv file, the json file stored in `../data/pacpcsv/`
Finally, some inappropriate data in the csv file is filtered.



Note that the json format exported by different versions of tshark may differ. Our code should work for 4.0.x versions. The protocols supported by our code so far are dnp3, s7comm, modbus, omron.
