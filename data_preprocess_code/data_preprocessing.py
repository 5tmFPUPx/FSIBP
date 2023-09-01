import argparse
import os
from wireshark_field_crawler import preparing_fields_descriptions
from pcap2json import pcap2json
from pcapjson2csv import pcapjson2csv
from filter_pcapcsv_files import filtering_pcapcsv_files
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--protocol', nargs='+', default=['dnp3'], dest='protocol_name', help='protocol name')

    args = parser.parse_args()

    preparing_fields_descriptions(args.protocol_name)
    pcap2json(args.protocol_name)
    pcapjson2csv(args.protocol_name)
    filtering_pcapcsv_files(args.protocol_name)
