# -*- coding: utf-8 -*-
# this code aims to read the csv file that generates from tshark json
# and processes the data in the csv file
# and label each feild sentence based on the description embedding results

import csv
import argparse
import os
import glob
import random


def get_number_of_column(csv_file):
    with open(csv_file, 'r') as f:
        first_row = next(csv.reader(f))
        num_columns = len(first_row)
        return num_columns
    

def get_number_of_row(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        num_rows = sum(1 for row in reader)
        return num_rows


def csv2dict(filename, delimiter=','):
    """Reads a CSV file into a dictionary.
    This function is used to read fieldname_discription_csv and description_label_csv

    Args:
        filename: The name of the CSV file.
        delimiter: The delimiter used to separate the columns in the CSV file.

    Returns:
        A dictionary with the columns of the CSV file as the keys and the values of the columns as the values.
    """

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        dictionary = {}
        for row in reader:
            dictionary[row[0]] = row[1]

    return dictionary


def generate_fieldname_description_dict(csvfiles_path_list):
    """Reads fieldname_discription_csv and returns a dictionary that fieldname is the key and the decription is the value 

    Args:
        csvfiles_path_list: The path list of the fieldname_discription CSV files.
    """
    fieldname_description_dict = {}
    for csvfile in csvfiles_path_list:
        fieldname_description_dict.update(csv2dict(csvfile))
    return fieldname_description_dict


def generate_description_label_dict(csvfile_path):
    """Reads description_label_csv and returns a dictionary that description is the key and the clustering label is the value 

    Args:
        csvfile_path: The file path of the description_label CSV file.
    """
    description_label_dict = {}
    description_label_dict.update(csv2dict(csvfile_path))
    return description_label_dict


def traverse_field_name_description_csvfile(protocol_name_list, fields_description_csvfile_dir):
    """Get the field_name_description_csv file of the specified protocol

    Args:
        protocol_name_list: The name list of the protocol.
        fields_description_csvfile_dir: The path to the folder where the fields description csvfile is stored.

    Returns:
        A list that contains all the field_name_description_csv file path.
    """
    if 'modbus' in protocol_name_list:
        protocol_name_list.append('mbtcp')

    file_names_list = os.listdir(fields_description_csvfile_dir)
    csvfiles_to_read_list = []

    for protocol_name in protocol_name_list:
        for file_names in file_names_list:
            if file_names[:len(protocol_name)] == protocol_name:
                csvfiles_to_read_list.append(fields_description_csvfile_dir + file_names)
    
    return csvfiles_to_read_list


def covert_fieldname_to_label(fieldname_list, fieldname_description_dict, description_label_dict):
    """Get the corresponding label according to the field name.

    Args:
        fieldname_list: The list of the field names.
        fieldname_description_dict: Dictionary that fieldname is the key and the decription is the value.
        description_label_dict: Dictionary that description is the key and the clustering label is the value.

    Returns:
        A label list.
    """
    field_label_list = []
    description_list = []
    for fieldname in fieldname_list:
        # fieldname[:-4] 去掉 _raw
        if fieldname[:-4] in fieldname_description_dict:
            description_list.append(fieldname_description_dict[fieldname[:-4]])
    for description in description_list:
        if description in description_label_dict:
            field_label_list.append(description_label_dict[description])
    return field_label_list


def get_sentences_and_labels(pcapcsv_file_dir, protocol_names_list, fieldname_description_dict, description_label_dict):
    """Get the field value sequence (i.e., sentence) and the corresponding field label. 
       Also get the protocol label, which indicates the sentence belongs to which protocol.
       And also get the packet group label. Sentences that have the same packet group label are from the same packets.

    Args:
        pcapcsv_file_dir: The path to the folder where the pcap_csv files are stored.
        protocol_names_list: The name of the protocol to process.
        fieldname_description_dict: Dictionary that fieldname is the key and the decription is the value.
        description_label_dict: Dictionary that description is the key and the clustering label is the value.

    Returns:
        A sentence list, a corresponding field label list, a protocol label list and a packet_group_label_list
    """
    sentence_list = []
    field_label_list = []
    protocol_label_list = []
    packet_group_label_list = []
    packet_group_label = 0

    for protocol_name in protocol_names_list:
        protocol_pcapcsv_path_list = glob.glob(pcapcsv_file_dir + protocol_name + '\\*.csv', recursive=False)

        for pcapcsv_path in protocol_pcapcsv_path_list:
            num_columns = get_number_of_column(pcapcsv_path)
            num_rows = get_number_of_row(pcapcsv_path) - 1 # 不包括第一行的field name

            with open(pcapcsv_path, 'r') as f:
                reader = csv.reader(f)
                column_fieldname_list = next(reader)
                columns_labels_list = covert_fieldname_to_label(column_fieldname_list, fieldname_description_dict, description_label_dict)
                if len(columns_labels_list) != num_columns:
                    print('error')
                # Each field (column) is read in turn and concatenated into a sentence
                columns = [[] for _ in range(num_columns)]  # create an empty list for each column
                for row in reader:
                    for i in range(num_columns):
                        columns[i].append(row[i])  
                if num_rows <= 10:
                    sentence_list.extend(columns)
                    field_label_list.extend(columns_labels_list)
                    protocol_label_list.extend([protocol_name for _ in range(num_columns)])
                    packet_group_label_list.extend([packet_group_label for _ in range(num_columns)])
                    packet_group_label += 1
                if num_rows > 10:
                    # 从(num_rows % 10)中(包括num_rows % 10)选取随机数，以随机数作为sentence开始的index，每10个field一个sentence直到读取完
                    # 每个column的sentence数量为num_rows // 10
                    num_sentences_a_column = num_rows // 10
                    init_index_range = num_rows%10
                    random_init_index = 0
                    if init_index_range > 0:
                        random_init_index = random.randint(0, init_index_range)

                    #for i in range(len(columns)):
                    #    sentences_list_a_column = []
                    #    for j in range(num_sentences_a_column):
                    #        sentences_list_a_column.extend(columns[i][random_init_index+j*10:random_init_index+(j+1)*10])
                    #        sentence_list.extend([sentences_list_a_column])
                    #    field_label_list.extend([columns_labels_list[i]] * num_sentences_a_column)

                    sentences_list_a_packet_group = []
                    for j in range(num_sentences_a_column):
                        for i in range(num_columns):
                            sentences_list_a_packet_group.extend(columns[i][random_init_index+j*10:random_init_index+(j+1)*10])
                            sentence_list.extend([sentences_list_a_packet_group])
                            sentences_list_a_packet_group = []
                            field_label_list.extend([columns_labels_list[i]])
                            protocol_label_list.extend([protocol_name])
                            packet_group_label_list.extend([packet_group_label])
                        packet_group_label += 1
                    
    return sentence_list, field_label_list, protocol_label_list, packet_group_label_list


def generate_data_and_labels(protocol_name_list, fieldname_description_dir, description_label_path, pcapcsv_file_dir):
    fieldname_description_csvfiles_path_list = traverse_field_name_description_csvfile(protocol_name_list, fieldname_description_dir)
    fieldname_description_dict = generate_fieldname_description_dict(fieldname_description_csvfiles_path_list)
    description_label_dict = generate_description_label_dict(description_label_path)

    sentence_list, field_label_list, protocol_label_list, packet_group_label_list = get_sentences_and_labels(pcapcsv_file_dir, protocol_name_list, fieldname_description_dict, description_label_dict)
    field_label_list = list(map(int, field_label_list))

    return sentence_list, field_label_list, protocol_label_list, packet_group_label_list


if __name__ == "__main__":

    generate_data_and_labels(['omron'], '..\\data\\field-name-description\\', '..\\data\\description-label\\description_label.csv', '..\\data\\pcapcsv\\')

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--description', default='..\\data\\field-name-description\\', dest='field_name_description_filepath', help='filepath of fields description file')
    parser.add_argument('-p', '--protocol', nargs='+', default=['modbus', 's7comm', 'omron', 'dnp3'], dest='protocol_name', help='protocol name')

    args = parser.parse_args()

    protocol_names_list = []
    if isinstance(args.protocol_name,str):
        protocol_names_list.append(args.protocol_name)
    if isinstance(args.protocol_name,list):
        protocol_names_list = args.protocol_name
    
    #num_rows = get_number_of_row('C:\\Users\\ZMQ\\Documents\\Research\\协议逆向\\pcap\\dnp3\\dnp3_read.csv')
    #read_csv_file(args.field_name_description_filepath, protocol_names_list)
