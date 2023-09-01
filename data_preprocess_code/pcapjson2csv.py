import json
import csv
import glob
import os


def write_csv(csv_filepath, longest_common_field_name_sequence, field_value_list):
    with open(csv_filepath, 'w', encoding='UTF8', newline='') as f: 
        writer = csv.writer(f)
        #writer.writerows(csv_data_list)
        writer.writerow(longest_common_field_name_sequence)
        writer.writerows(field_value_list)


def get_common_field_names(field_name_list):
    shortest_len = float('inf')  # set to positive infinity initially
    for i in range(len(field_name_list)):
        if len(field_name_list[i]) < shortest_len:
            shortest_len = len(field_name_list[i])

    for j in range(shortest_len):
        name = field_name_list[0][j]
        for i in range(1, len(field_name_list)):
            if field_name_list[i][j] != name:
                shortest_len = j
    
    return field_name_list[0][0:shortest_len]


def read_field(pcap_data, field_name_list, packet_field_value_list, packet_byte_shift, protocol_name, last_field_len):
    """Reads packet json data.

  Args:
    pcap_data: The json data of a packet.
    field_name_list: field names of a packet.
    packet_field_value_list: field value of a packet.
    packet_byte_shift: byte shift of a field.
    protocol_name: str of the protocol name.
    last_field_len: length of the last field.

  """
    
    for k,v in pcap_data.items():
        # check if v is dict. If v is dict, execute read_field recursively.
        if(isinstance(v,dict)): 
            last_field_len = read_field(v, field_name_list, packet_field_value_list, packet_byte_shift, protocol_name, last_field_len)
            #print(k+":")
            continue
        # check if v is parseable. First handle the case where v contains only one set of values (i.e., the case where k has no repetitions)
        if(isinstance(v,list) and len(v)==5 and isinstance(v[0],str)):
            # Values smaller than one byte are not processed
            #if(v[1]==packet_byte_shift and v[2]==1 and last_field_len==1):
            if(v[1]==packet_byte_shift and v[2]==last_field_len):
                last_field_len = v[2]
                continue
            else:
                if(isinstance(protocol_name,str)):
                    if(k[:len(protocol_name)+1] == protocol_name + '.'):
                        # Only one parsing result for the same start field is reserved.
                        if(packet_byte_shift==v[1] and v[2]<=last_field_len):
                            if(len(field_name_list)>0 and len(packet_field_value_list)>0):
                                field_name_list.pop()
                                packet_field_value_list.pop()
                        field_name_list.append(k)
                        packet_field_value_list.append(v[0])
                        packet_byte_shift = v[1]
                        last_field_len = v[2]
                elif(isinstance(protocol_name,list)):
                    for name in protocol_name:
                        if(k[:len(name)+1] == name + '.'):
                            if(packet_byte_shift==v[1] and v[2]<=last_field_len):
                                if(len(field_name_list)>0 and len(packet_field_value_list)>0):
                                    field_name_list.pop()
                                    packet_field_value_list.pop()
                            field_name_list.append(k)
                            packet_field_value_list.append(v[0])
                            packet_byte_shift = v[1]
                            last_field_len = v[2]
        if(isinstance(v,list) and isinstance(v[0],list)):
            # If there are duplicate keys in the original json file, then k corresponds to a list v, which contains all the values corresponding to k.
            for i in range(len(v)):
                if(len(v[i])==5 and isinstance(v[i][0],str)):
                    if(v[i][1] == packet_byte_shift and v[i][2] == 1 and last_field_len==1):
                        last_field_len = v[i][2]
                        continue
                    else:
                        if(isinstance(protocol_name,str)):
                            if(k[:len(protocol_name)+1] == protocol_name + '.'):
                                if(packet_byte_shift==v[i][1] and v[i][2]<=last_field_len):
                                    field_name_list.pop()
                                    packet_field_value_list.pop()
                                field_name_list.append(k)
                                packet_field_value_list.append(v[i][0])
                                packet_byte_shift = v[i][1]
                                last_field_len = v[i][2]
                        elif(isinstance(protocol_name,list)):
                            for name in protocol_name:
                                if(k[:len(name)+1] == name + '.'):
                                    if(packet_byte_shift==v[i][1] and v[i][2]<=last_field_len):
                                        if(len(field_name_list)>0 and len(packet_field_value_list)>0):
                                            field_name_list.pop()
                                            packet_field_value_list.pop()
                                    field_name_list.append(k)
                                    packet_field_value_list.append(v[i][0])
                                    packet_byte_shift = v[i][1]
                                    last_field_len = v[i][2]
        #print(k+" : "+str(v))
    return last_field_len


# Handling duplicate keys
# For duplicate keys, return a list containing all values corresponding to the key.
def obj_pairs_hook(lst):
    result={}
    count={}
    for key,val in lst:
        if key in count:count[key]=1+count[key]
        else:count[key]=1
        if key in result:
            if count[key] > 2:
                result[key].append(val)
            else:
                result[key]=[result[key], val]
        else:
            result[key]=val
    return result


def read_pcap_json_file(json_filepath, csv_filepath, protocol_name):
    with open(json_filepath, 'r', encoding='UTF-8') as pcap_json_file:
        pcap_data = json.load(pcap_json_file, object_pairs_hook=obj_pairs_hook)
        packet_number = len(pcap_data)

        csv_field_name_list = []
        csv_field_value_list = []

        for i in range(packet_number):
            field_name_list = []
            field_value_list = []

            packet_byte_shift = 0
            last_field_len = [0]

            read_field(pcap_data[i], field_name_list, field_value_list, packet_byte_shift, protocol_name, last_field_len[0])

            csv_field_name_list.append(field_name_list)
            csv_field_value_list.append(field_value_list)
        
        common_field_names = get_common_field_names(csv_field_name_list)
        field_value_list = [lst[:len(common_field_names)] for lst in csv_field_value_list]

        write_csv(csv_filepath, common_field_names, field_value_list)


def pcapjson2csv(protocol_name_list):
    for protocol_name in protocol_name_list:

        jsonfile_path_list = glob.glob(os.path.join('..', 'data', 'json', protocol_name, '*.json'), recursive=True)

        protocol_pcapcsvfolder_path = os.path.join('..', 'data', 'pcapcsv', protocol_name)
        if not os.path.exists(protocol_pcapcsvfolder_path):
            os.makedirs(protocol_pcapcsvfolder_path)

        if protocol_name == 'modbus':
            protocol_name = ['mbtcp','modbus']
        for json_filepath in jsonfile_path_list:
            file_name, _ = os.path.splitext(os.path.basename(json_filepath))
            csvfile_path = os.path.join(protocol_pcapcsvfolder_path, file_name+'.csv')
            read_pcap_json_file(json_filepath, csvfile_path, protocol_name)