import os
import glob
import subprocess

def pcap2json(protocol_name_list):
    for protocol_name in protocol_name_list:
        pcapfile_path_list = glob.glob(os.path.join('..', 'data', 'pcap', protocol_name, '*.pcap'), recursive=True)

        protocol_jsonfolder_path = os.path.join('..', 'data', 'json', protocol_name)
        if not os.path.exists(protocol_jsonfolder_path):
            os.makedirs(protocol_jsonfolder_path)

        for path in pcapfile_path_list:
            file_name, _ = os.path.splitext(os.path.basename(path))

            jsonfile_path = os.path.join(protocol_jsonfolder_path, file_name+'.json')

            #tshark -T json -J "protocol_name" -x -r input_file_path > output_file_path
            if protocol_name == 'modbus':
                command = 'tshark -T json -J \"mbtcp modbus\" -x -r ' + path + ' > '+ jsonfile_path
            else:
                command = 'tshark -T json -J \"' + protocol_name + '\" -x -r ' + path + ' > '+ jsonfile_path
            #print(command)
            subprocess.run(command, shell=True) 