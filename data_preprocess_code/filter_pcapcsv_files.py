import csv
import glob
import os

def delete_csv_column(csvfile_path, target_col_list):
    with open(csvfile_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        target_index_list = []

        for target_col in target_col_list:
            if target_col in header:
                target_index_list.append(header.index(target_col))
                header.remove(target_col)

        if len(target_index_list) > 0:
            data = []
            for row in reader:
                for target_index in target_index_list:
                    del row[target_index]
                data.append(row)
            with open(csvfile_path, 'w', newline='') as new_csvfile:
                writer = csv.writer(new_csvfile)
                writer.writerow(header)
                writer.writerows(data)


def merge_csv_column(csvfile_path, n):
    with open(csvfile_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        header[-n:] = ['s7comm.data_raw']
        rows = [row for row in reader]
        for i in range(len(rows)):
            last_cols = rows[i][-n:]  
            merged_col = ''.join(last_cols)  
            rows[i][-n:] = [merged_col]  
        with open(csvfile_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  
            writer.writerows(rows) 


def filtering_pcapcsv_files(protocol_name_list):
    pcapcsv_file_dir = os.path.join('..', 'data', 'pcapcsv')
    for protocol_name in protocol_name_list:
        protocol_pcapcsvfolder_path = os.path.join(pcapcsv_file_dir, protocol_name)
        protocol_pcapcsv_path_list = glob.glob(os.path.join(protocol_pcapcsvfolder_path, '*.csv'), recursive=False)

        if protocol_name == 'dnp3':
            target_col_list = []
            target_col_list.extend(['dnp3.al.objq.prefix_raw', 'dnp3.al.range.quantity_raw'])
            for pcapcsv_path in protocol_pcapcsv_path_list:
                delete_csv_column(pcapcsv_path, target_col_list)
        
        if protocol_name == 's7comm':
            for pcapcsv_path in protocol_pcapcsv_path_list:
                with open(pcapcsv_path, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    if header[-1:] == ['s7comm.resp.data_raw'] and header[-2:-1] == ['s7comm.data.length_raw'] and \
                        header[-3:-2] == ['s7comm.data.transportsize_raw'] and header[-4:-3] == ['s7comm.data.returncode_raw']:
                        merge_csv_column(pcapcsv_path, 4)


if __name__ == '__main__':
    protocol_name_list = ['dnp3']
    filtering_pcapcsv_files(protocol_name_list)