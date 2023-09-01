import csv
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import argparse
import os

def field_crawler(protocol_names_list, field_name_description_csv_dir, description_csv_dir):
    for protocol_name in protocol_names_list:
        url = 'https://www.wireshark.org/docs/dfref/' + str(protocol_name[0]) + '/' + str(protocol_name) + '.html'
        response = requests.get(url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        field_name_list = []
        description_list = []
        for row in table.findAll('tr'):
            columns = row.findAll('td')
            if columns:
                field_name = columns[0].text.strip()
                description = columns[1].text.strip()
                field_name_list.append(field_name)
                description_list.append(description)
        csv_file_name = str(protocol_name) + '_field_name_description.csv'
        csv_rows = zip(field_name_list, description_list)
        with open(os.path.join(field_name_description_csv_dir, csv_file_name), 'w', newline='') as file:
            writer = csv.writer(file)
            for row in csv_rows:
                writer.writerow(row)
        
        description_without_duplicate_list = list(OrderedDict.fromkeys(description_list)) # Remove duplicate descriptions
        csv_file_name = str(protocol_name) + '_description.csv'
        csv_rows = zip(description_without_duplicate_list)
        with open(os.path.join(description_csv_dir, csv_file_name), 'w', newline='') as file:
            writer = csv.writer(file)
            for row in csv_rows:
                writer.writerow(row)

def preparing_fields_descriptions(protocol_names_list):
    field_name_description_csv_dir = os.path.join('..', 'data', 'field-name-description')
    description_csv_dir = os.path.join('..', 'data', 'description')

    if 'modbus' in protocol_names_list:
        protocol_names_list.append('mbtcp')
    for protocol_name in protocol_names_list:
        csv_field_name_description = str(protocol_name) + '_field_name_description.csv'
        csv_field_name_description_path = os.path.join(field_name_description_csv_dir, csv_field_name_description)
        csv_description = str(protocol_name) + '_description.csv'
        csv_description_path = os.path.join(description_csv_dir, csv_description)
        if os.path.exists(csv_field_name_description_path) and os.path.isfile(csv_field_name_description_path) \
            and os.path.exists(csv_description_path) and os.path.isfile(csv_description_path):
            continue
        else:
            field_crawler(protocol_names_list, field_name_description_csv_dir, description_csv_dir)


if __name__ == '__main__':

    field_name_description_csv_dir = os.path.join('..', 'data', 'field-name-description')
    description_csv_dir = os.path.join('..', 'data', 'description')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--protocol', nargs='+', default=['dnp3'], dest='protocol_name', help='protocol name')

    args = parser.parse_args()

    field_crawler(args.protocol_name, field_name_description_csv_dir, description_csv_dir)