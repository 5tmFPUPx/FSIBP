import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import glob
import csv
import random
import math
import os
import pickle
from FSIBP_net import FSIBP_Net
from SupConLoss import SupConLoss
from InfoNCE import InfoNCE
from pytorchtools import EarlyStopping
from description_embedding import descritpion_cossim
from pcapcsv2data import generate_data_and_labels, traverse_field_name_description_csvfile, generate_fieldname_description_dict


class FSIBP_pcapcsv_dataset(Dataset):
    def __init__(self, sentence_list, field_label_list, protocol_label_list, packet_group_label_list):
        self.sentence_list = sentence_list
        self.field_label_list = field_label_list
        self.protocol_label_list = protocol_label_list
        self.packet_group_label_list = packet_group_label_list
 
    def __len__(self):
        return len(set(self.packet_group_label_list))
    
    def _get_start_end_index(self, idx, packet_group_value):
        data_start_idx = self.packet_group_label_list.index(packet_group_value)
        for i in range(1, len(self.packet_group_label_list)-data_start_idx):
            if self.packet_group_label_list[data_start_idx+i]==idx:
                data_end_idx = data_start_idx + i
            if self.packet_group_label_list[data_start_idx+i]!=idx:
                break
        return data_start_idx, data_end_idx
    
    def _get_data(self, data_start_idx, data_end_idx, sentences, field_label, protocol_label, packet_group_label, group_field_length):
        sentences.extend([self.sentence_list[data_start_idx:data_end_idx+1]])
        field_label.extend([self.field_label_list[data_start_idx:data_end_idx+1]])
        protocol_label.extend([self.protocol_label_list[data_start_idx:data_end_idx+1]])
        packet_group_label.extend([self.packet_group_label_list[data_start_idx:data_end_idx+1]])
        group_field_length.append(len(self.packet_group_label_list[data_start_idx:data_end_idx+1]))
        return sentences, field_label, protocol_label, packet_group_label, group_field_length
    
    def _data_aug(self, aug_start_idx, aug_end_idx, data_start_idx, data_end_idx, sentences, field_label, protocol_label, packet_group_label, group_field_length):
        s_temp = []
        for s in sentences[0]:
            s_temp.extend([s[aug_start_idx:aug_end_idx+1]])
        sentences.extend([s_temp])
        field_label.extend([self.field_label_list[data_start_idx:data_end_idx+1]])
        protocol_label.extend([self.protocol_label_list[data_start_idx:data_end_idx+1]])
        packet_group_label.extend([[packet_group_label[0][0]+len(set(self.packet_group_label_list)) for _ in range(len(self.packet_group_label_list[data_start_idx:data_end_idx+1]))]])
        group_field_length.append(len(self.packet_group_label_list[data_start_idx:data_end_idx+1]))
        return sentences, field_label, protocol_label, packet_group_label, group_field_length
 
    def __getitem__(self, idx):
        sentences, field_label, protocol_label, packet_group_label, group_field_length = [], [], [], [], []

        # anchor
        packet_group_label_value_list = list(set(self.packet_group_label_list))
        packet_group_label_value = packet_group_label_value_list[idx]

        data_start_idx, data_end_idx = self._get_start_end_index(idx, packet_group_label_value)
        sentences, field_label, protocol_label, packet_group_label, group_field_length = \
            self._get_data(data_start_idx, data_end_idx, sentences, field_label, protocol_label, packet_group_label, group_field_length)

        # submessage
        sentence_length = len(sentences[0][0]) 
        if sentence_length <= 2:
            aug_start_idx = 0
            aug_end_idx = 0
        if sentence_length >= 3:
            aug_length = random.randint(2, math.ceil(sentence_length / 2))
            aug_start_idx = random.randint(0, sentence_length-aug_length) 
            aug_end_idx = aug_start_idx + aug_length - 1
        sentences, field_label, protocol_label, packet_group_label, group_field_length = \
            self._data_aug(aug_start_idx, aug_end_idx, data_start_idx, data_end_idx, sentences, field_label, protocol_label, packet_group_label, group_field_length)

        # Choose one packet_group from other protocol.
        protocol_label_value_list = list(dict.fromkeys(self.protocol_label_list)) 
        protocol_start_index_list = []
        for protocol in protocol_label_value_list:
            protocol_start_index_list.append(self.protocol_label_list.index(protocol))
        protocol_start_index_list.append(len(self.protocol_label_list))
        for protocol in protocol_label_value_list:
            if protocol != protocol_label[0][0]:
                neg_protocol_start_index = self.protocol_label_list.index(protocol)
                neg_packet_group_range = self.packet_group_label_list[neg_protocol_start_index : protocol_start_index_list[protocol_start_index_list.index(neg_protocol_start_index)+1]]
                neg_packet_gourp_value = list(set(neg_packet_group_range))
                random_neg_packet_gourp = random.choice(neg_packet_gourp_value)
                data_start_idx, data_end_idx = self._get_start_end_index(random_neg_packet_gourp, random_neg_packet_gourp)
                sentences, field_label, protocol_label, packet_group_label, group_field_length = \
                    self._get_data(data_start_idx, data_end_idx, sentences, field_label, protocol_label, packet_group_label, group_field_length)

        return sentences, field_label, protocol_label, packet_group_label


class centroid_pcapcsv_dataset(Dataset):
    def __init__(self, sentence_list, field_label_list, protocol_label_list, packet_group_label_list):
        self.sentence_list = sentence_list
        self.field_label_list = field_label_list
        self.protocol_label_list = protocol_label_list
        self.packet_group_label_list = packet_group_label_list
 
    def __len__(self):
        return len(set(self.packet_group_label_list))
 
    def __getitem__(self, idx):
        packet_group_label_value_list = list(set(self.packet_group_label_list))
        packet_group_label_value = packet_group_label_value_list[idx]

        data_start_idx = self.packet_group_label_list.index(packet_group_label_value)
        for i in range(1, len(self.packet_group_label_list)-data_start_idx):
            if self.packet_group_label_list[data_start_idx+i]==idx:
                data_end_idx = data_start_idx + i
            if self.packet_group_label_list[data_start_idx+i]!=idx:
                break

        sentences = self.sentence_list[data_start_idx:data_end_idx+1]
        field_label = self.field_label_list[data_start_idx:data_end_idx+1]
        protocol_label = self.protocol_label_list[data_start_idx:data_end_idx+1]
        packet_group_label = self.packet_group_label_list[data_start_idx:data_end_idx+1]

        return sentences, field_label, protocol_label, packet_group_label
    


def custom_collate(batch):
    sentences, field_label, protocol_label, packet_group_label = batch[0][0], batch[0][1], batch[0][2], batch[0][3]
    return sentences, field_label, protocol_label, packet_group_label


def field_data_process(sentence_list):
    """Covert the field vaule from HEX to DEC and pad the field to the fixed size. 
    The sentence_list contains sentences that belong to the same packet group.
    """
    sentence_list = [[[field[i:i+2] for i in range(0, len(field), 2)] for field in sentence] for sentence in sentence_list] 
    sentence_list = [[[int(hex_str, 16) for hex_str in field_list] for field_list in sentence] for sentence in sentence_list] # Hex to DEC

    field_lengths = []
    for sentence in sentence_list:
        for field in sentence:
            field_lengths.append(len(field))
    longest_field_length = max(field_lengths)
    sentence_list_size = len(sentence_list) 
    sentence_length = len(sentence_list[0])
    # padding
    for s in range(sentence_list_size):
        for l in range(sentence_length):
            if(len(sentence_list[s][l])) < longest_field_length:
                sentence_list[s][l] += [256] * (longest_field_length - len(sentence_list[s][l]))
    
    return sentence_list, field_lengths, longest_field_length, sentence_list_size, sentence_length


def read_csv_headers(file_names):
  """Reads the first row of each CSV file in `file_names` and returns a list of the column names.

  Args:
    file_names: A list of strings containing the names of the CSV files to read.

  Returns:
    A list of strings containing the column names from each CSV file.
  """

  column_names = []
  for file_name in file_names:
    with open(file_name, 'r') as f:
      reader = csv.reader(f)
      column_names.extend(next(reader))

  return column_names


def fieldname_to_description(field_names, fieldname_description_dict):
    description_list = []
    for fieldname in field_names:
        # [:-4] remove '_raw'
        if fieldname[:-4] in fieldname_description_dict:
            description_list.append(fieldname_description_dict[fieldname[:-4]])
    return description_list


def generate_train_test_description(train_protocol_name_list, test_protocol_name_list, pcapcsv_file_dir, fieldname_description_dir):
    #get all field name
    train_field_names = []
    for train_protocol_name in train_protocol_name_list:
        train_protocol_pcapcsv_path_list = glob.glob(os.path.join(pcapcsv_file_dir, train_protocol_name, '*.csv'), recursive=False)
        train_protocol_field_names = read_csv_headers(train_protocol_pcapcsv_path_list)
        train_protocol_field_names = list(set(train_protocol_field_names))
        train_field_names.extend(train_protocol_field_names)

    train_field_names = list(set(train_field_names))
    train_fieldname_description_csvfiles_path_list = traverse_field_name_description_csvfile(train_protocol_name_list, fieldname_description_dir)
    train_fieldname_description_dict = generate_fieldname_description_dict(train_fieldname_description_csvfiles_path_list)
    
    train_description_list = fieldname_to_description(train_field_names, train_fieldname_description_dict)

    test_field_names = []
    for test_protocol_name in test_protocol_name_list:
        test_protocol_pcapcsv_path_list = glob.glob(os.path.join(pcapcsv_file_dir, test_protocol_name, '*.csv'), recursive=False)
        test_protocol_field_names = read_csv_headers(test_protocol_pcapcsv_path_list)
        test_protocol_field_names = list(set(test_protocol_field_names))
        test_field_names.extend(test_protocol_field_names)

    test_field_names = list(set(test_field_names))
    test_fieldname_description_csvfiles_path_list = traverse_field_name_description_csvfile(test_protocol_name_list, fieldname_description_dir)
    test_fieldname_description_dict = generate_fieldname_description_dict(test_fieldname_description_csvfiles_path_list)
    
    test_description_list = fieldname_to_description(test_field_names, test_fieldname_description_dict)

    return train_description_list, test_description_list


def train_FSIBP(train_data_loader, num_field_class, device):
    model = FSIBP_Net()
    supcon_loss_fn = SupConLoss(temperature=0.07)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    selfcon_loss_fn = InfoNCE(temperature=0.1, negative_mode='unpaired')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #optimizer =torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    supcon_loss_fn.to(device)
    selfcon_loss_fn.to(device)
    ce_loss_fn.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=20, verbose=False, path='trained_FSIBP.pt')

    for epoch in range(50):
        model.train(True)
        for sentences, field_label, protocol_label, packet_group_label in train_data_loader:
            
            contrastive_group_length = len(sentences)
            packet_gru_output_list = []
            for i in range(contrastive_group_length):
                sentences[i], field_lengths, longest_field_len, sentences_size, sentence_len = field_data_process(sentences[i])
                sentences[i] = torch.IntTensor(np.array(sentences[i])) 
                sentences[i] = sentences[i].to(device) 
                packet_gru_output = model(sentences[i], field_lengths, longest_field_len, sentences_size, sentence_len)
                packet_gru_output_list.append(packet_gru_output)
            
            merged_gru_output = torch.cat((packet_gru_output_list),0)
            merged_field_label = sum(field_label, [])
            merged_field_label = torch.LongTensor(np.array(merged_field_label))
            merged_field_label = merged_field_label.to(device)
            supcon_loss = supcon_loss_fn(merged_gru_output, merged_field_label)

            for i,output in enumerate(packet_gru_output_list):
                packet_gru_output_list[i] = output.view(output.shape[0],output.shape[2])

            query = packet_gru_output_list[0]
            positive_key = packet_gru_output_list[1]
            negative_gru_output_list = []
            if contrastive_group_length <= 2:
                selfcon_loss = selfcon_loss_fn(query, positive_key)
            if contrastive_group_length > 2:
                for i in range(2, contrastive_group_length):
                    negative_gru_output_list.append(packet_gru_output_list[i])
                negative_keys = torch.cat((negative_gru_output_list), 0)
                selfcon_loss = selfcon_loss_fn(query, positive_key, negative_keys)

            loss = supcon_loss + selfcon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: ',epoch+1, 'supcon_loss: ',supcon_loss.item(), 'selfcon_loss:',selfcon_loss.item(), 'Loss: ',loss.item())
        
        scheduler.step(loss)
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return model

    return model


def train_model(fieldname_description_dir, description_label_path, pcapcsv_file_dir, train_protocol_name, test_protocol_name, sentence_embedding_model_name, device):

    train_description_list, test_description_list = generate_train_test_description(train_protocol_name, test_protocol_name, pcapcsv_file_dir, fieldname_description_dir)
    optimal_k = descritpion_cossim(train_description_list, test_description_list, sentence_embedding_model_name)

    train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list = \
        generate_data_and_labels(train_protocol_name, fieldname_description_dir, description_label_path, pcapcsv_file_dir)
    FSIBP_train_dataset = FSIBP_pcapcsv_dataset(train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list)
    FSIBP_train_data_loader = DataLoader(FSIBP_train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    trained_model = train_FSIBP(FSIBP_train_data_loader, optimal_k, device)

    return trained_model, train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list


def compute_centroid(trained_model, train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list, device):
    train_dataset = centroid_pcapcsv_dataset(train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    field_labels = list(set(train_field_label_list))

    centroid_list = []
    embedding_list = [[] for _ in range(len(field_labels))]

    trained_model.eval()
    with torch.no_grad():
        for sentences, field_label, _, _ in train_data_loader:

            sentences, field_lengths, longest_field_len, sentences_size, sentence_len = field_data_process(sentences)
            sentences = torch.IntTensor(np.array(sentences))
            field_label = torch.LongTensor(np.array(field_label))
            sentences = sentences.to(device) 
            field_label = field_label.to(device)

            packet_gru_output = trained_model(sentences, field_lengths, longest_field_len, sentences_size, sentence_len)
            packet_gru_output = packet_gru_output.view(sentences_size, 32) # 32 is sentence_hidden_size
            packet_gru_output = packet_gru_output.cpu().numpy()
            field_label = field_label.cpu().numpy()
            
            for output_index,output in enumerate(packet_gru_output):
                embedding_list[field_labels.index(field_label[output_index])].extend([output])
    
    for embeddings in embedding_list:
        norms = np.linalg.norm(embeddings, axis=1)
        norm_data = embeddings / norms[:, np.newaxis]
        centroid = np.mean(norm_data, axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        centroid_list.append(centroid_norm)
    
    return centroid_list


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("The model will be running on", device, "device\n")

    fieldname_description_dir = os.path.join('..', 'data', 'field-name-description')
    description_label_path = os.path.join('..', 'data', 'description-label', 'description_label.csv')
    pcapcsv_file_dir = os.path.join('..', 'data', 'pcapcsv')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='all-MiniLM-L12-v2', dest='model_name', help='name of the per-trained sentence embedding models')
    parser.add_argument('-trp', '--trainprotocol', type=str, nargs='+', default=['dnp3'], dest='train_protocol_name', help='train protocol name')
    parser.add_argument('-tep', '--testprotocol', type=str, nargs='+', default=['dnp3'], dest='test_protocol_name', help='test protocol name')

    args = parser.parse_args()
    sentence_embedding_model_name = args.model_name

    trained_model, train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list = \
        train_model(fieldname_description_dir, description_label_path, pcapcsv_file_dir, args.train_protocol_name, args.test_protocol_name, sentence_embedding_model_name, device)

    centroid_list = compute_centroid(trained_model, train_sentence_list, train_field_label_list, train_protocol_label_list, train_packet_group_label_list, device)

    with open('centroid_list.pkl', 'wb') as file:
        pickle.dump(centroid_list, file)

