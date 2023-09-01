import torch
import numpy as np
import argparse
import pickle
import os
from pcapcsv2data import generate_data_and_labels
from torch.utils.data import Dataset, DataLoader
from FSIBP_net import FSIBP_Net


class eval_pcapcsv_dataset(Dataset):
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
    sentence_list = [[[field[i:i+2] for i in range(0, len(field), 2)] for field in sentence] for sentence in sentence_list]  
    sentence_list = [[[int(hex_str, 16) for hex_str in field_list] for field_list in sentence] for sentence in sentence_list] # Hex to DEC

    field_lengths = []
    for sentence in sentence_list:
        for field in sentence:
            field_lengths.append(len(field))
    longest_field_length = max(field_lengths)
    sentence_list_size = len(sentence_list) # batch_size
    sentence_length = len(sentence_list[0])
    # padding
    for s in range(sentence_list_size):
        for l in range(sentence_length):
            if(len(sentence_list[s][l])) < longest_field_length:
                sentence_list[s][l] += [256] * (longest_field_length - len(sentence_list[s][l]))
    
    return sentence_list, field_lengths, longest_field_length, sentence_list_size, sentence_length


def eval_centroid(test_data_loader, model, centroid_list, device):
    correct = 0
    total = 0
    top2_correct = 0

    similarities_list = []
    with torch.no_grad():
        for sentences, field_label, _, _ in test_data_loader:
            sentences, field_lengths, longest_field_len, sentences_size, sentence_len = field_data_process(sentences)
            sentences = torch.IntTensor(np.array(sentences))
            field_label = torch.LongTensor(np.array(field_label))
            sentences = sentences.to(device) 
            field_label = field_label.to(device)

            packet_gru_output = model(sentences, field_lengths, longest_field_len, sentences_size, sentence_len)
            packet_gru_output = packet_gru_output.view(sentences_size, 32) 
            packet_gru_output = packet_gru_output.cpu().numpy()
            field_label = field_label.cpu().numpy()

            centroids = np.array(centroid_list)
            cosine_similarities = np.dot(packet_gru_output, centroids.T) / (np.linalg.norm(packet_gru_output, axis=1, keepdims=True) * np.linalg.norm(centroids, axis=1))
            closest_centroids = np.argmax(cosine_similarities, axis=1)

            closest_centroids_sort = np.argsort(cosine_similarities, axis=1)
            top2_closest_centroids = closest_centroids_sort[:, [-2,-1]]

            for i in range(len(closest_centroids)):
                if closest_centroids[i] == field_label[i]:
                    correct += 1
                if field_label[i] in top2_closest_centroids[i]:
                    top2_correct += 1
            total += len(field_label)

            similarities_list.extend([cosine_similarities])

    print('centroid total:', total)
    print('centroid top1 correct:', correct)
    print('centroid top1 Accuracy: %.2f %%' % (100 * correct / total))
    print('centroid top2 correct:', top2_correct)
    print('centroid top2 Accuracy: %.2f %%' % (100 * top2_correct / total))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("The model will be running on", device, "device\n")

    fieldname_description_dir = os.path.join('..', 'data', 'field-name-description')
    description_label_path = os.path.join('..', 'data', 'description-label', 'description_label.csv')
    pcapcsv_file_dir = os.path.join('..', 'data', 'pcapcsv')

    parser = argparse.ArgumentParser()
    parser.add_argument('-tep', '--testprotocol', nargs='+', default=['dnp3'], dest='test_protocol_name', help='test protocol name')

    args = parser.parse_args()
    test_protocol_name = args.test_protocol_name

    test_sentence_list, test_field_label_list, test_protocol_label_list, test_packet_group_label_list = generate_data_and_labels(test_protocol_name, fieldname_description_dir, description_label_path, pcapcsv_file_dir)
    test_dataset = eval_pcapcsv_dataset(test_sentence_list, test_field_label_list, test_protocol_label_list, test_packet_group_label_list)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    with open('centroid_list.pkl', 'rb') as file:
        centroid_list = pickle.load(file)

    trained_model = FSIBP_Net()
    trained_model.load_state_dict(torch.load('trained_FSIBP.pt'))
    trained_model.eval()
    eval_centroid(test_data_loader, trained_model, centroid_list, device)