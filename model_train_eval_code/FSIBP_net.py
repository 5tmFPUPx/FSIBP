import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class FSIBP_Net(nn.Module):
    def __init__(self, vocab_size=257, embed_size=8, field_hidden_size=16, sentence_hidden_size = 32, packet_hidden_size=32, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.field_hidden_size = field_hidden_size
        self.sentence_hidden_size = sentence_hidden_size
        self.packet_hidden_size  = packet_hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=256)
        self.field_gru = nn.GRU(embed_size, field_hidden_size, batch_first=True)
        self.sentence_gru = nn.GRU(field_hidden_size, sentence_hidden_size, batch_first=True)
        self.packet_gru = nn.GRU(sentence_hidden_size, packet_hidden_size, batch_first=True)

        self.LN = nn.LayerNorm(packet_hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(packet_hidden_size, sentence_hidden_size),
            #nn.LayerNorm(sentence_hidden_size),
            nn.ReLU(),
            nn.Linear(sentence_hidden_size, field_hidden_size)
        )
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    def init_field_hidden(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.field_hidden_size))
    
    def init_sentence_hidden(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.sentence_hidden_size))
    
    def init_packet_hidden(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.packet_hidden_size))

    def forward(self, sentence_list, field_lengths, longest_field_len, sentence_list_size, sentence_len):
        field_hidden = self.init_field_hidden(batch_size=sentence_list_size*sentence_len)
        sentence_hidden = self.init_sentence_hidden(batch_size=sentence_list_size)
        packet_hidden = self.init_packet_hidden(batch_size=1)

        field_hidden = field_hidden.to(self.device)
        sentence_hidden = sentence_hidden.to(self.device)
        packet_hidden = packet_hidden.to(self.device)
        
        field_embedding = self.embedding(sentence_list)
        field_embedding = field_embedding.view(sentence_list_size*sentence_len, longest_field_len, self.embed_size)

        packed_field_embedding = pack_padded_sequence(field_embedding, field_lengths, batch_first=True, enforce_sorted=False)

        _, field_hn = self.field_gru(packed_field_embedding, field_hidden)
        field_hn = field_hn.view(sentence_list_size, sentence_len, self.field_hidden_size)

        _, sentence_hn = self.sentence_gru(field_hn, sentence_hidden)
        sentence_hn = sentence_hn.view(1, sentence_list_size, self.sentence_hidden_size)

        packet_gru_output, packet_hn = self.packet_gru(sentence_hn, packet_hidden)

        packet_gru_output = packet_gru_output.view(sentence_list_size, self.packet_hidden_size)

        packet_gru_output = packet_gru_output.view(sentence_list_size, 1, self.packet_hidden_size)

        return packet_gru_output