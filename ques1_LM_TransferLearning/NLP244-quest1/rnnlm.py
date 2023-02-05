import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_mtrix,
        in_embedding_dim,
        n_hidden,
        n_layers,
        dropout=0.5,
        rnn_type="lstm",  # can be elman, lstm, gru
    ):
        super(RNNModel, self).__init__()

        if rnn_type == "elman":
            print("\nelman rnn\n")
            self.rnn = nn.RNN(
                in_embedding_dim,
                n_hidden,
                n_layers,
                nonlinearity="tanh",
                dropout=dropout,
                bidirectional=False,
            )
        elif rnn_type == "lstm":
            print("\nlstm model\n")
            self.rnn = nn.LSTM(
                in_embedding_dim,
                n_hidden,
                n_layers,
                # nonlinearity="tanh",
                dropout=dropout,
                bidirectional=False,
            ) 
        elif rnn_type == "bilstm":
            print("\nbilstm model\n")
            self.rnn = nn.LSTM(
                in_embedding_dim,
                n_hidden,
                n_layers,
                # nonlinearity="tanh",
                dropout=dropout,
                bidirectional=True,
            )                       

        else:
            print("\ngru model\n")
            self.rnn = nn.GRU(
                in_embedding_dim,
                n_hidden,
                n_layers,
                # nonlinearity="tanh",
                dropout=dropout,
                bidirectional=False,
            )           
        self.model_type = rnn_type        
        self.in_embedder = nn.Embedding(vocab_size, in_embedding_dim)
        self.in_embedder.weight = nn.Parameter(torch.tensor(emb_mtrix, dtype=torch.float32))
        self.in_embedder.weight.requires_grad = False #use True to unfreeze the embedding layer
        self.dropout = nn.Dropout(dropout)
        if self.model_type == "bilstm":
          self.pooling = nn.Linear(n_hidden*2, vocab_size) #BiLSTM
        else:
          self.pooling = nn.Linear(n_hidden, vocab_size)
        self.init_weights()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.vocab_size = vocab_size


    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.in_embedder.weight, -initrange, initrange)
        nn.init.zeros_(self.pooling.bias)
        nn.init.uniform_(self.pooling.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.dropout(self.in_embedder(input))
        # hidden = self.init_hidden(20)
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        pooled = self.pooling(output)
        pooled = pooled.view(-1, self.vocab_size)
        return F.log_softmax(pooled, dim=1), hidden

    def init_hidden(self, batch_size):
        if (self.model_type == "elman" or self.model_type == "gru"):
          weight = next(self.parameters())
          return weight.new_zeros(self.n_layers, batch_size, self.n_hidden)
        elif self.model_type == "bilstm":
          hidden = torch.zeros(self.n_layers*2, batch_size, self.n_hidden) #BiLSTM
          cell = torch.zeros(self.n_layers*2, batch_size, self.n_hidden) #BiLSTM
          return hidden, cell          
        else:
          hidden = torch.zeros(self.n_layers, batch_size, self.n_hidden)
          cell = torch.zeros(self.n_layers, batch_size, self.n_hidden)
          return hidden, cell


    @staticmethod
    def load_model(path):
        with open(path, "rb") as f:
            model = torch.load(f)
            model.rnn.flatten_parameters()
            return model
