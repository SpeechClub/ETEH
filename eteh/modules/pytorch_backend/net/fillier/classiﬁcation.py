import torch


class Classification(torch.nn.Module):

    def __init__(self, embedding_channel, embedding_size, output_size, dropout_rate, conv_1x1=False):
        super(Classification, self).__init__()
        self.embedding_channel = embedding_channel
        self.embedding_size = embedding_size
        self.max_pooling = torch.nn.MaxPool1d(embedding_size)
        self.conv_1x1 = conv_1x1
        if conv_1x1:
            self.head = torch.nn.Conv1d(embedding_channel, embedding_channel, 1)
        self.classify = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(embedding_channel, output_size)
        )
        

    def forward(self, x):
        assert x.size(1) == self.embedding_channel
        assert x.size(2) == self.embedding_size
        assert x.size(3) == 1
        x = x.squeeze(-1)
        x = self.max_pooling(x)
        if self.conv_1x1:
            x = self.head(x)
        x = x.squeeze(-1)
        x = self.classify(x)
        return x
        