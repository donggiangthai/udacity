import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-50 as similar to the Notebook 1 practice
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.2):
        super(DecoderRNN, self).__init__()
        # TODO: Complete this function
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Define the LSTM
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        # Define a Fully connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        # TODO: Complete this function
        # Concatenating the features and embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # clean out hidden state, trying to fix the issue of same result for every image
        hidden = (torch.randn(self.num_layers, inputs.shape[0], self.hidden_size, device=inputs.device), \
                  torch.randn(self.num_layers, inputs.shape[0], self.hidden_size, device=inputs.device))

        lstm_outputs, hidden = self.lstm(inputs, hidden)

        outputs = self.dropout(self.fc(lstm_outputs))

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"
        predicted_sentence = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            hiddens = hiddens.squeeze(1)
            outputs = self.fc(hiddens)

            # Get the max index
            _, predicted = torch.max(outputs, 1)
            predicted_sentence.append(predicted.item())

            # if predicted is <end>, break the loop
            if predicted == 1:
                break
            
            # Update the inputs
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence