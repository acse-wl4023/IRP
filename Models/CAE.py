import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channel, output_size):
        super(Encoder, self).__init__()

        self.input_channel=input_channel
        self.output_size=output_size

        self.conv1 = nn.Conv1d(self.input_channel, 4, kernel_size=8, padding='same', dilation=2)
        self.dropout1 = nn.Dropout(0.1)
        self.pool1 = nn.MaxPool1d(5)
        
        self.conv2 = nn.Conv1d(4, 4, kernel_size=8, padding='same', dilation=2)
        self.dropout2 = nn.Dropout(0.1)
        self.pool2 = nn.MaxPool1d(5)

        self.conv3 = nn.Conv1d(4, 1, kernel_size=8, padding='same', dilation=2)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.dropout3 = nn.Dropout(0.1)
        self.pool3 = nn.MaxPool1d(5)

        self.avg_pool = nn.AdaptiveAvgPool1d(1000)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1000, self.output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.leaky_relu(x)
        x = self.dropout3(x)
        x = self.pool3(x)
        
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, output_channel):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_channel = output_channel

        self.dense1 = nn.Linear(self.input_size, 512)  # Reduce feature dimensions to meet the requirement
        self.upsample1 = nn.Upsample(size=1024)  # First upsampling

        self.conv1 = nn.Conv1d(1, 1, kernel_size=8, padding=4, dilation=2)
        
        self.linear = nn.Linear(1018, 97149)
        #self.upsample2 = nn.Upsample(size=16384)  # Second upsampling to reach the target size
        
        self.conv2 = nn.Conv1d(1, self.output_channel, kernel_size=1, padding='same')  # Final convolution to adjust channel
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.dense1(x)
        x = x.view(-1, 1, 512)  # Reshape to match the convolution input dimensions
        
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
        #x = self.upsample2(x)
        x = self.linear(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Autoencoder, self).__init__()
        self.channels = channels
        self.latent_size = latent_size
        self.encoder = Encoder(self.channels, self.latent_size)
        self.decoder = Decoder(self.latent_size, self.channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
