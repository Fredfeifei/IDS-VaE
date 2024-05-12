import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderByte(nn.Module):
    def __init__(self):
        super(EncoderByte, self).__init__()
        self.conv1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=256 if i == 0 else 8, out_channels=8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(8),
                nn.Dropout(0.35)
            ) for i in range(5)
        ])
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=1, padding=1)

        self.conv2_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=8 if i == 0 else 16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(16),
                nn.Dropout(0.3)
            ) for i in range(5)
        ])
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=16 if i == 0 else 32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3)
            ) for i in range(5)
        ])
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, packet_len, byte_len, _ = x.size()
        x = x.view(batch_size * packet_len, byte_len, -1).permute(0, 2, 1)

        for layer in self.conv1_layers:
            x = layer(x)

        x = self.maxpool1(x)
        x = self.avgpool(x)
        #x = self.maxpool1(x)

        for layer in self.conv2_layers:
            x = layer(x)

        x = self.maxpool2(x)
        x = self.avgpool(x)
        x = self.maxpool1(x)


        for layer in self.conv3_layers:
            x = layer(x)

        x = self.maxpool1(x)
        x = self.maxpool1(x)

        final_dim = x.size(-2) * x.size(-1)
        #print(final_dim)
        x = x.view(batch_size, packet_len, final_dim)
    
        return x


class EncoderPacket(nn.Module):
    def __init__(self, embed_size, num_heads, dropout_rate=0.35):
        super(EncoderPacket, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads),
                nn.LayerNorm(embed_size),
                nn.Dropout(dropout_rate)
            ) for _ in range(3)
        ])

    def forward(self, x):
        x = x.permute(1, 0, 2)
        for attn, norm, dropout in self.attention_layers:
            attn_output, _ = attn(x, x, x)
            x = x + dropout(attn_output)
            x = norm(x)
        return x.permute(1, 0, 2)

class EncoderPacket_Cnn(nn.Module):
    def __init__(self):
        super(EncoderPacket_Cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(20 * 64, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x1 = x.permute(0, 2, 1)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)

        x2 = x2.view(x2.size(0), -1)

        x2 = self.fc1(x2)
        x2 = self.fc_bn(x2)
        x2 = self.fc_relu(x2)

        x2 = self.fc2(x2)

        return x2

class WholeEncoder(nn.Module):
    def __init__(self, num_heads=8):
        super(WholeEncoder, self).__init__()
        self.encoder_byte = EncoderByte()
        self.encoder_packet = EncoderPacket(embed_size=256, num_heads=num_heads)
        self.encoder_packet_aggregrate = EncoderPacket_Cnn()
        self.mu = nn.Linear(128, 128)
        self.log_var = nn.Linear(128, 128)

    def forward(self, x):
        x = self.encoder_byte(x)
        x = self.encoder_packet(x)
        x = self.encoder_packet_aggregrate(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return x, mu, log_var

class Decoder(nn.Module):
    def __init__(self, output_channels=20, output_size=16, latent_dims=128):
        super(Decoder, self).__init__()
        self.output_channels = output_channels
        self.output_size = output_size
        self.latent_dims = latent_dims

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 20 * 16 * 16)
        )

    def forward(self, z):
        z = self.decoder(z)
        #print(z.shape)
        return z.view(-1, 20, 256)

class VAE(nn.Module):
    def __init__(self, is_train = True, channels=20, img_size=16, latent_dims=64):
        super(VAE, self).__init__()
        self.is_train = is_train
        self.encoder = WholeEncoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        if self.is_train:
            x, mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
        else:
            z, mu, log_var = self.encoder(x)
            
        return self.decoder(z), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD