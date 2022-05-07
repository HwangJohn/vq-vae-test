from torch import nn 
from torch.nn import functional as F

from model.vqvae import VectorQuantizer, VectorQuantizerEMA

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_norm, activation):
        super(Residual, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU(True)
        elif activation == "gelu":
            self.activation = nn.GELU(True)

        self.conv3x3 = nn.Conv2d(in_channels=in_channels,
                        out_channels=num_residual_hiddens,
                        kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1x1 = nn.Conv2d(in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1, stride=1, bias=False)  

        if use_norm:
            self.bn1 = nn.BatchNorm2d(num_residual_hiddens)
            self.bn2 = nn.BatchNorm2d(num_hiddens)
            self._block = nn.Sequential(
                self.conv3x3,
                self.bn1,
                self.activation,
                self.conv1x1,
                self.bn2,
                self.activation,
            )
        else:
            self._block = nn.Sequential(
                self.activation,
                self.conv3x3,
                self.activation,
                self.conv1x1
            )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_norm, activation):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, use_norm, activation)
                             for _ in range(self._num_residual_layers)])

        if activation == "relu":
            self.activation = nn.ReLU(True)
        elif activation == "gelu":
            self.activation = nn.GELU(True)                            

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return self.activation(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_norm, activation):
        super(Encoder, self).__init__()

        self.use_norm = use_norm

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens,
                                             use_norm=self.use_norm,
                                             activation=activation)

        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(num_hiddens//2)
            self.bn2 = nn.BatchNorm2d(num_hiddens)
            self.bn3 = nn.BatchNorm2d(num_hiddens)

        if activation == "relu":
            self.activation = nn.ReLU(True)
        elif activation == "gelu":
            self.activation = nn.GELU(True)


    def forward(self, inputs):

        if self.use_norm:
            x = self._conv_1(inputs)
            x = self.bn1(x)
            x = self.activation(x)
            
            x = self._conv_2(x)
            x = self.bn2(x)
            x = self.activation(x)
            
            x = self._conv_3(x)
            x = self.bn3(x)
        else:
            x = self._conv_1(inputs)
            x = self.activation(x)
            
            x = self._conv_2(x)
            x = self.activation(x)
            
            x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_norm, activation):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens,
                                             use_norm=use_norm,
                                             activation=activation)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

        if activation == "relu":
            self.activation = nn.ReLU(True)
        elif activation == "gelu":
            self.activation = nn.GELU(True)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = self.activation(x)
        
        return self._conv_trans_2(x)

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0, use_norm=False, activation="relu"):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens,
                                use_norm,
                                activation)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens,
                                use_norm,
                                activation)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity