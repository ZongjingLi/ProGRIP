import torch
import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.net(input)
class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)

    def __getitem__(self,item):
        return self.net[item]

    def forward(self, input):
        return self.net(input)

class FCTLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)
class FCTBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCTLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)

    def __getitem__(self,item):
        return self.net[item]

    def forward(self, input):
        return self.net(input)

class VAE(torch.nn.Module):

    # For simplicity we assume encoder and decoder share same hyperparameters
    def __init__(self, in_features, num_hidden_layers=2, hidden_ch=128,
                                                 latent_features=5, beta=10):
        super().__init__()

        self.beta = beta

        self.encoder = FCBlock(in_features=in_features,
                               hidden_ch = hidden_ch,
                               num_hidden_layers=num_hidden_layers,
                               out_features=2*latent_features,
                               outermost_linear=True)

        self.decoder = FCBlock(in_features = latent_features,
                               hidden_ch = hidden_ch,
                               num_hidden_layers=num_hidden_layers,
                               out_features=in_features,
                               outermost_linear=True)

    # Returns reconstruction, reconstruction loss, and KL loss
    def forward(self, x):

        # Split encoding into mu/logvar, reparameterize, and decode

        mu, log_var = self.encoder(x).chunk(2,dim=1)

        std    = torch.exp(0.5*log_var)
        eps    = torch.randn_like(std)
        sample = mu + (eps * std)

        recon  = self.decoder(sample)

        # Compute reconstruction and kld losses

        recon_loss = torch.linalg.norm(recon-x,dim=1)
        kld_loss   = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        return recon, recon_loss, self.beta*kld_loss

if __name__ == "__main__":
    inputs = torch.randn([7,3])
    net = FCBlock(128,3,3,1) 
    net2 = FCTBlock(128,3,3,1)
    outputs = net(inputs)
    outputs2 = net2(inputs)
    print(outputs,outputs2)