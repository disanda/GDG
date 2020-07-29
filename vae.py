from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import models.BF_v1 as model
import os
#-------------setting---------------
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = True
torch.manual_seed(args.seed)
device = torch.device("cuda")

#------------------dataSet----------------
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),batch_size=args.batch_size, shuffle=True, **kwargs)

#-------------------loss function--------------------------
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

Encode = model.Encode().to(device)
Decode = model.Decode().to(device)
optimizerE = optim.Adam(Encode.parameters(), lr=1e-3)
optimizerD = optim.Adam(Decode.parameters(), lr=1e-3)

def train(epoch):
    Encode.train()
    Decode.train()
    train_loss_KLD = 0
    train_loss_BCE = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        #-----------------training Encode---------------
        optimizerE.zero_grad()
        mu, logvar = Encode(data)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_KLD.backward()
        train_loss_KLD += loss_KLD.item()
        optimizerE.step()
        #-----------------training Decode---------------
        optimizerD.zero_grad()
        x_fake = Decode(mu, logvar)
        loss_BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        loss_BCE.backward()
        train_loss_BCE += loss_BCE.item()
        optimizerD.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_KLD: {:.6f}\tLoss_BCE: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),loss_KLD.item() / len(data),loss_BCE.item() / len(data)))
    print('====> Epoch: {} Average loss KLD: {:.4f}--Average loss BCE: {:.4f}'.format(epoch, train_loss_KLD / len(train_loader.dataset) , train_loss_BCE / len(train_loader.dataset)))

path_dir='results_1'
if not os.path.exists(path_dir):
    os.mkdir(path_dir)

def test(epoch):
    Decode.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),path_dir+'/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),path_dir+'/sample_' + str(epoch) + '.png')