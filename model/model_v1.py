import torch
import torch.nn as nn

#---------------------------------第1版------------------------------
#--------CGAN--------G: (z,c)-> f -> x'
#kernel_size是4，stride是1-2-2-2-2, padding是0-1-1-1-1
class Generator_v1(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,512,kernel_size=4,stride=1),
                nn.BatchNorm2d(512),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block2= nn.Sequential(
                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block3= nn.Sequential(
                nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block4= nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(64),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.convT=nn.ConvTranspose2d(64,  1,  kernel_size=4, stride=2, padding=1)
        self.tanh=nn.Tanh()
        #self.LRelu=nn.LeakyReLU()
    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim) or bool
        if type(c) == type(False):
           y=z
        else:
           y = torch.cat([z, c], axis=1)
        y = self.block1(y.view(y.size(0), y.size(1), 1, 1)) #1*1-->4*4,out_dim=512
        y = self.block2(y) # 4*4-->8*8
        y = self.block3(y) # 8*8-->16*16
        y = self.block4(y) # 16*16-->32*32
        y = self.tanh(self.convT(y))# 32*32-->64*64
        return y
#--------CGAN--------D: (z,c)-> f -> s
class Discriminator_v1(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#64->32
        self.lrelu=nn.LeakyReLU(0.2)
        self.block1=nn.Sequential(
                nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        self.block2=nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
        self.block3=nn.Sequential(
                nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
            )
        self.conv2=nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)#out_dim:1
    def forward(self, x, c=False):
        # x: (N, x_dim, 32, 32), c: (N, c_dim) or bool
        if type(c)!=type(False):
           c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
           x = torch.cat([x, c])
        y = self.lrelu(self.conv1(x))
        y = self.block1(y)#32->32
        y = self.block2(y)#32->32
        y2 = self.block3(y)#32->32
        y1 = self.conv2(y2)#32->29 :[-1,c,29,29]
        return y1,y2







