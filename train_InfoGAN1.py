import argparse
import json
import loss
import numpy as np
import os
import PIL.Image as Image
import tensorboardX
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as tforms
import torchlib
import data

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# command line arguments
parser = argparse.ArgumentParser()
# training
parser.add_argument('--weight_norm', dest='weight_norm', choices=['none', 'spectral_norm', 'weight_norm'], default='spectral_norm')
# others
parser.add_argument('--experiment_name', dest='experiment_name', default='infoGANv1_')

# parse arguments
args = parser.parse_args()
# pra
z_dim = 100
epoch = 50
batch_size = 64
d_learning_rate = 0.0002
g_learning_rate = 0.001
n_d = 1
# loss
loss_mode = 'hinge_v2' #choices=['gan', 'lsgan', 'wgan', 'hinge_v1', 'hinge_v2']
gp_mode = 'none' #choices=['none', 'dragan', 'wgan-gp'], default='none']
gp_coef = 1.0 
norm = 'none' #['none', 'batch_norm', 'instance_norm']
# ohters
experiment_name = args.experiment_name+loss_mode

# save settings
if not os.path.exists('./output/%s' % experiment_name):
    os.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

ckpt_dir = './output/%s/checkpoints' % experiment_name
save_dir = './output/%s/sample_training' % experiment_name
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
c_dim = 10

# ==============================================================================
# =                                   setting                                  =
# ==============================================================================

# data
train_loader = data.getDataloader(batch_size,use_gpu)

# model
import models.ACGAN as model
D = model.Discriminator(x_dim=3, c_dim=c_dim, norm=norm, weight_norm=args.weight_norm).to(device)
G = model.Generator(z_dim=z_dim, c_dim=c_dim).to(device)

# gan loss function
d_loss_fn, g_loss_fn = model.get_losses_fn(loss_mode)

# optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))


# ==============================================================================
# =                                    train                                   =
# ==============================================================================
start_ep = 0

# writer
writer = tensorboardX.SummaryWriter('./output/%s/summaries' % experiment_name)

# run
z_sample = torch.randn(c_dim * 10, z_dim).to(device)
c_sample = torch.tensor(np.concatenate([np.eye(c_dim)] * 10), dtype=z_sample.dtype).to(device)
for ep in range(start_ep, epoch):
    for i, (x, _) in enumerate(train_loader):
        step = ep * len(train_loader) + i + 1
        D.train()
        G.train()

        # train D
        x = x.to(device)
        c_dense = torch.tensor(np.random.randint(c_dim, size=[batch_size])).to(device)
        z = torch.randn(batch_size, z_dim).to(device)
        c = torch.tensor(np.eye(c_dim)[c_dense.cpu().numpy()], dtype=z.dtype).to(device)

        x_f = G(z, c).detach()
        x_gan_logit, _ = D(x)
        x_f_gan_logit, x_f_c_logit = D(x_f)

        d_x_gan_loss, d_x_f_gan_loss = d_loss_fn(x_gan_logit, x_f_gan_logit)
        d_x_f_c_logit = torch.nn.functional.cross_entropy(x_f_c_logit, c_dense)
        gp = model.gradient_penalty(D, x, x_f, mode=gp_mode)
        d_loss = d_x_gan_loss + d_x_f_gan_loss + gp * gp_coef + d_x_f_c_logit

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/d_gan_loss', (d_x_gan_loss + d_x_f_gan_loss).data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/d_q_loss', d_x_f_c_logit.data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        # train G
        if step % n_d == 0:
            c_dense = torch.tensor(np.random.randint(c_dim, size=[batch_size])).to(device)
            c = torch.tensor(np.eye(c_dim)[c_dense.cpu().numpy()], dtype=z.dtype).to(device)
            z = torch.randn(batch_size, z_dim).to(device)

            x_f = G(z, c)
            x_f_gan_logit, x_f_c_logit = D(x_f)

            g_gan_loss = g_loss_fn(x_f_gan_logit)
            d_x_f_c_logit = torch.nn.functional.cross_entropy(x_f_c_logit, c_dense)
            g_loss = g_gan_loss + d_x_f_c_logit

            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalar('G/g_gan_loss', g_gan_loss.data.cpu().numpy(), global_step=step)
            writer.add_scalar('G/g_q_loss', d_x_f_c_logit.data.cpu().numpy(), global_step=step)

        # display
        if step % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (ep, i + 1, len(train_loader)))

        # sample
        if step % 100 == 0:
            G.eval()
            x_f_sample = (G(z_sample, c_sample) + 1) / 2.0
            torchvision.utils.save_image(x_f_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, len(train_loader)), nrow=10)
            
    torchlib.save_checkpoint({'epoch': ep + 1,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'd_optimizer': d_optimizer.state_dict(),
                              'g_optimizer': g_optimizer.state_dict()},
                             '%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1),
                             max_keep=2)
