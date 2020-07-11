
import functools
import numpy as np
import pylib as py
import tensorboardX
import torch
import loss
import gradient_penalty as gp
import tqdm
import data
import module
import torchvision
import os

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
import argparse
parser = argparse.ArgumentParser(description='the training args')
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--dataset_name',default='mnist')#choices=['cifar10', 'fashion_mnist', 'mnist', 'celeba', 'pose']
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--epochs', type=int, default=10)
#parser.add_argument('--lr', type=float, default=0.0002,help='learning_rate')
parser.add_argument('--beta_1', type=float, default=0.5)
parser.add_argument('--n_d', type=int, default=1)# d updates per g update
parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
parser.add_argument('--gradient_penalty_mode', default='none', choices=['none', '1-gp', '0-gp', 'lp'])
parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
parser.add_argument('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])
parser.add_argument('--img_size',type=int,default=64)
args = parser.parse_args()

args.experiment_name = 'mnist-3-CGAN'
args.lr = 0.0002


# output_dir
if args.experiment_name == 'none':
    args.experiment_name = '%s_%s_%s_%s' % (args.dataset_name, args.adversarial_loss_mode,args.batch_size,args.epochs)
    if args.gradient_penalty_mode != 'none':
        args.experiment_name += '_%s_%s' % (args.gradient_penalty_mode, args.gradient_penalty_sample_mode)
output_dir = os.path.join('output', args.experiment_name)

if not os.path.exists('output'):
    os.mkdir('output')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#os.mkdirs(output_dir)

# save settings
import yaml
with open(os.path.join(output_dir, 'settings.yml'), "w", encoding="utf-8") as f:
    yaml.dump(args, f)

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
if args.dataset_name in ['cifar10', 'fashion_mnist', 'mnist']:  # 32x32
    data_loader, shape = data.make_dataset(args.dataset_name, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 3

elif args.dataset_name == 'celeba':  # 64x64
    data_loader, shape = data.make_dataset(args.dataset_name, args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4

elif args.dataset_name.find('pose') != -1:  # 32x32
    #img_paths = os.listdir('data/pose')
    #img_payhs = list(filter(lambda x:x.endswith('png'),img_paths))
    data_loader, shape = data.make_dataset(args.dataset_name,args.batch_size,args.img_size,pin_memory=use_gpu)
    n_G_upsamplings = n_D_downsamplings = 4  # 3 for 32x32 and 4 for 64x64

# ==============================================================================
# =                                   model                                    =
# ==============================================================================


# setup the normalization function for discriminator
if args.gradient_penalty_mode == 'none':
    d_norm = 'batch_norm'
else:  # cannot use batch normalization with gradient penalty
    d_norm = args.gradient_penalty_d_norm

# networks
G = module.ConvGenerator(args.z_dim, shape[-1], n_upsamplings=n_G_upsamplings).to(device)
D = module.ConvDiscriminator(shape[-1], n_downsamplings=n_D_downsamplings, norm=d_norm).to(device)
print(G)
print(D)

# adversarial_loss_functions
d_loss_fn, g_loss_fn = loss.get_adversarial_losses_fn(args.adversarial_loss_mode)

# optimizer
G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))

# ==============================================================================
# =                                    run                                     =
# ==============================================================================
ep, it_d, it_g = 0, 0, 0

# sample
sample_dir = os.path.join(output_dir, 'samples_training')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

# main loop
writer = tensorboardX.SummaryWriter(os.path.join(output_dir, 'summaries'))
z = torch.randn(100, args.z_dim, 1, 1).to(device)  # a fixed noise for sampling

@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

for ep_ in tqdm.trange(args.epochs):#epoch:n*batch
    ep = ep+1
    G.train()
    D.train()
    for i in tqdm.tqdm(data_loader, desc='Inner Epoch Loop'):#batch_size
        if args.dataset_name == 'cifar10' or 'mnist':#数据有标签
            x,c = i
            x = x.to(device)
        else:
            x = i
            x = x.to(device)

#training D
        z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
        x_fake = G(z).detach()
        x_real_score = D(x)
        x_fake_score = D(x_fake)
        bce = torch.nn.BCEWithLogitsLoss()
        r_loss = bce(x_real_score, torch.ones_like(x_real_score))
        f_loss = bce(x_fake_score, torch.zeros_like(x_fake_score))
        gp_value = gp.gradient_penalty(functools.partial(D), x, x_fake, gp_mode=args.gradient_penalty_mode, sample_mode=args.gradient_penalty_sample_mode)
        D_loss = (r_loss + f_loss) + gp_value * args.gradient_penalty_weight
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        D_loss_dict={'d_loss': r_loss + f_loss, 'gp': gp_value}
        it_d += 1
        for k, v in D_loss_dict.items():
            writer.add_scalar('D/%s' % k, v.data.cpu().numpy(), global_step=it_d)

#training G
        if it_d % args.n_d == 0:
            G_loss_dict = train_G(labels)
            #CGAN: (x,c)->G->s
            z = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
            x_fake = G(z,c)
            x_fake_score = D(x_fake,c)
            G_loss = g_loss_fn(x_fake_score)
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            G_loss_dict = {'g_loss': G_loss}

            it_g += 1
            for k, v in G_loss_dict.items():
                writer.add_scalar('G/%s' % k, v.data.cpu().numpy(), global_step=it_g)

        # sample
        if it_g % 100 == 0:
            x_fake = sample(z)
            #x_fake = np.transpose(x_fake.data.cpu().numpy(), (0, 2, 3, 1))#(n,w,h,c)
            torchvision.utils.save_image(x_fake,sample_dir+'/%d.jpg'%(it_g), nrow=10)

# save checkpoint
with torch.no_grad():
	G.eval()
	D.eval()
	torch.save({'epoch': ep + 1,'G': G.state_dict(),'D': D.state_dict()},'%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1))#save model
