import module
import torch
import torchvision
import os

shape = [32, 32, 1]
n_G_upsamplings = 3 #32*32

# others
#use_gpu = torch.cuda.is_available()
#device = torch.device("cuda" if use_gpu else "cpu")
#G = module.ConvGenerator(128, shape[-1], n_upsamplings=n_G_upsamplings).to(device)

G = module.ConvGenerator(128, shape[-1], n_upsamplings=n_G_upsamplings)
model_dir = './pre-model/Ep11-hingev2-0gp-line.ckpt'
ckpt=torch.load(model_dir,map_location=torch.device('cpu'))#dict
#print(ckpt)
G.load_state_dict(ckpt['G'])
#D_optimizer.load_state_dict(ckpt['D_optimizer'])
#G_optimizer.load_state_dict(ckpt['G_optimizer'])

sample_dir='./result'
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

@torch.no_grad()
def sample(z):
    G.eval()
    return G(z)

for i in range(100):
	z = torch.randn(1, 128, 1, 1).to('cpu')  # a fixed noise for sampling
	x_fake = sample(z)
#print(x_fake.shape)
	torchvision.utils.save_image(x_fake,sample_dir+'/a%d.jpg'%(i))


load checkpoint if exists
ckpt_dir = os.path.join(output_dir, 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

ckpt_path = os.path.join(ckpt_dir, 'xxx.ckpt')
ckpt=torch.load(ckpt_path)
ep, it_d, it_g = ckpt['ep'], ckpt['it_d'], ckpt['it_g']
D.load_state_dict(ckpt['D'])
G.load_state_dict(ckpt['G'])
D_optimizer.load_state_dict(ckpt['D_optimizer'])
G_optimizer.load_state_dict(ckpt['G_optimizer'])
