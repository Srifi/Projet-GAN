##Personal notes
#cd Documents/Work/ProTech

#sources:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://medium.com/mlearning-ai/how-to-improve-image-generation-using-wasserstein-gan-1297f449ca75

##GAN

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import uuid
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Look for rundir to find all paths to potentially modifiy
# dataroot should also be checked
# dataroot folder should contain sub-folder(s) containing images


# Set random seed for reproducibility
# manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Discriminator training rate (this is the number of time G will be trained for one training of D)
DTP = 1


# Unique run id
runid = str(uuid.uuid4())

# Root directory for dataset
dataroot = "RandomDataset"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 128

# Lamda for the gradient penalty
LAMBDA_GP = 10

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

##

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.Grayscale(num_output_channels=nc),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) for nc=3
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

##

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
print(netG)

##

#Discriminator code
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


##
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup AdamW optimizers for both G and D
#optimizerD = optim.SGD(netD.parameters(), lr=0.0001)
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))


## Gradient Penalty

def gradient_penalty( critic, real_image, fake_image, device=device):
    batch_size, channel, height, width= real_image.shape
    #alpha is selected randomly between 0 and 1
    alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width).to(device)
    # interpolated image=randomly weighted average between a real and fake image
    #interpolated image ← alpha *real image  + (1 − alpha) fake image
    interpolatted_image=(alpha*real_image) + (1-alpha) * fake_image

    # calculate the critic score on the interpolated image
    interpolated_score= critic(interpolatted_image)

    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=interpolatted_image,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty


## Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

if not os.path.exists(f'.\\runs\\{runid}\\images'): #rundir
    os.makedirs(f'.\\runs\\{runid}\\images') #rundir

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):


        # (1) Update D network: use of Wasserstein loss with gradient penalty

        # Zero the gradient
        netD.zero_grad()

        # Training on a real image batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        outputD_real = netD(real_cpu).view(-1)
        D_x = outputD_real.mean().item()

        # Training on a fake image batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        outputD_fake = netD(fake.detach()).view(-1)
        D_G_z1 = outputD_fake.mean().item()

        # Gradient penalty and loss calculation
        gp = gradient_penalty(netD, real_cpu, fake, device)
        errD = -(torch.mean(outputD_real) -torch.mean(outputD_fake)) + LAMBDA_GP * gp # discriminator loss
        errD.backward(retain_graph=True)

        # Update the NN
        optimizerD.step()


        # (2) Update G network: maximize log(D(G(z)))

        # Zero the gradient
        netG.zero_grad()

        # Generate fake images
        outputG = netD(fake).view(-1)
        D_G_z2 = outputG.mean().item()

        # Loss calculation
        errG = -torch.mean(outputG) # generator loss
        errG.backward(retain_graph=True)

        # Update the NN
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if i % 25 == 0:
            # Output training stats
            print('[%d/%d][%d/%d] Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs-1, i, len(dataloader)-1,
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            plt.title("Generated Images")
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
            plt.savefig(f'.\\runs\\{runid}\\images\\epoch{epoch}_batch{i}.jpg') #rundir

        iters += 1

# Loss visualize
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'.\\runs\\{runid}\\losses.jpg') #rundir



