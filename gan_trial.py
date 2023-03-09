# Imported libraries
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import cv2


# Configurable variables
NUM_EPOCHS = 100
NOISE_DIMENSION = 64
BATCH_SIZE = 16
TRAIN_ON_GPU = False
UNIQUE_RUN_ID = str(uuid.uuid4())
PRINT_STATS_AFTER_BATCH = 50
OPTIMIZER_LR = 0.0002
OPTIMIZER_BETAS = (0.5, 0.999)
GENERATOR_OUTPUT_IMAGE_SHAPE = 64 * 64 * 1

# Paths
IMG_DIR = ''
ANNOTATIONS_FILE = ''


# Speed ups
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


# Generator
class Generator(nn.Module):
  """
    Vanilla GAN Generator
  """
  def __init__(self,):
    super().__init__()
    self.layers = nn.Sequential(
      # First upsampling
      nn.Linear(NOISE_DIMENSION, 128, bias=False),
      nn.BatchNorm1d(128, 0.8),
      nn.LeakyReLU(0.25),
      # Second upsampling
      nn.Linear(128, 256, bias=False),
      nn.BatchNorm1d(256, 0.8),
      nn.LeakyReLU(0.25),
      # Third upsampling
      nn.Linear(256, 512, bias=False),
      nn.BatchNorm1d(512, 0.8),
      nn.LeakyReLU(0.25),
      # Final upsampling
      nn.Linear(512, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    """Forward pass"""
    return self.layers(x)


# Discriminator
class Discriminator(nn.Module):
  """
    Vanilla GAN Discriminator
  """
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE, 1024),
      nn.LeakyReLU(0.25),
      nn.Linear(1024, 512),
      nn.LeakyReLU(0.25),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.25),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    """Forward pass"""
    return self.layers(x)


#Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file = ANNOTATIONS_FILE, img_dir = IMG_DIR, transform=None, target_transform=None):
        annotations_file = annotations_file
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_dim = (64, 64)
        self.data = []
        for img in os.listdir(img_dir):
            self.data.append([img_dir + '/' + img, ''])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx] #consider class labels?
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.float()


# Prepare the real data
def prepare_dataset():
  """ Prepare dataset through DataLoader """
  # Prepare Dataset
  dataset = CustomImageDataset()
  # Batch and shuffle data with DataLoader
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
  # Return dataset through DataLoader
  return trainloader


def get_device():
  """ Retrieve device based on settings and availability. """
  return torch.device("cuda:0" if torch.cuda.is_available() and TRAIN_ON_GPU else "cpu")


def make_directory_for_run():
  """ Make a directory for this training run. """
  print(f'Preparing training run {UNIQUE_RUN_ID}')
  if not os.path.exists('./runs'):
    os.mkdir('./runs')
  os.mkdir(f'./runs/{UNIQUE_RUN_ID}')


# Image generation by the generator
def generate_image(generator, epoch = 0, batch = 0, device=get_device()):
  """ Generate subplots with generated examples. """
  images = []
  noise = generate_noise(BATCH_SIZE, device=device)
  generator.eval()
  images = generator(noise)
  plt.figure(figsize=(10, 10))
  for i in range(16):
    # Get image
    image = images[i]
    # Convert image back onto CPU and reshape
    image = image.cpu().detach().numpy()
    image = np.reshape(image, (64, 64))
    # Plot
    plt.subplot(4, 4, i+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
  if not os.path.exists(f'./runs/{UNIQUE_RUN_ID}/images'):
    os.mkdir(f'./runs/{UNIQUE_RUN_ID}/images')
  plt.savefig(f'./runs/{UNIQUE_RUN_ID}/images/epoch{epoch}_batch{batch}.jpg')


# Save the generator/discriminator status
def save_models(generator, discriminator, epoch):
  """ Save models at specific point in time. """
  torch.save(generator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/generator_{epoch}.pth')
  torch.save(discriminator.state_dict(), f'./runs/{UNIQUE_RUN_ID}/discriminator_{epoch}.pth')


def print_training_progress(batch, generator_loss, discriminator_loss):
  """ Print training progress. """
  print('Losses after mini-batch %5d: generator %e, discriminator %e' %
        (batch, generator_loss, discriminator_loss))


# Initialization
def initialize_models(device = get_device()):
  """ Initialize Generator and Discriminator models """
  generator = Generator()
  discriminator = Discriminator()
  # Move models to specific device
  generator.to(device)
  discriminator.to(device)
  # Return models
  return generator, discriminator






    #!!!!!!!!!!!Custom Loss Function

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)



def initialize_loss():
  """ Initialize loss function. """
  return nn.BCELoss()
#nn.BCELoss() default
#nn.MSELoss()
#nn.L1Loss()
#nn.NLLoss()
#custom_weighted

def initialize_optimizers(generator, discriminator):
  """ Initialize optimizers for Generator and Discriminator. """
  generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=OPTIMIZER_LR, betas=OPTIMIZER_BETAS)
  discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=OPTIMIZER_LR, betas=OPTIMIZER_BETAS)
  return generator_optimizer, discriminator_optimizer


# Noise generation
def generate_noise(number_of_images = 1, noise_dimension = NOISE_DIMENSION, device=None):
  """ Generate noise for number_of_images images, with a specific noise_dimension """
  return torch.randn(number_of_images, noise_dimension, device=device)


# Zero gradient
def efficient_zero_grad(model):
  """
    Apply zero_grad more efficiently
    Source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
  """
  for param in model.parameters():
    param.grad = None


# Loss calculation + feedback
def forward_and_backward(model, data, loss_function, targets):
  """
    Perform forward and backward pass in a generic way. Returns loss value.
  """
  outputs = model(data)
  error = loss_function(outputs, targets)
  error.backward(retain_graph=False) #retain_graph ??
  return error.item()




    #!!!!!!!!!!!!Training step, differential training frequency
  # +modify perform_epoch




def perform_train_step(generator, discriminator, real_data, \
  loss_function, generator_optimizer, discriminator_optimizer, device = get_device()):
  """ Perform a single training step. """

  # 1. PREPARATION
  # Set real and fake labels.
  real_label, fake_label = 1.0, 0.0
  # Get images on CPU or GPU as configured and available
  # Also set 'actual batch size', whih can be smaller than BATCH_SIZE
  # in some cases.
  real_images = real_data[0].to(device)
  actual_batch_size = real_images.size(0)
  label = torch.full((actual_batch_size,1), real_label, device=device)

  # 2. TRAINING THE DISCRIMINATOR
  # Zero the gradients for discriminator
  efficient_zero_grad(discriminator)
  # Forward + backward on real images, reshaped
  real_images = real_images.view(real_images.size(0), -1)
  error_real_images = forward_and_backward(discriminator, real_images, \
    loss_function, label)
  # Forward + backward on generated images
  noise = generate_noise(actual_batch_size, device=device)
  generated_images = generator(noise)
  label.fill_(fake_label)
  error_generated_images =forward_and_backward(discriminator, \
    generated_images.detach(), loss_function, label)
  # Optim for discriminator
  discriminator_optimizer.step()

  # 3. TRAINING THE GENERATOR
  # Forward + backward + optim for generator, including zero grad
  efficient_zero_grad(generator)
  label.fill_(real_label)
  error_generator = forward_and_backward(discriminator, generated_images, loss_function, label)
  generator_optimizer.step()

  # 4. COMPUTING RESULTS
  # Compute loss values in floats for discriminator, which is joint loss.
  error_discriminator = error_real_images + error_generated_images
  # Return generator and discriminator loss so that it can be printed.
  return error_generator, error_discriminator


def perform_epoch(dataloader, generator, discriminator, loss_function, \
    generator_optimizer, discriminator_optimizer, epoch):
  """ Perform a single epoch. """
  for batch_no, real_data in enumerate(dataloader, 0):
    # Perform training step
    generator_loss_val, discriminator_loss_val = perform_train_step(generator, \
      discriminator, real_data, loss_function, \
      generator_optimizer, discriminator_optimizer)
    # Print statistics and generate image after every n-th batch
    if batch_no % PRINT_STATS_AFTER_BATCH == 0:
      print_training_progress(batch_no, generator_loss_val, discriminator_loss_val)
      generate_image(generator, epoch, batch_no)
  # Save models on epoch completion.
  save_models(generator, discriminator, epoch)
  # Clear memory after every epoch
  torch.cuda.empty_cache()


def train_dcgan():
  """ Train the DCGAN. """
  # Make directory for unique run
  make_directory_for_run()
  # Set fixed random number seed
  torch.manual_seed(42)
  # Get prepared dataset
  dataloader = prepare_dataset()
  # Initialize models
  generator, discriminator = initialize_models()
  # Initialize loss and optimizers
  loss_function = initialize_loss()
  generator_optimizer, discriminator_optimizer = initialize_optimizers(generator, discriminator)
  # Train the model
  for epoch in range(NUM_EPOCHS):
    print(f'Starting epoch {epoch}...')
    perform_epoch(dataloader, generator, discriminator, loss_function, \
      generator_optimizer, discriminator_optimizer, epoch)
  # Finished :-)
  print(f'Finished unique run {UNIQUE_RUN_ID}')


if __name__ == '__main__':
  train_dcgan()



##adaptive weighted loss
# https://github.com/vasily789/adaptive-weighted-gans
#real_validity = discriminator(real_image)
#fake_validity = discriminator(fake_image)





class aw_method():
    def __init__(self, alpha1=0.5, alpha2=0.75, delta=0.05, epsilon=0.05, normalized_aw=True):
        assert alpha1 < alpha2
        self._alpha1 = alpha1
        self._alpha2 = alpha2
        self._delta = delta
        self._epsilon = epsilon
        self._normalized_aw = normalized_aw

    def aw_loss(self, Dloss_real, Dloss_fake, Dis_opt, Dis_Net, real_validity, fake_validity):
        # resetting gradient back to zero
        Dis_opt.zero_grad()

        # computing real batch gradient
        Dloss_real.backward(retain_graph=True)
        # tensor with real gradients
        grad_real_tensor = [param.grad.clone() for _, param in Dis_Net.named_parameters()]
        grad_real_list = torch.cat([grad.reshape(-1) for grad in grad_real_tensor], dim=0)
        # calculating the norm of the real gradient
        rdotr = torch.dot(grad_real_list, grad_real_list).item() + 1e-4 # 1e-4 added to avoid division by zero
        r_norm = np.sqrt(rdotr)

        # resetting gradient back to zero
        Dis_opt.zero_grad()

        # computing fake batch gradient
        Dloss_fake.backward()#(retain_graph=True)
        # tensor with real gradients
        grad_fake_tensor = [param.grad.clone() for _, param in Dis_Net.named_parameters()]
        grad_fake_list = torch.cat([grad.reshape(-1) for grad in grad_fake_tensor], dim=0)
        # calculating the norm of the fake gradient
        fdotf = torch.dot(grad_fake_list, grad_fake_list).item() + 1e-4 # 1e-4 added to avoid division by zero
        f_norm = np.sqrt(fdotf)

        # resetting gradient back to zero
        Dis_opt.zero_grad()

        # dot product between real and fake gradients
        rdotf = torch.dot(grad_real_list,grad_fake_list).item()
        fdotr = rdotf

        # Real and Fake scores
        rs = torch.mean(torch.sigmoid(real_validity))
        fs = torch.mean(torch.sigmoid(fake_validity))

        if self._normalized_aw:
            # Implementation of normalized version of aw-method, i.e. Algorithm 1
            if rs < self._alpha1 or rs < (fs - self._delta):
                if rdotf <= 0:
                    # Case 1:
                    w_r = (1/r_norm) + self._epsilon
                    w_f = (-fdotr/(fdotf*r_norm)) + self._epsilon
                else:
                    # Case 2:
                    w_r = (1/r_norm) + self._epsilon
                    w_f = self._epsilon
            elif rs > self._alpha2 and rs > (fs - self._delta):
                if rdotf <= 0:
                    # Case 3:
                    w_r = (-rdotf/(rdotr*f_norm)) + self._epsilon
                    w_f = (1/f_norm) + self._epsilon
                else:
                    # Case 4:
                    w_r = self._epsilon
                    w_f = (1/f_norm) + self._epsilon
            else:
                # Case 5
                w_r = (1/r_norm) + self._epsilon
                w_f = (1/f_norm) + self._epsilon
        else:
            # Implementation of non-normalized version of aw-method, i.e. Algorithm 2
            if rs < self._alpha1 or rs < (fs - self._delta):
                if rdotf <= 0:
                    # Case 1:
                    w_r = 1 + self._epsilon
                    w_f = (-fdotr/fdotf) + self._epsilon
                else:
                    # Case 2:
                    w_r = 1 + self._epsilon
                    w_f = self._epsilon
            elif rs > self._alpha2 and rs > (fs - self._delta):
                if rdotf <= 0:
                    # Case 3:
                    w_r = (-rdotf/rdotr) + self._epsilon
                    w_f = 1 + self._epsilon
                else:
                    # Case 4:
                    w_r = self._epsilon
                    w_f = 1 + self._epsilon
            else:
                # Case 5
                w_r = 1 + self._epsilon
                w_f = 1 + self._epsilon

        # calculating aw_loss
        aw_loss = w_r * Dloss_real + w_f * Dloss_fake

        # updating gradient, i.e. getting aw_loss gradient
        for index, (_, param) in enumerate(Dis_Net.named_parameters()):
            print(grad_real_tensor[index])
            print(grad_fake_tensor[index])
            param.grad = w_r * grad_real_tensor[index] + w_f * grad_fake_tensor[index]

        return aw_loss