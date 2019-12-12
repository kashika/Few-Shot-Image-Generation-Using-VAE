import os
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import pickle as pkl
import numpy as np
from docopt import docopt

import torch
import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter
import numpy as np
import os


import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weight_initialization(net):
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()
        elif isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.02)
            module.bias.data.zero_()


class Convae(nn.Module):
    def __init__(self, image_channels=1, input_size=28, nz=40):
        super(Convae, self).__init__()

        if torch.cuda.is_available():
            self.FloatTensor = torch.cuda.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

        self.is_cuda_available = True
        self.nz = nz
        self.size_of_input = input_size
        self.channels = image_channels
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.encode_fc = nn.Sequential(
            nn.Linear(128 * (input_size // 4) * (input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )

        self.qzmean = nn.Linear(1024, self.nz)
        self.qzvar = nn.Linear(1024, self.nz)

        self.decode = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * (input_size // 4) * (input_size // 4)),
            nn.BatchNorm1d(128 * (input_size // 4) * (input_size // 4)),
            nn.ReLU(True),
        )
        self.decode_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1),
            nn.Sigmoid()
        )
        weight_initialization(self)

    def encoder(self, x):
        conv = self.conv(x)
        h = self.encode_fc(conv.view(-1, 128 * (self.size_of_input // 4) * (self.size_of_input // 4)))
        # h = self.encode_fc(x.view(-1, 64*(self.input_size//4) * (self.input_size // 4)))
        return self.qzmean(h), self.qzvar(h)

    def decoder(self, z):
        deconv_input = self.decode(z)
        deconv_input = deconv_input.view(-1, 128, self.size_of_input // 4, self.size_of_input // 4)
        return self.decode_conv(deconv_input)

    def parameter_recalc(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward_prop(self, x):
        mu, logvar = self.encoder(x)
        z = self.parameter_recalc(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar

class Mnist_MetaENV:
    def __init__(self, height=28, length=28):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.MNIST(root='./data', train=True, download=True)
        self.task_maker()
        self.split_dataset()
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.length))
        self.training_task_sampling()

    def training_task_sampling(self, batch_size=64):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float, device=device)
        return batch, task

    def validation_task_sampling(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.to_tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float, device=device)
        return batch, task

    def task_maker(self):
        self.task_to_examples = {}
        self.all_tasks = set(self.data.targets.numpy())
        for i, digit in enumerate(self.data.targets.numpy()):
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_dataset(self):
        self.validation_task = {9}
        self.training_task = self.all_tasks - self.validation_task


class Omniglot_metaENV:
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.data = datasets.Omniglot(root='./data', download=True)
        self.task_maker()
        self.split_dataset()
        self.resize = transforms.Resize((self.height, self.length))
        self.tensor = transforms.ToTensor()

    def training_task_sampling(self, batch_size=4):
        task = str(random.sample(self.training_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.tensor(self.resize(self.data[idx][0])).numpy() for idx in task_idx]),
                             dtype=torch.float, device=device)
        return batch, task

    def validation_task_sampling(self, batch_size=64):
        task = str(random.sample(self.validation_task, 1)[0])
        task_idx = random.sample(self.task_to_examples[task], batch_size)

        batch = torch.tensor(np.array([self.tensor(self.resize(self.data[idx][0])).numpy() for idx in task_dix]),
                             dtype=torch.float, device=device)
        return batch, task

    def task_maker(self):
        self.task_to_examples = {}
        self.all_tasks = set()
        for i, (_, digit) in enumerate(self.data):
            self.all_tasks.update([digit])
            if str(digit) not in self.task_to_examples:
                self.task_to_examples[str(digit)] = []
            self.task_to_examples[str(digit)].append(i)

    def split_dataset(self):
        self.validation_task = set(random.sample(self.all_tasks, 20))
        self.training_task = self.all_tasks - self.validation_task


class FIGR8_MetaENV(Dataset):
    def __init__(self, height=32, length=32):
        self.channels = 1
        self.height = height
        self.length = length
        self.resize = transforms.Resize((height, length))
        self.tensor = transforms.ToTensor()

        self.tasks = self.get_tasks()
        self.tasks = set(self.tasks)
        self.split_dataset()

    def get_tasks(self):
        tasks = dict()
        path = './data/'
        for task in os.listdir(path):
            tasks[task] = []
            task_path = os.path.join(path, task)
            for imgs in os.listdir(task_path):
                img = Image.open(os.path.join(task_path, imgs))
                tasks[task].append(np.array(self.tensor(self.resize(img))))
            tasks[task] = np.array(tasks[task])
        return tasks

    def split_dataset(self):
        self.validation_task = set(random.sample(self.all_tasks, 50))
        self.training_task = self.all_tasks - self.validation_task
        pkl.dump(self.validation_task, open('validation_task.pkl', 'wb'))
        pkl.dump(self.training_task, open('training_task.pkl', 'wb'))

    def sample_training_task(self, batch_size=4):
        task = random.sample(self.training_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def sample_validation_task(self, batch_size=4):
        task = random.sample(self.validation_task, 1)[0]
        task_idx = random.sample([i for i in range(self.tasks[task].shape[0])], batch_size)
        batch = self.tasks[task][task_idx]
        batch = torch.tensor(batch, dtype=torch.float, device=device)
        return batch, task

    def __len__(self):
        return len(self.files)

class FewShotImageGeneration:
    def __init__(self, args):
        self.load_arguments(args)
        self.id = self.get_id()
        self.shape = 100
        self.summary_writter = SummaryWriter('logs/' + self.id)
        self.environment = eval(self.dset + 'MetaEnv(height=self.height, length=self.length)')
        self.vae_initialization()
        self.checkpoint_loader()
        self.out_lr = ""
        self.in_lr = ""
        self.b_size = ""
        self.epochs = ""
        self.height = ""
        self.length = ""
        self.dset = ""
        self.network_option = ""
        self.x_dime = ""

    
    def load_arguments(self, args):
        self.out_lr = float(args['--outer_learning_rate'])
        self.in_lr = float(args['--inner_learning_rate'])
        self.b_size = int(args['--batch_size'])
        self.epochs = int(args['--inner_epochs'])
        self.height = int(args['--height'])
        self.length = int(args['--length'])
        self.dset = args['--dataset']
        self.network_option = args['--network']
        self.x_dime = int(args['--height']) * int(args['--length'])
    
    def get_id(self):
        return '{}_{}_olr{}_ilr{}_bsize{}_ie{}_h{}_l{}'.format(
            self.network_option,
            self.dset,
            str(self.out_lr),
            str(self.in_lr),
            str(self.b_size),
            str(self.epochs),
            str(self.height),
            str(self.length)
        )
    
    def checkpoint_loader(self):
        if os.path.isfile('logs/' + self.id + '/checkpointFolder'):
            checkpoint = torch.load('logs/' + self.id + '/checkpointFolder')
            self.netAE.load_state_dict(checkpoint['convae'])
            self.eps = checkpoint['episode']
        else:
            self.eps = 0
    
    def vae_initialization(self):
        self.netAE = eval(self.network_option + '(self.environment.channels, self.environment.height, self.shape)')
        self.meta_netAE = eval(self.network_option + '(self.environment.channels, self.environment.height, self.shape)').to(device)
        self.netAE_optim = optim.Adam(params=self.netAE.parameters(), lr=self.out_lr)
        self.meta_netAE_optim = optim.Adam(params=self.meta_netAE.parameters(), lr=self.in_lr)
    
    def reset_model(self):
        self.meta_netAE.train()
        self.meta_netAE.load_state_dict(self.netAE.state_dict())

    def get_loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, self.x_dime), x.view(-1, self.x_dime), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def inner_loop(self, real_batch):
        self.meta_netAE.train()
        self.meta_netAE_optim.zero_grad()
        
        rec_batch, mye, logarithmVar = self.meta_netAE(real_batch)
        
        reconstruction_loss = self.get_loss_function(rec_batch, real_batch, mye, logarithmVar)
        
        reconstruction_loss.backward()
        self.meta_netAE_optim.step()

        return reconstruction_loss.item()

    
    def mt_learning_loop(self):
        data, task = self.environment.training_task_sampling(self.b_size)
        real_batch = data.to(device)

        convae_total_loss = 0

        for _ in range(self.epochs):
            recon_loss = self.inner_loop(real_batch)
            convae_total_loss += recon_loss
        
        self.summary_writter.add_scalar('Training_convolutional_ae_loss', convae_total_loss, self.eps)

        for p, meta_p in zip(self.netAE.parameters(), self.meta_netAE.parameters()):
             diff = p - meta_p.cpu()
             p.grad = diff
        self.netAE_optim.step()
    
    def validat_run(self):
        data, task = self.environment.validation_task_sampling(self.b_size)
        training_set = data.cpu().numpy()
        training_set = np.concatenate([training_set[i] for i in range(self.b_size)], axis=-1)
        data_batch = data.to(device)

        c_loss = 0

        for _ in range(self.epochs):
            recon_loss = self.inner_loop(data_batch)
            c_loss += recon_loss
        
        self.meta_netAE.eval()
        with torch.no_grad():
            image = self.meta_netAE.decoder(torch.tensor(np.random.normal(size=(self.b_size * 3, self.shape)), dtype=torch.float, device=device))
            image = image.cpu()
            image = np.concatenate([np.concatenate([image[i * 3 + j] for j in range(3)], axis=-2) for i in range(self.b_size)], axis=-1)

            image = np.concatenate([training_set, image], axis=-2)
            self.summary_writter.add_image('Validation_generated', image, self.eps)
            self.summary_writter.add_scalar('Validation_convae_loss', c_loss, self.eps)
            
            plt.imshow(image[0], cmap='Greys_r')
            plt.show()
            print("Validation_convae_loss:", c_loss)
            

    def training(self):
        while self.eps <= 1000000:
            self.reset_model()
            self.mt_learning_loop()

            # Validation run every 10000 training loop
            if self.eps % 1000 == 0:
                self.validat_run()
                self.checkpoint_model()
            self.eps += 1
    
    def checkpoint_model(self):
        checkpoint = {'convae': self.netAE.state_dict(),
                      'episode': self.eps}
        torch.save(checkpoint, 'logs/' + self.id + '/checkpointFolder')

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    print(device)
    environment = FewShotImageGeneration(args)
    environment.training()
