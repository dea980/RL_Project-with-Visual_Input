import torch as T
import torch.nn.functional as F
import numpy as np
from conv_vae import ConvVAE
from linear_vae import LinearVAE
from dataset import CustomDataset
from matplotlib import pyplot as plt
from scipy.io import loadmat

def loss_function(true_batch, recon_batch, mean, logvar, beta):
    """
    Minimize the reconstruction loss + KL divergence
    """
    recon_loss = F.mse_loss(recon_batch, true_batch)
    kl_div = -0.5 * T.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def train(model, optimizer, epoch, dataloader, beta):
    """
    Performs one iteration of training using examples from dataloader.
    Assumption: Data is of shape (n_examples, height, width, n_channels)
    """
    model.train()
    train_loss = []
    for _, x in enumerate(dataloader):
        optimizer.zero_grad()
        # After permutation, shape of data (n_examples, n_channels, height, width)
        # x_ = x.permute(0, 3, 1, 2)
        x_ = x.reshape(x.shape[0], -1)
        recon_batch, mean, variance = model.forward(x_)
        loss = loss_function(x_, recon_batch, mean, variance, beta)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print('====> Epoch: {} Training loss: {:.4f}'.format(epoch, np.mean(train_loss)))
    return model
    
def generate_images(model, z_random):
    model.eval()
    res = model.decode(z_random.squeeze())
    # return res.permute(0,2,3,1).detach().cpu().numpy()
    return res.reshape(10, 64, 64).detach().cpu().numpy()

def save_imgs(imgs, base_path):
    for idx, img in enumerate(imgs):
        fig = plt.figure(figsize=(25,25))
        plt.imshow(img)
        plt.title(f"I:{idx}")
        plt.savefig(f"{base_path}/{idx}.jpg")
    
if __name__ == "__main__":    
    """
    n: number of examples to generate
    latent_dim: dimensions of the bottleneck layer
    n_epochs: number of epochs to train the model
    frames: numpy array of frames. shape: (n_examples, height, width) for LinearVAE, (n_examples, height, width, channel) for ConvVAE
    img_dim: (64, 64)
    """
    n = 10
    latent_dim = 32
    n_epochs = 30
    frames = loadmat("dataset/cartpole.mat")["frames"][:, :, :, 0]

    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    img_dim = (64, 64)
    # model = ConvVAE(img_dim, latent_dim, device)
    model = LinearVAE(img_dim, latent_dim, device)

    optimizer = T.optim.Adam(model.parameters())
    
    dataset = CustomDataset(frames, device)
    dataloader = T.utils.data.DataLoader(dataset, batch_size = 10)
    
    for i in range(1, n_epochs+1):
        model = train(model, optimizer, i, dataloader, beta=1)
    
    normal_distribution = T.distributions.normal.Normal(T.Tensor([0.0]), T.Tensor([1.0]))
    z_random = normal_distribution.sample((n, latent_dim)).to(device)
    recon_images = generate_images(model, z_random)
    save_imgs(recon_images, "dataset/cartpole/recon")
    