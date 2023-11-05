import torch
import config
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True


def test():
    # dataset = MyImageFolder(root_dir="input/")
    # loader = DataLoader(
    #     dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=config.NUM_WORKERS,
    # )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    # mse = nn.MSELoss()
    # bce = nn.BCEWithLogitsLoss()
    # vgg_loss = VGGLoss()

    
    load_checkpoint(
        config.CHECKPOINT_GEN,
        gen,
        opt_gen,
        config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    )


    plot_examples(low_res_folder=config.TEST_DIR, gen=gen)







if __name__ == '__main__':
    test()