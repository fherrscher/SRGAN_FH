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


if config.USE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(config.TB_LOG_DIR)



def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, epoch):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        #l2_loss = mse(fake, high_res)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss


        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    if config.USE_TENSORBOARD:
        writer.add_scalar('Adv_Loss', adversarial_loss.item(), global_step=epoch)
        writer.add_scalar('VGG_Loss', loss_for_vgg.item(), global_step=epoch)
        writer.add_scalar('Disc_Loss_Real', disc_loss_real.item(), global_step=epoch)
        writer.add_scalar('Disc_Loss_Fake', disc_loss_fake.item(), global_step=epoch)
        writer.add_scalar('Disc_Loss', loss_disc.item(), global_step=epoch)
        writer.add_scalar('Gen_Loss', gen_loss.item(), global_step=epoch)



        #writer.flush()
        # print("MSE: ", gen_loss_item)
        # print(epoch)



def main():
    dataset = MyImageFolder(root_dir=config.INPUT_DIR)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    for epoch in range(config.NUM_EPOCHS):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, epoch)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


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
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, test')

    args = parser.parse_args()

    if args.mode == 'train':
        main()
    elif args.mode == 'test':
        test()
    else:
        raise Exception("Unknow --mode")
