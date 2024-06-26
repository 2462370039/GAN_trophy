import argparse
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import time


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoches", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--type", type=str, default='GAN', help="The type of GAN")
    parser.add_argument("--type", type=str, default='DCGAN', help="The type of DCGAN")
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator_CNN, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))  # 100 ——> 128 * 8 * 8 = 8192

        self.conv_blocks = nn.Sequential(
            # 这是一个批标准化层，用于对输入的特征图进行标准化处理，有助于加速网络训练并提高泛化性能。128 是输入特征图的通道数。
            nn.BatchNorm2d(128),
            # 这是一个上采样层，用于将特征图的尺寸沿着两个维度放大两倍。这个操作会增加特征图的尺寸，从而在生成器中实现特征图的上采样过程。
            nn.Upsample(scale_factor=2),
            # 这是一个二维卷积层，用于对特征图进行卷积操作。128 是输入特征图的通道数，128 是输出特征图的通道数，3 是卷积核的大小，1 是卷积核的步长，1 是卷积核的填充。
            nn.Conv2d(128, 128, 3, stride=1, padding=1),

            # 这是另一个批标准化层，用于对输入的特征图进行标准化处理，动量参数为 0.8。
            nn.BatchNorm2d(128, 0.8),
            # 这是一个 LeakyReLU 激活层，用于对特征图进行非线性变换。0.2 是负半轴的斜率。inplace=True 表示激活函数将会直接对输入数据进行修改。
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),

            # 这是第三个批标准化层，用于对输入的特征图进行标准化处理，动量参数为 0.8。
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),

            # 这是一个 Tanh 激活层，用于对特征图进行非线性变换。Tanh 函数将特征图的像素值标准化到 -1 到 1 之间。
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_CNN(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator_CNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                # 对特征图下采样，降低分辨率。
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # 在卷积层之后添加一个 2D Dropout 层，用于对特征图进行随机失活操作，以防止过拟合。丢弃比例为 25%。
                nn.Dropout2d(0.25)
            ]
            if bn:
                # 在卷积层之后添加一个批标准化层，用于对特征图进行标准化处理。
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())  # 128 * 2 * 2 ——> 1

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def train():
    # 创建 out 文件夹存放训练结果
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists("out"):
        print("移除现有 out 文件夹！")
        os.system("rmdir /s /q out")
    time.sleep(1)
    print("创建 out 文件夹！")
    os.mkdir("./out")
    os.mkdir("./out/loss")
    os.mkdir("./out/module")

    opt = args_parse()

    # Configure data loader
    transform = transforms.Compose(
        [
            # 统一图像大小
            transforms.Resize((32, 32)),
            # 转换成张量
            transforms.ToTensor(),
            # 归一化，将输入数据的像素值标准化到-1到1之间（具体操作： 每个通道均值减0.5 和 每个通道像素值除以0.5）
            transforms.Normalize([0.5], [0.5])
        ])

    data = datasets.ImageFolder('./data', transform=transform)

    # # 下载MNIST数据集
    # mnist_data = datasets.MNIST(
    #     "mnist-data",
    #     train=True,
    #     download=False,
    #     transform=transform
    # )

    # 加载训练数据
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=True)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Construct generator and discriminator
    if opt.type == 'DCGAN':
        generator = Generator_CNN(opt.latent_dim, img_shape)
        discriminator = Discriminator_CNN(img_shape)
    else:
        generator = Generator(opt.latent_dim, img_shape)
        discriminator = Discriminator(img_shape)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    cuda = True if torch.cuda.is_available() else False
    print(cuda)
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=(opt.lr * 6 / 9), betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    print(generator)
    print(discriminator)

    # 保存网络结构
    with open('out\\module\\network.json', 'w') as f:
        f.write(cuda.__str__())
        f.write('\n\n')
        f.write(generator.__str__())
        f.write('\n\n')
        f.write(discriminator.__str__())


    d_loss_all = []
    g_loss_all = []

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(opt.n_epoches):
        for i, (imgs, _) in enumerate(train_loader):
            # adversarial ground truths
            valid = torch.ones(imgs.shape[0], 1).type(Tensor)
            fake = torch.zeros(imgs.shape[0], 1).type(Tensor)

            # 使用软标签，0.9~1.0之间的随机数
            valid -= 0.1 * torch.rand(valid.shape).type(Tensor)
            fake += 0.1 * torch.rand(fake.shape).type(Tensor)

            real_imgs = imgs.type(Tensor)

            #############    Train Generator    ################
            for j in range(6):
                optimizer_G.zero_grad()

                # sample noise as generator input
                z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).type(Tensor)

                # Generate a batch of images
                gen_imgs = generator(z)

                # G-Loss
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

            #############  Train Discriminator ################
            optimizer_D.zero_grad()

            # D-Loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_loss_all.append(d_loss.item())
            g_loss_all.append(g_loss.item())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G Loss: %f]"
                % (epoch, opt.n_epoches, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(train_loader) + i
            os.makedirs("out\\images", exist_ok=True)
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:128], "out/images/%d.png" % (batches_done), nrow=16, normalize=True)


    ###### 保存训练结果 ######
    # 保存模型
    torch.save(generator.state_dict(), 'out\\module\\generator.pth')
    torch.save(discriminator.state_dict(), 'out\\module\\discriminator.pth')

    # 保存loss
    np.save('out\\loss\\d_loss.npy', np.array(d_loss_all))
    np.save('out\\loss\\g_loss.npy', np.array(g_loss_all))

    # 画出loss曲线
    plt.figure()
    plt.plot(d_loss_all, label='D loss')
    plt.plot(g_loss_all, label='G loss')
    plt.legend()
    plt.savefig('out\\loss\\loss.png')
    plt.show()



if __name__ == '__main__':
    train()

    # 保存训练超参数到json文件
    with open('out\\module\\args.json', 'w') as f:
        for k, v in vars(args_parse()).items():
            f.write(f'{k:<20}: {v}\n')