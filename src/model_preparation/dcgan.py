from torch import nn


class DCGan:
    def __init__(self, latent_size, channels, feature_maps, name):
        self.generator = Generator(latent_size, channels, feature_maps)
        self.discriminator = Discriminator(channels, feature_maps)


    def train_generator(self):
        pass

    def train_discriminator(self):
        pass


        


class Generator(nn.Module):
    def __init__(self, latent_size, channels, feature_maps):
        super(Generator, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent_size, feature_maps * 8, 4, 1, 0, bias=False)
        self.batch1 = nn.BatchNorm2d(feature_maps * 8)

        self.conv2 = nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False)
        self.batch2 = nn.BatchNorm2d(feature_maps * 4)

        self.conv3 = nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(feature_maps * 2)

        self.conv4 = nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False)
        self.batch4 = nn.BatchNorm2d(feature_maps)

        self.conv5 = nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_maps) x 32 x 32``
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_maps*2) x 16 x 16``
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_maps*4) x 8 x 8``
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(feature_maps*8) x 4 x 4``
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
