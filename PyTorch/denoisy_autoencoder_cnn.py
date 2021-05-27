import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5
noise_factor = 0.5

class DenoisyAutoEncoder(nn.Module):
    def __init__(self):
        super(DenoisyAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1)
        )

        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded

denoisyautoencoder = DenoisyAutoEncoder()

optimizer = torch.optim.Adam(denoisyautoencoder.parameters(), lr=LR)
# loss_func = nn.BCELoss()
loss_func = nn.MSELoss()


train_noisy = (train_data.train_data.numpy() / 255.) + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_data.train_data.shape)
train_noisy = np.clip(train_noisy, 0.0, 1.0)

train_and_noisy_merged = Data.TensorDataset(torch.from_numpy(train_noisy).type(torch.FloatTensor), train_data.train_data / 255.)

train_merged_loader = Data.DataLoader(train_and_noisy_merged, batch_size=BATCH_SIZE, shuffle=True)


f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 2))
plt.ion()

for i in range(N_TEST_IMG):
    a[0][i].imshow(train_noisy[i].reshape(28, 28), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

    a[2][i].imshow(train_data.test_data.numpy()[i].reshape(28, 28), cmap='gray')
    a[2][i].set_xticks(())
    a[2][i].set_yticks(())



for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_merged_loader):
        b_x = x.view(-1, 1, 28, 28)
        b_y = y.view(-1, 1, 28, 28)

        encoded, decoded = denoisyautoencoder(b_x)

        loss = loss_func(decoded, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Epoch: {epoch}, | train loss: {loss.data.numpy():.4f}')

            _, decoded_data = denoisyautoencoder(torch.from_numpy(train_noisy[:N_TEST_IMG].reshape([-1, 1, 28, 28])).type(torch.FloatTensor))

            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(decoded_data.data.numpy()[i].reshape(28, 28), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)


plt.ioff()
plt.show()

