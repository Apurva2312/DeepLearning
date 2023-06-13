import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os


# Define the encoder1 network
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        x = self.pool(x)
        return x


# Define the encoder2 network
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        x = self.pool(x)
        return x


# Define the decoder network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 50, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(50)
        self.deconv2 = nn.ConvTranspose2d(50, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)


    def forward(self, x):
        x = F.relu(self.deconv1(x))
        # x = self.bn1(x)
        x = F.relu(self.deconv2(x))
        # x = self.bn2(x)
        x = self.deconv3(x)
        return x


# Define the denoising network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder1 = Encoder1()
        self.encoder2 = Encoder2()
        self.decoder = Decoder()

    def forward(self, x1, x2):
        x1_enc1 = self.encoder1(x1)
        x2_enc2 = self.encoder2(x2)
        x1_resized = F.interpolate(x1_enc1, size=(32, 32), mode='nearest')
        x_concat = torch.cat((x1_resized, x2_enc2), dim=1)
        x = self.decoder(x_concat)
        return x


class NoisyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.image_files[index])
        img = Image.open(img_path).convert('RGB')

        # Add random noise to image
        noise_img = self.add_noise(img)

        # Resize images to required sizes
        img1 = self.transform(noise_img.resize((32, 32)))
        img2 = self.transform(noise_img.resize((128, 128)))

        return img1, img2

    def __len__(self):
        return len(self.image_files)

    def add_noise(self, img):

        image_tensor = TF.to_tensor(img)

        # set the standard deviation of the noise
        sigma = 0.01

        # generate a tensor of noise with the same shape as the image
        noise_tensor = torch.randn_like(image_tensor) * sigma

        # add the noise to the image
        noisy_image_tensor = image_tensor + noise_tensor

        # convert the noisy image tensor back to a PIL image
        noise_img = TF.to_pil_image(noisy_image_tensor)
        # noise_img.show()

        return noise_img


# Instantiate the network and move it to the GPU if available
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Load the training data
#Define transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
train_dataset = NoisyImageDataset(root='./training_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)


# Train model
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (img1, img2) in enumerate(train_loader):
        img1, img2 = img1.to(device), img2.to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2)
        outputs_resized = F.interpolate(outputs, size=(128, 128), mode='nearest')
        loss = criterion(outputs_resized, img2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


# Train and test the model
for epoch in range(10):
    train_loss = train(net, optimizer, criterion, train_loader, device)
    print('Epoch: {} Train Loss: {:.6f} '.format(epoch + 1, train_loss))

torch.save(net.state_dict(), 'denoising_model.pth')
