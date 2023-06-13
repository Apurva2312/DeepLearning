import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from training import Net, NoisyImageDataset
import os


def test_model(model, test_loader, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch_idx, (img1, img2) in enumerate(test_loader):
            img1, img2 = img1.to(device), img2.to(device)
            output = model(img1, img2)

            # Convert the denoised image tensor back into a numpy array
            denoised_image_array = output.squeeze().permute(1, 2, 0).cpu().numpy()

            # Display the image using imshow
            plt.imshow(denoised_image_array)
            plt.show()

            # print progress
            print(f'Processed image {batch_idx+1}/{len(test_loader)}')

    print('Testing complete!')


transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
test_dataset = NoisyImageDataset(root='./testing_data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_path = "./denoising_model.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model1 = Net()
# load the saved state dictionary into the model
model1.load_state_dict(state_dict)
test_model(model1, test_loader, "./")
