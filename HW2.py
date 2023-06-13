import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.models as models


device = torch.device("cpu")
model_X = models.resnet18(pretrained=True)

# Define the transform to be applied to the input image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the testing image T and the database of images
T1 = Image.open('testing1.png')

T2 = Image.open('testing2.png')

database = ['01.png', '15.png', '83.png', 'd3.png', 'd5.png', 'd6.png', 'd31.png', 'x10.png']

# Preprocess the query image T
T1 = transform(T1).unsqueeze(0)
T2 = transform(T2).unsqueeze(0)
K = 5


def process(t_image):
    # Extract the CNN feature vector for the query image T
    with torch.no_grad():
        f = model_X(t_image)

    # Compute the L2 distance between the feature vector of T and each image in the database
    R = []
    for i in range(len(database)):
        # Load the i-th image from the database and preprocess it
        Ii = Image.open("./database/" + database[i])
        Ii = transform(Ii).unsqueeze(0)

        # Extract the CNN feature vector for the i-th image in the database
        with torch.no_grad():
            fi = model_X(Ii)

        # Compute the L2 distance between f and fi
        distance = torch.norm(f - fi, p=2).item()

        # Save the distance to the list R
        R.append(distance)

    # Sort the distances and retrieve the top-K similar images
    sorted_indices = np.argsort(R)[:K]
    top_K_images = [database[i] for i in sorted_indices]
    return top_K_images, R


top_k_images_T1, R_T1 = process(T1)

top_k_images_T2, R_T2 = process(T2)


def plot(R_vec, top_k, image_name):
    # Display the query image T at the top of the plot

    fig, axs = plt.subplots(1, K, figsize=(30, 15))

    fig.suptitle('Retrieved top-5 results using ' + image_name + ' as T for image retrieval:')
    for i in range(K):
        # Load the i-th top-K image and display it
        sorted_indices = np.argsort(R_vec)[:K]
        # top_K_images = [database[i] for i in sorted_indices]
        image_path = top_k[i]
        image = Image.open("./database/" + image_path)
        axs[i].imshow(image)

        # Display the distance between the query image T and the i-th top-K image

        distance = R_vec[sorted_indices[i]]
        axs[i].set_title(image_path + '  Distance: {:.2f}'.format(distance))

        # Remove the x and y-axis labels
        axs[i].axis('off')
    plt.show()


image_name_input = input('Enter the testing image name: \n')
if image_name_input.__contains__("1"):
    plot(R_T1, top_k_images_T1, image_name_input)
else:
    plot(R_T2, top_k_images_T2, image_name_input)

