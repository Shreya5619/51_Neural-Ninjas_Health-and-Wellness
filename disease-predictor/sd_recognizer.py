import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from PIL import Image
from timeit import default_timer as timer
import neuronix

neuronix.set_seeds(50, 50)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the image and label from the original dataset
        image, label = self.dataset[idx]

        # Unsqueeze the label
        label = torch.tensor(label, dtype=torch.long)  # Change dtype to long
        return image, label

class CNN_Model(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
            )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*32*32, 
                      out_features=output_shape)
        )

    def forward(self, x:torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

'''
dataset = ImageFolder(root=r'Datasets/skin disease recognizer/train', transform=transform)
class_names = dataset.classes
dataset= CustomDataset(dataset=dataset)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

model_0 = CNN_Model(input_shape=3, hidden_units=60, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()  # Change to CrossEntropyLoss
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

start_time = timer()

train_test_results = neuronix.train_and_eval(model=model_0,
                        train_loader=train_dataloader,
                        test_loader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        accuracy_fn=neuronix.accuracy_fn,
                        epochs=25,
                        device=device
)

stop_time = timer()

neuronix.print_train_time(start=start_time, end=stop_time, device=device)

neuronix.plot_loss_curves(train_test_results)
neuronix.plot_confusion_matrix(train_test_results['test_labels'], train_test_results['test_predictions'], class_names=class_names)

# Directory containing test images
test_image_dir = r'Datasets/skin disease recognizer/test/Actinic keratosis'

# Load image file paths
image_files = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

# Limit the number of images to display
num_images = 9
image_files = image_files[:num_images]

# Initialize lists to store images and predictions
images = []
pred_classes = []
true_classes = []

# Iterate through image files
for img_path in image_files:
    # Load the image using OpenCV
    image = cv.imread(img_path, cv.IMREAD_COLOR)
    # Resize to match model input size
    image = cv.resize(image, (128, 128))

    # Convert the image to a PIL Image and preprocess it
    image_pil = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    data = transform(image_pil).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Predict using the model
    with torch.inference_mode():
        pred_logit = model_0(data)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_class = pred_prob.argmax(dim=1).item()

    # Save for visualization
    images.append(image)  # Save the original OpenCV image
    pred_classes.append(class_names[pred_class])  # Save the predicted class name
    true_classes.append(os.path.basename(os.path.dirname(img_path)))  # Save the true class from directory structure

# Plot results
plt.figure(figsize=(12, 12))
nrows = int(np.sqrt(len(images))) + 1
ncols = nrows

for idx, img in enumerate(images):
    plt.subplot(nrows, ncols, idx + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    pred_label = pred_classes[idx]
    truth_label = true_classes[idx]
    plt.title(f"Pred: {pred_label}\nTruth: {truth_label}",
              color="green" if pred_label == truth_label else "red",
              fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()

torch.save(model_0.state_dict(), "Skin Disease Recognizer.pth")
'''