import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder # for operators training
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split # to train using more than one dataset

from model import CNN


class RemapLabelsDataset(Dataset):
    # Here we shift labels so instead of having (+ : 0), (- : 1) ...
    # Now we'll have (+ : 10), (- : 11)
    def __init__(self, dataset: ImageFolder, class_name_to_global_label: dict):
        self.dataset = dataset
        self.class_name_to_global_label = class_name_to_global_label
        self.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, local_label = self.dataset[idx]
        class_name = self.idx_to_class[local_label]  # e.g. "add"
        global_label = self.class_name_to_global_label[class_name]  # e.g. 10
        return img, global_label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # MNIST transfrom (already 28*28 and white on black)
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # [0, 1] -> [-1, 1]
    ])

    op_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 -x), # to invert : ink becomes white, bg becomes black
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    # Load operators from folders
    ops_folder = ImageFolder(root='./operators', transform=op_transform)

    # Map operator folder names
    op_map = {
        "add": 10,
        "sub": 11,
        "mul": 12,
        "div": 13,
    }

    ops_dataset = RemapLabelsDataset(ops_folder, op_map) # We use our class to shif labels
    
    # we need to split ops dataset (train/test) we use 80% for train and 20% for test
    ops_train_size = int(0.8 * len(ops_dataset))
    ops_test_size = len(ops_dataset) - ops_train_size
    ops_train, ops_test = random_split(ops_dataset, [ops_train_size, ops_test_size])


    # Combine datasets
    trainset = ConcatDataset([mnist_train, ops_train])
    testset = ConcatDataset([mnist_test, ops_test])


    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader  = DataLoader(testset, batch_size=64, shuffle=False)


    # Model
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images) # predicted labels
            loss = loss_fn(outputs, labels) # compare predicted to real labels
            loss.backward() # Backpropagation
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1) # gets the class with the highest probability (basically the prediction)
            # it returns : values, indexes. We don't care about values but mostly about index (prediction)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print(f"Accuracy: {100 * correct/total:.2f}%")

    # Save weights
    torch.save(model.state_dict(), "digits_ops_CNN.pth")
    print("Saved model to digits_ops_CNN.pth")

if __name__ == "__main__":
    main()
