import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", default=120)
    parser.add_argument('--epochs', dest="epochs", type=int, default=10)
    parser.add_argument('--gpu', dest="gpu", default="gpu")
    args = parser.parse_args()
    return args

def check_gpu(args.gpu):
    if not gpu_arg:
        return torch.device("cpu")
    else
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, trainloader, validationloader):
    epochs = args.epochs
    running_loss = 0
    print_every = 5
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        steps = 0
        
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        validation_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Batch: {steps / 5}.. "
                    f"Train loss: {running_loss/print_every:.4f}.. "
                    f"Validation loss: {validation_loss/len(validationloader):.4f}.. "
                  f"Validation accuracy: {accuracy:.4f}")
                running_loss = 0
                model.train()
    return model
    
def validate_model(model, validationloader):
    accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in validationloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            validation_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f"Validation accuracy: {accuracy:.4f}")
    
def save_model(model, save_dir):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'classifier' : model.classifier,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
def main():
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
 

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_transforms),
        'valid': datasets.ImageFolder(valid_dir, transform=test_transforms),
        'test': datasets.ImageFolder(test_dir, transform=test_transforms)
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
    validationloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    num_categories = len(cat_to_name)
    
    model = models.vgg16(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    

    model.classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, args.hidden_units)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(0.5)),
                ('hidden_layer', nn.Linear(args.hidden_units, num_categories)),
                ('output', nn.LogSoftmax(dim=1))]))
   
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    device = check_gpu();
    
    trained_model = train_model(model, trainloader, validationloader):
    
    print("\nModel is trained!")
    
    validate_model(trained_model, validation_loader)
   
    save_model(trained_model, args.save_dir)

if __name__ == '__main__': main()
    
        


