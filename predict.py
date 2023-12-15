import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from train import check_gpu
from torchvision import models

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file.', required=True)
    parser.add_argument('--top_k', type=int, default=5, help='Number of Top K matches.')
    parser.add_argument('--labels', type=str, help='JSON file containing label names')
    parser.add_argument('--gpu', default="gpu", action="store",  help='Use GPU if available')

    args = parser.parse_args()
    
    return args

def load_checkpoint():
    checkpoint = torch.load('checkpoint.pth')
    
    model = models.vgg16(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def process_image(image):
    # Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)
    
    # Resize the images where the shortest side is 256 pixels, 
    # keeping the aspect ratio
    aspect_ratio = img.size[0] / img.size[1]
    
    if aspect_ratio > 1:
        img.resize((round(aspect_ratio * 256), 256))
    else:
        img.resize((256, round(aspect_ratio * 256)))
    
    # Crop out the center 224x224 portion of the image
    left = (img.size[0] - 224)/2
    top = (img.size[1] - 224)/2
    right = (img.size[0] + 224)/2
    bottom = (img.size[1] + 224)/2
    img = img.crop((round(left), round(top), round(right), round(bottom)))
    
    # Convert Color Channels to float 0-1 from integers 0-255
    np_img = np.array(img) / 255
    
    # Normalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_img = (np_img - mean) / std
    
    # Reorder dimensions
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img

def predict(image_path, model, topk=5):
    # Predict the class (or classes) of an image using a trained deep learning model.
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    
    # Convert image from numpy to torch
    img = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor)
    
    with torch.no_grad():
        ps = torch.exp(model.forward(img))
    
    # Find the top 5 results
    top_probs, top_labels = ps.topk(topk)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_labels]
    
    return top_probs, top_classes

def main():
    
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    
    processed_image = process_image(args.image)
    
    device = check_gpu(gpu_arg=args.gpu);
    
    probs, labels = predict(processed_image, model, args.topk)
    
    # Get flower names from using cat_to_name json.
    flowers = [cat_to_name[item] for item in labels]
    
    # Print prediction results as dictionary.
    # Reference: https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
    result_dict = dict(zip(flowers, probs * 100))
    print(result_dict)
    
if __name__ == '__main__': main()

