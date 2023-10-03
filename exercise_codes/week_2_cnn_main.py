import numpy as np
import torch
import torch.nn as nn
from keras.datasets import mnist
import random
import torch.optim as optim
from tqdm import tqdm
import os
import sys

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

#import custom models
from exercise_codes.week_2_models import *

def load_data():
    #load mnist dataset.
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    #since dataset is too big for cpu, reduce size
    train_random_indices = random.sample(range(len(train_images)), 60000)
    test_random_indices = random.sample(range(len(test_images)), 10000)

    train_images = train_images[train_random_indices]
    train_labels = train_labels[train_random_indices]

    test_images = test_images[test_random_indices]
    test_labels = test_labels[test_random_indices]

    #print datset shape
    print("train image shape:", train_images.shape)
    print("train label shape:", train_labels.shape)
    print("test image shape:", test_images.shape)
    print("test label shape:", test_labels.shape)
    
    return train_images, train_labels, test_images, test_labels

def train_th(model, image, label, test_image, test_label, criterion):
    
    
    image = torch.tensor(image, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)

    data_len = image.size()[0]

    optimizer = optim.SGD(model.parameters(), lr=LR)
    
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(range(data_len // BATCH_SIZE), desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for i, batch in enumerate(tqdm_batch):
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            
            image_batch = image[start:end]
            label_batch = label[start:end]
            
            optimizer.zero_grad()
            
            output = model(image_batch)
            
            loss = criterion(output, label_batch)
            epoch_loss+=loss.item()
            
            loss.backward()
            
            optimizer.step()
            
            tqdm_batch.set_postfix(loss=epoch_loss/(i+1))
        
        accuracy = validate_th(model, test_image, test_label)
    return accuracy

def validate_th(model, image, label):
    image = torch.tensor(image, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    
    data_len = image.size()[0]
    
    total_data = 0
    total_correct = 0
    tqdm_batch = tqdm(range(data_len // BATCH_SIZE), desc=f"validation")
    for i, batch in enumerate(tqdm_batch):
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            
            image_batch = image[start:end]
            label_batch = label[start:end]
                       
            output = model(image_batch)
            
            predicted_classes = torch.argmax(output, dim=1)
            total_correct += (predicted_classes == label_batch).sum().item()
            total_data += label_batch.size(0)
            
            tqdm_batch.set_postfix(accuracy=total_correct/total_data)
            
    accuracy = total_correct / total_data
    return accuracy

def train_np(model,image,label,test_image,test_label):
    
    image = np.array(image, dtype=np.float32)
    label = np.array(label, dtype=int)

    data_len = image.shape[0]
    
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(range(data_len // BATCH_SIZE), desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for i, batch in enumerate(tqdm_batch):
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            
            image_batch = image[start:end]
            label_batch = label[start:end]
            
            output = model.forward(image_batch)
            loss = model.loss(output,label_batch)
            
            epoch_loss+=loss
            
            model.backward()
            model.update_grad(LR,BATCH_SIZE)
            
            tqdm_batch.set_postfix(loss=epoch_loss/(i+1))
        
        accuracy = validate_np(model, test_image, test_label)
    return accuracy

def validate_np(model, image, label):
    image = np.array(image, dtype=np.float32)
    label = np.array(label, dtype=int)

    data_len = image.shape[0]
    
    total_data = 0
    total_correct = 0
    tqdm_batch = tqdm(range(data_len // BATCH_SIZE), desc=f"validation")
    for i, batch in enumerate(tqdm_batch):
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            
            image_batch = image[start:end]
            label_batch = label[start:end]
                       
            output = model.forward(image_batch)
            
            predicted_classes = np.argmax(output, axis=1)
            total_correct += (predicted_classes == label_batch).sum().item()
            total_data += label_batch.shape[0]
            
            tqdm_batch.set_postfix(accuracy=total_correct/total_data)
            
    accuracy = total_correct / total_data
    return accuracy   

def start(train_linear_th=False, train_linear_np=False, train_cnn_th=False, train_cnn_np=False):
    print("loading...")
    train_images, train_labels, test_images, test_labels = load_data()
    print("========================\n\n")
    
    if(train_linear_th):
        print("========================")
        print("training linear torch model...")
        model = linear_model_th()
        accuracy = train_th(model, train_images, train_labels, test_images, test_labels, criterion=nn.CrossEntropyLoss())
        print(f"training with linear_th accuracy: {accuracy * 100:.2f}%")
        print("========================")
    
    
    if(train_linear_np):
        print("========================")
        print("training linear numpy model...")
        model = linear_model_np()
        accuracy = train_np(model, train_images, train_labels, test_images, test_labels)
        print(f"training with linear_np accuracy: {accuracy * 100:.2f}%")
        print("========================")
    
    
    if(train_cnn_th):
        print("========================")
        print("training cnn torch model...")
        model = cnn_model_th()
        accuracy = train_th(model, train_images, train_labels, test_images, test_labels, criterion=nn.CrossEntropyLoss())
        print(f"training with cnn_th accuracy: {accuracy * 100:.2f}%")
        print("========================")


    if(train_cnn_np):
        print("========================")
        print("training cnn numpy model...")
        model = linear_model_np()
        accuracy = train_np(model, train_images, train_labels, test_images, test_labels)
        print(f"training with cnn_np accuracy: {accuracy * 100:.2f}%")
        print("========================")

if __name__=="__main__":
    random.seed(71)
    TOTAL_EPOCH=5
    BATCH_SIZE=48
    LR=0.001
    start(train_linear_th=False,train_linear_np=False,train_cnn_th=False,train_cnn_np=True)