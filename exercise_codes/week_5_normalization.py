import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple,List, Dict
import os
import sys
from keras.datasets import fashion_mnist
import random
from tqdm import tqdm

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from week_5_models import *

def load_data():
    #load mnist dataset.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    #since dataset is too big for cpu, reduce size
    train_random_indices = random.sample(range(len(train_images)), 30000)
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


def main():
    print("loading...")
    train_images, train_labels, test_images, test_labels = load_data()
    print("========================\n\n")

    model_1 = model_without_norm()
    model_2 = model_with_layer_norm()
    model_3 = model_with_batch_norm()
    
    print("========train without normalization=======================")
    train_np(model_1,train_images,train_labels,test_images,test_labels)
    print("\n\n")
    print("========train with layer normalization=======================")
    train_np(model_2,train_images,train_labels,test_images,test_labels)
    print("\n\n")
    print("========train with batch normalization=======================")
    train_np(model_3,train_images,train_labels,test_images,test_labels)

if __name__=="__main__":
    
    TOTAL_EPOCH = 5
    LR = 0.001
    BATCH_SIZE = 12
    
    main()