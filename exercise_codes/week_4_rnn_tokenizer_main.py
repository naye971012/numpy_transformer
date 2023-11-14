import numpy as np
import os
import sys
from tqdm import tqdm

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from exercise_codes.week_4_models import *
from numpy_models.tokenizer.word_tokenizer import Word_tokenizer_np

"""
    download small chatting data from here
    https://www.kaggle.com/datasets/projjal1/human-conversation-training-data    
"""

DATA_PATH = "numpy_transformer/data/human_chat.txt"

def process_data():
    
    #read chatting txt file    
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        chatting_txt = file.read()

    #split by line
    chatting_list = chatting_txt.split('\n')
    
    #change format "Human 1: ~~" into "~~"
    output_list = list()
    for line in chatting_list:
        try:
            output_list.append(line.split(':')[1])
        except:
            continue
        
    return output_list

def train(model, dataloader):
    
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for i, batch in enumerate(tqdm_batch):
            
            input_text = batch['input']
            output_text = batch['output']
            
            pred = model.forward(input_text)
            
            loss = model.loss(pred,output_text)
            
            epoch_loss+=loss
            
            model.backward()
            model.update_grad(LR,BATCH_SIZE)
            
            tqdm_batch.set_postfix(loss=epoch_loss/(i+1))
        
    return model

def inference(model):
    text = ""
    while(True):
        text = input()
        if text=='quit':
            break
        
        output = model.predict(text)
        print(output)

def main():
    
    #get data list
    output_list = process_data()
    
    #define tokenizer and train with our data
    tokenizer = Word_tokenizer_np()
    tokenizer.train(output_list)
    
    #define dataset and loader
    dataset = CustomDataset(output_list)
    dataloader = CustomDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #define RNN model
    model = myModel(tokenizer)
    
    #train start!
    model = train(model, dataloader)

    #inference
    print("train done! write chatting input.")
    inference(model)

if __name__=="__main__":
    BATCH_SIZE = 1
    TOTAL_EPOCH = 1
    LR = 1e-3
    main()