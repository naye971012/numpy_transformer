import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from week_6_models import *

def train(model, dataloader):
    
    for epoch in range(TOTAL_EPOCH):
        epoch_loss = 0.0
        tqdm_batch = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{TOTAL_EPOCH}")
        for i, batch in enumerate(tqdm_batch):
            
            input_text = np.array(batch['data'])
            output_text = np.array(batch['label'])
            
            pred, att_map = model.forward(input_text)
            
            loss = model.loss(pred,output_text)
            
            epoch_loss+=loss
            
            model.backward()
            model.update_grad(LR,BATCH_SIZE)
            
            tqdm_batch.set_postfix(loss=epoch_loss/(i+1))
            
            if(len(dataloader)==i):
                break
        
        if(epoch>10):
            plt.xticks(np.arange(10), input_text[0])
            plt.yticks(np.arange(10), input_text[0])
            plt.imshow(att_map[0], cmap='viridis', interpolation='nearest')
            plt.colorbar()  # 컬러바 추가
            plt.show()
            
    return model

def main():
    #define dataset and loader
    dataset = CustomDataset(make_len=10)
    dataloader = CustomDataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #define RNN model
    model = model_with_attention(input_channel=10,output_channel=1)
    
    #train start!
    model = train(model, dataloader)


if __name__=="__main__":
    BATCH_SIZE = 1
    TOTAL_EPOCH = 20
    LR = 1e-4
    main()