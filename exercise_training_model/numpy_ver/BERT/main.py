#BERT = MLM 학습
from typing import Any, Tuple
import sys
import os
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import wandb

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
grand_path = os.path.dirname(grand_path)
sys.path.append(grand_path)
print(grand_path)
###########################################################

from numpy_functions import *
from numpy_models.transformer import transformer_encoder_np
from bert import Bert_np
from dataset import CustomDataset, CustomDataloader

def train(args):

    train_loader, valid_loader, tokenizer = get_loader(args)
    Bert_model = Bert_np(num_blocks=args.num_blocks,
                                     sentence_length=args.max_len,
                                     embedding_dim=args.embedding,
                                     num_heads=args.num_heads,
                                     vocab_size=len(tokenizer.vocab))
    optimizer = SGD_momentum_np()
    
    #ignore padding token when calculate accuracy
    ignore_idx = tokenizer.vocab.word2idx['[PAD]']
    
    for epoch in range(args.epoch):
        epoch_loss = 0.0
        epoch_acc = 0.0
        tqdm_batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for i, batch in enumerate(tqdm_batch):
            
            _input = np.array(batch['input'])
            _label = np.array(batch['label'])
            
            out = Bert_model.forward(_input)
            pred = Bert_model.predict(out)
            
            acc = accuracy(pred, _label, ignore_idx)
            loss = Bert_model.loss(out,_label)
            
            epoch_acc  += acc
            epoch_loss += loss
            
            Bert_model.backward()
            
            Bert_model.update_grad(optimizer, args.lr/args.batch_size)
            optimizer.step()
            
            tqdm_batch.set_postfix(loss=epoch_loss/(i+1), acc=epoch_acc/(i+1))

            if(i%10==0):
                print("____label_____")
                print(_label[0])
                print("____input_____")
                print(_input[0])
                print("____pred______")
                print(pred[0])
                print()

                wandb.log({"train_acc": epoch_acc/(i+1),
                        "train_loss": epoch_loss/(i+1)})
        
        validate(args,Bert_model,valid_loader,ignore_idx)
        
def validate(args, model, valid_loader, ignore_idx:int):

    valid_loss = 0.0
    valid_acc = 0.0
    tqdm_batch = tqdm(valid_loader, desc=f"Validation")
    for i, batch in enumerate(tqdm_batch):
            
        _input = np.array(batch['input'])
        _label = np.array(batch['label'])
            
        out = model.forward(_input)
        pred = model.predict(out)
            
        acc = accuracy(pred, _label, ignore_idx)
        loss = model.loss(out,_label)
            
        valid_acc  += acc
        valid_loss += loss
            
        tqdm_batch.set_postfix(loss=valid_loss/(i+1), acc=valid_acc/(i+1))

        if(i%10==0):
            print("____label_____")
            print(_label[0])
            print("____input_____")
            print(_input[0])
            print("____pred______")
            print(pred[0])
            print()

    wandb.log({"validation_acc": valid_acc/(i+1),
               "validation_loss": valid_loss/(i+1)})
    
def get_loader(args) -> Tuple:
    """
    for convenience, I use huggingface dataset library.
    
    return train_lodaer, valid_loader, trained_tokenizer
    """

    #prepare data list
    if args.data_name=="Abirate/english_quotes":
        data = load_dataset("Abirate/english_quotes")
        data_list = data['train']['quote']
        """
        print(data['train']['quote'][0])
            '“Be yourself; everyone else is already taken.”'
        """
    else:
        ############# TODO: make type(data_list)==List when you train custom data ##########
        data = load_dataset(args.data_name)
        data_list = None
        ####################################################################################
    
    #shuffle data
    np.random.shuffle(data_list)
    
    #prepare tokenizer
    tokenizer =Word_tokenizer_np()
    #train tokenizer
    tokenizer.train(data_list)
    
    train_data = CustomDataset(args, data_list[100:], tokenizer)
    valid_data = CustomDataset(args, data_list[:100], tokenizer)
    train_loader = CustomDataloader(train_data, args.batch_size, shuffle=False)
    valid_loader = CustomDataloader(valid_data, args.batch_size, shuffle=False)
    
    return train_loader, valid_loader, tokenizer
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Bert basic parser')

    #Data arguments
    parser.add_argument('--data_name', type=str, default="Abirate/english_quotes", \
                                        help='name of training data in huggingface.')
    
    #Train arguments
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')


    #Bert model arguments
    parser.add_argument('--num_blocks', type=int, default=6, help='num of block of bert')
    parser.add_argument('--max_len', type=int, default=50, help='max sentence length')
    parser.add_argument('--embedding', type=int, default=256, help='embedding layer dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='num of head for self attention')

    # Parse the arguments
    args = parser.parse_args()


    # Initialize wandb with specific settings
    wandb.init(
        project='Numpy_models',  # Set your project name
        name="Bert_numpy",
        config={                       # Set configuration parameters (optional)
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'num_blocks': args.num_blocks,
            'max_len': args.max_len,
            'embedding': args.embedding,
            'num_heads': args.num_heads,
            'optimizer': "sgd_momentum",
            'data_name': args.data_name
        }
    )


    train(args)
    
