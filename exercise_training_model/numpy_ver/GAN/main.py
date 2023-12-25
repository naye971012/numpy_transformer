from typing import *
import sys
import os
import numpy as np
from tqdm import tqdm
import wandb
import argparse

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
grand_path = os.path.dirname(grand_path)
sys.path.append(grand_path)
###########################################################

from numpy_functions import *
from numpy_models.gan import Generator_np,Discriminator_np
from dataset import load_data, CustomDataloader, CustomDataset


def train(args):

    train_loader, valid_loader = get_loader(args)
    
    generator_model = Generator_np(args.batch_size,args.c,args.w,args.h)
    discriminator_model = Discriminator_np((args.w,args.h),args.c)

    optimizer = SGD_momentum_np()
    
    for epoch in range(args.epoch):
        epoch_generator_loss = 0.0
        epoch_discriminator_loss = 0.0
        epoch_loss = 0.0
        tqdm_batch = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for i, batch in enumerate(tqdm_batch):
            
            _input = np.array(batch['input'])
            _label = np.array(batch['label'])
            
            #generate fake images
            gausian_noise = generator_model.make_normal_distribution()
            generated_output = generator_model(gausian_noise)
            
            #evaluate fake images
            generated_discriminator_output = discriminator_model(generated_output)
            generator_loss = discriminator_model.loss(generated_discriminator_output, np.zeros_like(_label))
            
            #save gradient of generator-discriminator step
            d_prev = discriminator_model.backward()
            generator_model.backward(d_prev)
            
            #update fake_image step gradient (in fact, this should be done with gan_dis2)
            discriminator_model.update_grad("gan_dis1",optimizer,args.lr)
            generator_model.update_grad("gan_gen1",optimizer,args.lr)
            
            #evaluate real images
            discriminator_output = discriminator_model(_input)
            discriminator_loss = discriminator_model.loss(discriminator_output,_label)

            #save gradient of discriminator step
            discriminator_model.backward()

            #update real image step gradient
            discriminator_model.update_grad("gan_dis2",optimizer,args.lr)

            #update step
            optimizer.step()
            
            epoch_loss+= (generator_loss+discriminator_loss)
            epoch_generator_loss+= generator_loss
            epoch_discriminator_loss+= discriminator_loss
            
            tqdm_batch.set_postfix(epoch_loss=epoch_loss/(i+1),
                                   gen_loss=epoch_generator_loss/(i+1),
                                   dis_loss=epoch_discriminator_loss/(i+1))
            
            if(i%10==0):
                #wandb.log({"generator_loss": epoch_generator_loss/(i+1),
                #           "discriminator_loss": epoch_discriminator_loss/(i+1),
                #           "epoch_loss": epoch_loss/(i+1)})
                pass

        visualize(args, generator_model)

def visualize(args, model):
    pass

def get_loader(args) -> Tuple:
    """
    for convenience, I use huggingface dataset library.
    
    return train_lodaer, valid_loader, trained_tokenizer
    """
    
    train_images, train_labels, test_images, test_labels = load_data()
    
    train_data = CustomDataset(args, train_images)
    valid_data = CustomDataset(args, test_images)
    train_loader = CustomDataloader(train_data, args.batch_size, shuffle=False)
    valid_loader = CustomDataloader(valid_data, args.batch_size, shuffle=False)
    
    return train_loader, valid_loader


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Bert basic parser')

    #Data arguments
    parser.add_argument('--data_name', type=str, default="fashion_mnist", \
                                        help='defulat model train fashion_mnist')
    parser.add_argument('--c', type=int, default=1, help='channel size')
    parser.add_argument('--w', type=int, default=28, help='width size')
    parser.add_argument('--h', type=int, default=28, help='height size')
    
    #Train arguments
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=1, help='epoch')

    # Parse the arguments
    args = parser.parse_args()

    """
    # Initialize wandb with specific settings
    wandb.init(
        project='Numpy_models',  # Set your project name
        name="Bert_numpy",
        config={                       # Set configuration parameters (optional)
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epoch': args.epoch,
            'optimizer': "sgd_momentum",
            'data_name': args.data_name
        }
    )
    """

    train(args)
    