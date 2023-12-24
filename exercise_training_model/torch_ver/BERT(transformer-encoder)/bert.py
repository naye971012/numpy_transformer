import sys
import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
###########################################################

class attention_block(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_len):
        super(attention_block, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.layernorm1 = nn.LayerNorm([max_len,embedding_dim])
        self.layernorm2 = nn.LayerNorm([max_len,embedding_dim])

    def forward(self, x):
        
        # Self-attention
        attention_output, _ = self.attention(x, x, x)
        x = x + attention_output
        x_norm1 = self.layernorm1(x)
        
        # Feedforward network
        ffn_output = self.ffn(x_norm1)
        x_norm1 = x_norm1 + ffn_output
        x_norm2 = self.layernorm2(x_norm1)
        
        return x_norm2

class Bert_th(nn.Module):
    def __init__(self,
                 num_blocks:int = 4,
                 sentence_length:int=50,
                 embedding_dim:int=128,
                 num_heads:int=4,
                 vocab_size:int=10000) -> None:
        """
        Args:
            num_blocks (int, optional): num of block in encoder. Defaults to 4.
            sentence_length (int, optional): length of sentence . Defaults to 50.
            embedding_dim (int, optional): embedding dimension. Defaults to 128.
            num_heads (int, optional): num of head in attention. Defaults to 4.
            vocab_size (int, optional): num of vocab size in tokenizer. Defaults to 10000.
        """
        super(Bert_th,self).__init__()
        
        assert embedding_dim%num_heads==0, "embedding dim should be divisiable to num_heads"
        
        self.num_blocks = num_blocks
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        self.init_param()
    
    def init_param(self):
        """
        use transformer encoder and cross entropy for MLM task
        """
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embedding_dim,
                                      )
        self.pos_emedding = nn.Embedding(num_embeddings=self.sentence_length,
                                      embedding_dim=self.embedding_dim,
                                      )
        
                
        self.block = nn.ModuleList([
            attention_block(embedding_dim=self.embedding_dim,
                            num_heads=self.num_heads,
                            max_len=self.sentence_length) for _ in range(self.num_blocks)
        ])

        self.output_layer = nn.Linear(self.embedding_dim, self.vocab_size)
        self.softmax = nn.Softmax(dim=2)
    

    
    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Args:
            x (np.array): [# of batch, sentence length]

        Returns:
            np.array: [# of batch, sentence length, vocab size]
        """
        x = self.embedding(x)
        x = x + self.pos_emedding(torch.arange(0,x.size(1)))
        
        for i in range(self.num_blocks):
            x = self.block[i](x)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x
    
    def predict(self, x:np.array) -> np.array:
        """
        Args:
            x (np.array): [# of batch, sentence length]

        Returns:
            np.array: [# of batch, sentence length]
        """
        #x = self.encoder(x)
        x = np.argmax(x, axis=2)
        return x
