import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple,List, Dict
import os
import sys

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from numpy_models.commons.linear import Linear_np
from numpy_models.activations.relu import Relu_np
from numpy_models.activations.softmax import softmax_np
from numpy_models.normalization.batchnorm import Batch_Normalization_1D_np

def main():
    pass

if __name__=="__main__":
    main()