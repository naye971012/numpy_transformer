import numpy as np

def accuracy(pred, label, ignore_idx:int):
    
    # Mask for elements to ignore
    ignore_mask = (label != ignore_idx)

    # Count correct predictions excluding ignored indices
    correct = np.sum((pred == label) & ignore_mask)
    # Count total non-ignored elements
    total = np.sum(ignore_mask)

    acc = correct / total if total != 0 else 0.0
    return acc