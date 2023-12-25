from keras.datasets import fashion_mnist
import random
import numpy as np


def load_data():
    #load mnist dataset.
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

class CustomDataset:
    def __init__(self,
                 args,
                 img,
                    ) -> None:
        """
        dataset with only real data (label==1)
        """
        self.args = args
        self.img = img
        
    def __getitem__(self,idx:int):
        """
        return (input, output) tuple
        """
        _input = np.array([self.img[idx]])/255
        _output = np.array([1])
        
        return _input, _output
    
    def __len__(self):
        return len(self.img)
    

class CustomDataloader:
    """
        my custom dataloader class
        if len(dataset)%batch_size !=0, ignore remain items
    """
    def __init__(self, dataset, batch_size:int, shuffle:bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __getitem__(self, idx: int):
        output = {
            "input" : [],
            "label" : []
        }
        for i in range(self.batch_size * idx , self.batch_size * (idx+1), 1 ):
            _input, _output = self.dataset[i]
            output["input"].append(_input)
            output["label"].append(_output)

        return output
    
    def __len__(self) ->int:
        return len(self.dataset) // self.batch_size
