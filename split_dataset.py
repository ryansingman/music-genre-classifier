import tensorflow as tf

def split(dataset, size, train_split=0.6, val_split=0.2, test_split=0.2, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    dataset = dataset.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * size)
    val_size = int(val_split * size)
    
    train_dataset = dataset.take(train_size)    
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)
    
    return train_dataset, val_dataset, test_dataset