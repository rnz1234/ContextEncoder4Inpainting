from tqdm import tqdm



def train_model(model, data_loaders, dataset_sizes):
    for batch in tqdm(data_loaders['train']):
        #import pdb
        #pdb.set_trace()
        inputs = batch['orig_image'] 

    
