import os
import pandas as pd

class DatasetSplitter:
    
    def __init__(self, task, val_split = 0) -> None:
        self.task = task
        self.fetch_annotations = eval(f"self.fetch_annotations_{task}")
        self.val_split = val_split
        
    def fetch_annotations_musdb18hq(self):
        
        folder_path = '/import/c4dm-datasets/MUSDB18HQ'
        
        dataframe = pd.DataFrame(columns=['folder_path', 'split'])
        
        # get train dataframe
        train_folder_path = os.path.join(folder_path, 'train')
        train_folders = os.listdir(train_folder_path)
        train_folders = [os.path.join(train_folder_path, f) for f in train_folders]
        train_df = pd.DataFrame(train_folders, columns=['folder_path'])
        train_df['split'] = 'train'
        
        #get test dataframe
        test_folder_path = os.path.join(folder_path, 'test')
        test_folders = os.listdir(test_folder_path)
        test_folders = [os.path.join(test_folder_path, f) for f in test_folders]
        test_df = pd.DataFrame(test_folders, columns=['folder_path'])
        test_df['split'] = 'test'
        
        if self.val_split > 0:
            val_df = test_df.sample(frac=self.val_split)
            test_df = test_df.drop(val_df.index)
            val_df['split'] = 'val'
            dataframe = pd.concat([train_df, test_df, val_df])
            
        else:
            dataframe = pd.concat([train_df, test_df])
            
        return dataframe
    