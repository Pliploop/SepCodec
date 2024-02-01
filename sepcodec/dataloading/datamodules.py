from pytorch_lightning import LightningDataModule

from sepcodec.dataloading.datasets import Musdb18HQ
from sepcodec.dataloading.datasetsplitter import DatasetSplitter
from torch.utils.data import DataLoader


class SepDataModule(LightningDataModule):
    
    def __init__(self, task, val_split = 0, batch_size = 32, num_workers = 4, target_sample_rate = 32000, target_length_s = 6) -> None:
        super().__init__()
        self.task = task
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sample_rate = target_sample_rate
        self.target_length_s = target_length_s
        
        self.splitter = DatasetSplitter(task, val_split)
        self.annotations = self.splitter.fetch_annotations()
        
        if self.task == 'musdb18hq':
            self.dataset_class = Musdb18HQ
        
    def setup(self, stage=None):
        self.train_annotations = self.annotations[self.annotations['split'] == 'train']
        self.test_annotations = self.annotations[self.annotations['split'] == 'test']
        if self.val_split > 0:
            self.val_annotations = self.annotations[self.annotations['split'] == 'val']
            
        self.train_dataset = self.dataset_class(self.train_annotations, target_sample_rate=self.target_sample_rate, target_length_s=self.target_length_s, train = True, transform = True)
        self.test_dataset = self.dataset_class(self.test_annotations, target_sample_rate=self.target_sample_rate, target_length_s=self.target_length_s, train = False, transform = False)
        if self.val_split > 0:
            self.val_dataset = self.dataset_class(self.val_annotations, target_sample_rate=self.target_sample_rate, target_length_s=self.target_length_s, train = True, transform = False)
    
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        