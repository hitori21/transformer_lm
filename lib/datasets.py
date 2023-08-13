import torch
from torch.utils.data import Dataset
	

class Text(Dataset):
	def __init__(self, data, seq_len):
		self.seq_len = seq_len
		self.data = data
		
	def __len__(self):
		return len(self.data) - self.seq_len
	
	def __getitem__(self, index):
		inputs = torch.LongTensor(self.data[index : index + self.seq_len])
		targets = torch.LongTensor(self.data[index + 1 : index + self.seq_len + 1])
		return inputs, targets