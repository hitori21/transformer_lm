import argparse
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.models import TransformerLM
from lib.datasets import Text
from lib.preprocess import (
	prepare_text,
	extract_responses
)
from lib.trainer import Trainer
from lib.utils import (
	number_formatter,
	calculate_params
)


def SaveData(file_path, *args, **kwargs):
	data = {
		"weights": {
			"state_dict": kwargs.get("model").state_dict(),
			"n_vocab": kwargs.get("n_vocab"),
			"seq_len": kwargs.get("seq_len"),
			"n_embedding": kwargs.get("n_embedding"),
			"n_heads": kwargs.get("n_heads"),
			"n_hidden": kwargs.get("n_hidden"),
			"n_layers": kwargs.get("n_layers"),
			"dropout": kwargs.get("dropout"),
			"word_to_index": kwargs.get("w2i"),
		}
	}

	torch.save(data, file_path)
	print(
		"\n{}Training complete! Model saved to{} {}{}{}".format(
			colorama.Fore.GREEN,
			colorama.Style.RESET_ALL,
			colorama.Fore.MAGENTA,
			file_path,
			colorama.Style.RESET_ALL,
		)
	)


def main():
	parser = argparse.ArgumentParser(description='Konfigurasi data training untuk Transformer-based Language Model')

	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate yang digunakan saat pelatihan model')
	parser.add_argument('--dataset', type=str, default='dataset.json',
						help='Lokasi dataset untuk pelatihan model')
	parser.add_argument('--batch_size', type=int, default=16,
						help='Ukuran batch saat pelatihan model')

	parser.add_argument('--n_epochs', type=int, default=100,
						help='Jumlah epoch pada Transformer-based Language Model')
	parser.add_argument('--seq_len', type=int, default=32,
						help='Panjang sequence pada Transformer-based Language Model')
	parser.add_argument('--n_embedding', type=int, default=64,
						help='Jumlah embedding pada Transformer-based Language Model')
	parser.add_argument('--n_heads', type=int, default=2,
						help='Jumlah attention heads pada Transformer-based Language Model')
	parser.add_argument('--n_hidden', type=int, default=32,
						help='Jumlah hidden layer pada Transformer-based Language Model')
	parser.add_argument('--n_layers', type=int, default=2,
						help='Jumlah layer pada Transformer-based Language Model')
	parser.add_argument('--dropout', type=float, default=0.1,
						help='Dropout pada Transformer-based Language Model')

	args = parser.parse_args()

	DATASET = args.dataset
	BATCH_SIZE = args.batch_size
	LR = args.lr

	N_EPOCHS = args.n_epochs
	SEQ_LEN = args.seq_len
	N_EMBEDDING = args.n_embedding
	N_HEADS = args.n_heads
	N_HIDDEN = args.n_hidden
	N_LAYERS = args.n_layers
	DROPOUT = args.dropout

	extract_responses(DATASET, "train.txt", "valid.txt")

	word2idx, data = prepare_text("train.txt")
	val_word2idx, val_data = prepare_text("valid.txt")
	
	n_vocab = len(word2idx)

	dataset = Text(data, SEQ_LEN)
	
	val_dataset = Text(val_data, SEQ_LEN)
	
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	model = TransformerLM(n_vocab, N_EMBEDDING, N_HEADS, N_HIDDEN, N_LAYERS, DROPOUT)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)

	trainer = Trainer()

	trainer.fit(loader, val_loader, model, optimizer, n_vocab, N_EPOCHS)

	_, params = calculate_params(model)
	
	file_path = f"ckpt-{number_formatter(params)}.bin"
	SaveData(
		file_path,
		model=model,
		seq_len=SEQ_LEN,
		n_embedding=N_EMBEDDING,
		n_heads=N_HEADS,
		n_hidden=N_HIDDEN,
		n_layers=N_LAYERS,
		dropout=DROPOUT,
		n_vocab=n_vocab,
		w2i=word2idx
	)


if __name__ == "__main__":
	import colorama
	colorama.init()
	main()
