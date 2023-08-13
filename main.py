import torch
import os
import random
import colorama
#from rich.pretty import pprint

from model.models import TransformerLM
from lib.preprocess import bag_of_words, tokenize, cleaning
from lib.utils import display, list_files_by_extension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = list_files_by_extension(".bin")

FILE = input(f"Masukkan nama file model {checkpoints}: ")
MODEL = torch.load(FILE, map_location=device)
transformerlm_model = MODEL["weights"]


class Chat:
	def __init__(self):
		self.transformerlm_model = TransformerLM(
			transformerlm_model["n_vocab"],
			transformerlm_model["n_embedding"],
			transformerlm_model["n_heads"],
			transformerlm_model["n_hidden"],
			transformerlm_model["n_layers"],
			transformerlm_model["dropout"],
		).to(device)
		self.transformerlm_model.load_state_dict(transformerlm_model["state_dict"])
		self.transformerlm_model.eval()

	def generate_text(self, message, filter_words=True):
		model = self.transformerlm_model
		model_info = transformerlm_model

		seq_len = model_info["seq_len"]
		word_to_index = model_info["word_to_index"]
		message = cleaning(message)
		words = message.split()
		inputs = [word_to_index.get(word, 0) for word in words]

		if len(inputs) < seq_len:
			inputs += [0] * (seq_len - len(inputs))
		else:
			inputs = inputs[:seq_len]

		inputs = torch.LongTensor(inputs).unsqueeze(0).to(device)
		with torch.no_grad():
			logits = model(inputs)

		outputs = logits.argmax(dim=-1).squeeze().tolist()
		response_words = []
		count = 0

		for i, output in enumerate(outputs):
			if filter_words:
				if i == 0 or output != outputs[i - 1]:
					response_words.append(
						list(word_to_index.keys())[list(word_to_index.values()).index(output)]
					)
					count = 1
				else:
					count += 1
					if count == 3:
						break
			else:
				response_words.append(
					list(word_to_index.keys())[list(word_to_index.values()).index(output)]
				)

		response = " ".join(response_words)
		return response