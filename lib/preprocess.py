import re
import json
import random

import numpy as np
from sklearn.model_selection import train_test_split

import nltk
from nltk.stem.porter import PorterStemmer


nltk.download("punkt", quiet=True)
stemmer = PorterStemmer()


def tokenize(sentence):
	return nltk.word_tokenize(sentence)


def stem(word):
	return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
	sentence_words = [stem(word) for word in tokenized_sentence]
	bag = np.zeros(len(words), dtype=np.float32)
	for idx, w in enumerate(words):
		if w in sentence_words:
			bag[idx] = 1
	return bag


def cleaning(text):
	text = re.sub(r"[^a-zA-Z0-9(),.!?\'\`]", " ", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = re.sub(r"([.!?]){2,}", r"\1", text)
	text = re.sub(r"\b(\w+)([.!?])(\w+)\b", r"\1\2 \3", text)
	text = re.sub(r"\n{2,}", "\n", text)
	text = text.strip()
	
	if not text.endswith((".", "?", "!")):
		text += "."
	
	return text
	

def extract_responses(source_file, train_file, valid_file):
	if source_file.split(".")[1] == "json":
		with open(source_file, "r") as f:
			intents_data = json.load(f)["intents"]

		conversation_set = set()

		for intent in intents_data:
			if len(intent["patterns"]) <= len(intent["responses"]):
				for idx in range(len(intent["patterns"])):
					conversation_set.add((intent["patterns"][idx], intent["responses"][idx]))
			else:
				for idx in range(len(intent["responses"])):
					conversation_set.add((intent["patterns"][idx], intent["responses"][idx]))

		sorted_conversation_set = sorted(conversation_set, key=lambda x: cleaning(x[0]))

		train_conversation, valid_conversation = train_test_split(sorted_conversation_set, test_size=0.15, random_state=42)

		with open(train_file, "w") as f:
			for pattern, response in train_conversation:
				f.write(cleaning(pattern) + " " + cleaning(response) + "\n")
		with open(valid_file, "w") as f:
			for pattern, response in valid_conversation:
				f.write(cleaning(pattern) + " " + cleaning(response) + "\n")
	else:
		with open(source_file, "r") as f:
			text_data = f.read()

		lines = text_data.splitlines()

		train_lines, valid_lines = train_test_split(lines, test_size=0.15, random_state=42)

		with open(train_file, "w") as f:
			for line in train_lines:
				f.write(cleaning(line) + "\n")
		with open(valid_file, "w") as f:
			for line in valid_lines:
				f.write(cleaning(line) + "\n")
			
			
def prepare_text(path):
	word_to_index = {}
	data = []

	with open(path, "r") as file:
		text = file.read()
	words = text.split()

	for word in words:
		if word not in word_to_index:
			word_to_index[word] = len(word_to_index)
		data.append(word_to_index[word])

	return word_to_index, data
