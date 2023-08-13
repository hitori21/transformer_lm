import torch
import torch.nn as nn
import time
import datetime
import colorama
from datetime import timedelta

from lib.utils import calculate_params

class Trainer:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	def fit(self, train_dataloader, valid_dataloader, model, optimizer, vocab_size, num_epochs):
		prev_loss = float("inf")
		prev_avg_val_loss = float("inf")
		start_time = time.monotonic()

		print(
			"{}[{}]{} {}Training model TransformerLM{}".format(
				colorama.Fore.MAGENTA,
				datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				colorama.Style.RESET_ALL,
				colorama.Fore.GREEN,
				colorama.Style.RESET_ALL,
			)
		)
		
		_, params = calculate_params(model)
		
		print(
			"{}[{}]{} {}Trainable parameters:{} {}{:,}{}".format(
				colorama.Fore.MAGENTA,
				datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				colorama.Style.RESET_ALL,
				colorama.Fore.GREEN,
				colorama.Style.RESET_ALL,
				colorama.Fore.YELLOW,
				params,
				colorama.Style.RESET_ALL,
			)
		)

		model.train()
		model.to(self.device)

		criterion = nn.CrossEntropyLoss()
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

		for epoch in range(num_epochs):
			for inputs, targets in train_dataloader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				optimizer.zero_grad()
				logits = model(inputs)
				loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
				loss.backward()
				optimizer.step()

			elapsed_time = time.monotonic() - start_time
			elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))

			model.eval()
			total_val_loss = 0.0
			total_val_samples = 0

			with torch.no_grad():
				for val_inputs, val_targets in valid_dataloader:
					val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
					val_logits = model(val_inputs)
					val_loss = criterion(val_logits.view(-1, vocab_size), val_targets.view(-1))
					total_val_loss += val_loss.item() * val_inputs.size(0)
					total_val_samples += val_inputs.size(0)

			avg_val_loss = total_val_loss / total_val_samples
			if (epoch + 1) % (num_epochs // 10) == 0:
				loss_str = "{:.8f}".format(loss.item())
				if loss.item() <= prev_loss:
					loss_color = colorama.Fore.YELLOW + loss_str + colorama.Style.RESET_ALL
				else:
					loss_color = colorama.Fore.RED + loss_str + colorama.Style.RESET_ALL

				avg_val_loss_str = "{:.8f}".format(avg_val_loss)
				if avg_val_loss <= prev_avg_val_loss:
					avg_val_loss_color = colorama.Fore.YELLOW + avg_val_loss_str + colorama.Style.RESET_ALL
				else:
					avg_val_loss_color = colorama.Fore.RED + avg_val_loss_str + colorama.Style.RESET_ALL

				print(
					"{}[{}]{} {}Epoch:{} {}{:04d}/{:04d}{} {}Train Loss:{} {} {}Valid Loss:{} {} {}Elapsed Time:{} {}{}{}".format(
						colorama.Fore.MAGENTA,
						datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						colorama.Style.RESET_ALL,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						colorama.Fore.BLUE,
						epoch + 1,
						num_epochs,
						colorama.Style.RESET_ALL,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						loss_color,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						avg_val_loss_color,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						colorama.Fore.GREEN,
						elapsed_formatted,
						colorama.Style.RESET_ALL
					)
				)
				prev_loss = loss.item()
				prev_avg_val_loss = avg_val_loss

			model.train()

			scheduler.step(avg_val_loss)
