from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path='input.txt', block_size=128):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

    def __len__(self):
        return self.block_size

    def __getitem__(self, idx):
        with open(self.file_path, 'r', encoding="utf8") as f:
            data = f.read()
        tokenized_data = self.tokenizer.encode(data)
        start_index = idx * self.block_size
        end_index = start_index + self.block_size
        if start_index > len(tokenized_data) - 1:
            return torch.tensor([self.tokenizer.eos_token_id] * self.block_size)
        else:
            return torch.tensor(tokenized_data[start_index:end_index])

def train_model(model, tokenizer, dataset, learning_rate, num_epochs, device):
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=-1, pad_token_id=tokenizer.pad_token_id)

    for epoch in range(num_epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

# Initialize the GPT-2 model, tokenizer, and training parameters
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set up the training parameters
learning_rate = 1e-5
num_epochs = 5
block_size = 128

# Fine-tune the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
dataset = CustomDataset(tokenizer, block_size=block_size)
train_model(model, tokenizer, dataset, learning_rate, num_epochs, device)

# Save the trained model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')