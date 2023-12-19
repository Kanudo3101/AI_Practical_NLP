import os
import time
from glob import glob
import random
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch_optimizer as optim

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

        with open(file_path, 'r', encoding="utf8") as f:
            data = f.read()
        self.examples = self.tokenizer.batch_encode_plus(
            [data[i: i + block_size] for i in range(0, len(data), block_size)],
            max_length=block_size,
            padding='max_length',
            truncation=True)['input_ids']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=5, no_repeat_ngram_size=2, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(i, skip_special_tokens=True) for i in output]

# Prepare tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to('cpu') #cuda

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
#    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#    model.resize_token_embeddings(len(tokenizer))

# Load training data
train_path = 'input.txt'
train_dataset = TextDataset(tokenizer, train_path, block_size=128)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Training settings
num_epochs = 3
learning_rate = 1e-4
warmup_steps = 1e2
total_steps = len(train_dataloader) * num_epochs

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Start training
for epoch in range(num_epochs):
    model.train()

    count=1
    for batch in train_dataloader:
        print('Epochs:{}/{}, Counter:{}/{}'.format(epoch+1, num_epochs, count, len(train_dataloader)))
        count = count+1

        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        scheduler.step()

# Save the trained model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Generate text
prompt = "Once upon a time"
generated_texts = generate_text(model, tokenizer, prompt)
for text in generated_texts:
    print(text)