import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')

# Enable evaluation mode
model.eval()

# Define the function to generate predictions
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

## Test the function
#print(generate_text('Once upon a time'))

if __name__ == "__main__":
    query = None

    while True:
        if not query:
            query = input("\nPrompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()

        generated_texts = generate_text(query)
        print('\nResponse: {}'.format(generated_texts))

        # chat_history.append((query, result['answer']))
        query = None