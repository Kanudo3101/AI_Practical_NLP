import sys
from glob import glob
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(i, skip_special_tokens=True) for i in output]

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')

# Enable evaluation mode
model.eval()

# Define the function to generate predictions
def generate_text2(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

## Test the function
# print(generate_text(model, tokenizer, 'Micro controller is '))

## Generate text
#prompt = "8085 Architecher"
#generated_texts = generate_text(model, tokenizer, prompt)
#for text in generated_texts:
#    print(text)

if __name__ == "__main__":
    query = None

    while True:
        if not query:
            query = input("\nPrompt: ")
        if query in ['quit', 'q', 'exit']:
            sys.exit()

        generated_texts = generate_text(model, tokenizer, query)
        #print(generated_texts)
        str1 = ''
        for text in generated_texts:
            str1 += text
        print('\nResponse: {}'.format(str1))

        # chat_history.append((query, result['answer']))
        query = None