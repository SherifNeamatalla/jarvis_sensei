from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# Define the prompt
prompt = "User: Hello, You are Jarvis, my personal assistant. Jarvis:"

# Generate a response to the prompt using the model with a stop criterion
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids=input_ids, max_length=50, do_sample=True, temperature=0.7, num_return_sequences=1, stop=["\n"])
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the response to the user
print(response)