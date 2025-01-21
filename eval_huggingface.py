import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import mean_absolute_error

# Load dataset from disk
dataset = load_from_disk('./DATA_ROOT/multitasktext-sepTrue/test')

# Extract text after the last '[SEP]' token
def extract_text_after_sep(text):
    return text.split('[SEP]')[-1]

# Generate historical annotation summary for each annotator without including the current text
def generate_historical_summary(dataset):
    ann_dict = {}
    for example in dataset:
        ann_id = example['ann_id']
        text = extract_text_after_sep(example['post'])
        rating = example['label']
        if ann_id not in ann_dict:
            ann_dict[ann_id] = []
        ann_dict[ann_id].append((text, rating))
    
    return ann_dict

# Generate historical summaries
ann_dict = generate_historical_summary(dataset)

# Add embeddings to the dataset with historical information, excluding the current text
def process_example(example):
    ann_id = example['ann_id']
    current_text = example['post']
    historical_annotations = [f"[{text}] is rated as [{rating}]" for text, rating in ann_dict[ann_id] if text != extract_text_after_sep(current_text)]
    historical_text = "The annotator has also annotated these texts from a scale of 0 to 4, where 0 is the least toxic and 4 is the most toxic: " + ', '.join(historical_annotations)
    combined_text = f"{historical_text} [SEP] {current_text}"
    return {
        'combined_text': combined_text,
        'label': example['label']
    }

dataset = dataset.map(process_example)

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set up the text generation pipeline
text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Function to get rating prediction using the Llama 2 model
def get_rating_prediction(text):
    sequences = text_gen_pipeline(
        text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    reply = sequences[0]['generated_text']
    # Extract rating from the reply; assuming the reply contains a numeric rating
    try:
        rating = float(reply.strip())
    except ValueError:
        rating = 0.0  # or any default value/error handling
    return rating

# Iterate over the dataset and make predictions
predictions = []
for example in dataset:
    combined_text = example['combined_text']
    predicted_rating = get_rating_prediction(combined_text)
    predictions.append(predicted_rating)

# Evaluate the model predictions
true_labels = [example['label'] for example in dataset]
mae = mean_absolute_error(true_labels, predictions)
print(f"MAE: {mae}")
