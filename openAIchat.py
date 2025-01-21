from openai import OpenAI
import torch
from datasets import load_from_disk
import json
from sklearn.metrics import mean_absolute_error

# Set your OpenAI API key
client = OpenAI(api_key="***")

# Load dataset from disk
dataset = load_from_disk('./DATA_ROOT/multitasktext-sepTrue/test')

# Extract text after the last '[SEP]' token
def extract_text_after_sep(text):
    return text.split('[SEP]')[-1]

# Generate historical annotation summary for each annotator without including the current text
def generate_historical_summary(dataset):
    ann_dict = {}
    for example in dataset:
        ann_id = example['ann_ids']
        text = extract_text_after_sep(example['post'])
        rating = example['labels']
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

# Function to get rating prediction using GPT-3.5 Turbo
def get_rating_prediction(text):
    response = client.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a model that predicts the toxicity rating of text from 0 to 4, where 0 is the least toxic and 4 is the most toxic."},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        stop=None,
        max_tokens=5,
        n=1
    )
    return float(json.loads(response['choices'][0]['message']['content']))

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
