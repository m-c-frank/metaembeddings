import os
from pathlib import Path

base_dir = Path("data")
temp_files = os.listdir(base_dir)
temp_files = [base_dir / f for f in temp_files]
files = []
for f in temp_files:
    if f.is_file():
        files.append(f)
# print(files)

# %%

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")

# %%

import torch

def embed(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the last hidden state
    last_hidden_states = outputs.hidden_states[-1]
    embedding = last_hidden_states.mean(dim=1)
    embedding = torch.squeeze(embedding)
    return embedding

print(embed("test string a"))
print(embed("test string b"))

# %%

def column(text, width=64):
    n_chars = len(text)
    n_chunks_remainder = n_chars % width
    n_chunks = n_chars // width
    print(n_chunks)

    chunks = []
    for i in range(0, n_chunks):
        chunks.append(text[i*width:(1+i)*width])

    if n_chunks_remainder != 0:
        chunks.append(text[n_chunks*width:].ljust(width, " "))

    return chunks


# lorem = """"Lorem ipsum dolor sit amet, qui minim labore adipisicing minim sint cillum sint consectetur cupidatat.""""
#
# column_lines = column(lorem*16)
# print(column_lines)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def compress_embeddings(list_of_embeddings):
    stacked_embeddings = torch.stack(list_of_embeddings, dim=0)
    mean_embedding = torch.mean(stacked_embeddings, dim=0)
    mean_embedding = torch.squeeze(mean_embedding)
    return mean_embedding

def visualize_tsne(list_of_embeddings, perplexity):
    # Flatten each tensor and convert to a numpy array
    flattened_embeddings = torch.cat(
        [e.view(-1) for e in list_of_embeddings]
    ).numpy()

    # Reshape for TSNE if necessary, depending on your specific use case
    # Here, each embedding is treated as a separate data point
    flattened_embeddings = flattened_embeddings.reshape(
        len(list_of_embeddings), -1
    )

    # Apply t-SNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(flattened_embeddings)

    # Visualize with matplotlib
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    plt.title('t-SNE visualization of mean embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

import random

random.seed(0)
random.shuffle(files)

file_embeddings = []
for i, file in enumerate(files):
    print(file)
    text = open(file, "r").read()
    print(text)
    text_columns = column(text)
    line_embeddings = []
    for j, line in enumerate(text_columns):
        print(line)
        embedding = embed(line)
        print(embedding)
        print(embedding.shape)
        line_embeddings.append(embedding)
        if j > 1:
            break

    compressed_line_embeddings = compress_embeddings(line_embeddings)

    file_embeddings.append(compressed_line_embeddings)

    if i > 8:
        break

print(file_embeddings)
visualize_tsne(file_embeddings, perplexity = 8)

