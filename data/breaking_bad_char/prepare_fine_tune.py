"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import torch
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A script that prepares data for GPT training")

# Add command-line arguments with specific flag names
parser.add_argument('--num_scripts', '-o', type=str, help="Number of scripts to use as input.")

# Parse the command-line arguments
args = parser.parse_args()

num_scripts = int(args.num_scripts)
data = ""
for i in range(num_scripts):
    input_file_path = os.path.join(os.path.dirname(__file__), "script_" + str(i+1) + ".txt")

    #read input file
    with open(input_file_path, 'r') as f:
        data += f.read()

print(f"length of dataset in characters: {len(data):,}")

# Write the aggregated data to an input file
input_file = os.path.join(os.path.dirname(__file__), "input.txt")

with open(input_file, 'w') as f:
    f.write(data)


# Define the list of allowed characters [need to match the shakespeare data for finetuning]
allowed_characters = " \n!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Function to filter characters based on the allowed list
def filter_characters(input_characters, allowed_characters):
    filtered_characters = [char for char in input_characters if char in allowed_characters]
    return filtered_characters

# Call the filter_characters function
data = filter_characters(data, allowed_characters)

# Using the same number and order of distinct characters as the Shakespeare data to ease finetuning

meta_path = '/Users/aliasgarsabir/nanoGPT/data/shakespeare_char/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
# TODO want to make this more general to arbitrary encoder/decoder schemes
stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(vocab_size)

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
