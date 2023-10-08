import argparse
import os
#import enchant
from nltk.tokenize import word_tokenize
from nltk.corpus import words

def is_word_punctuation(word):
    for char in word:
        if not (char >= 'a' and char <= 'z'):
            return True
    return False    

def word_exists(word):
    return word in words.words()



# # Create a British English dictionary object
# dictionary = enchant.Dict("en_GB")  # "en_GB" for British English

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="A script that evaluates how many generated words exist in English Language.")

# Add command-line arguments with specific flag names
parser.add_argument('--output_file', '-o', type=str, help="Path to output file name.")

# Parse the command-line arguments
args = parser.parse_args()

output_file_path = os.path.join('data', 'shakespeare_char',args.output_file)
generated_text = ""
#read output file
with open(output_file_path, 'r') as f:
    generated_text = f.read()
print(f"length of dataset in characters: {len(generated_text):,}")

# Tokenize the input string into words
generated_words = word_tokenize(generated_text)

# Check the spelling of each word in the input string
count = 0
for word in generated_words:
    word = word.lower()
    if is_word_punctuation(word):
        continue
    if not word_exists(word):
        print(word)
        count+=1

print(count)
print(len(generated_words))
print("Percentage of words not found in dictionary: ", (count/(len(generated_words)))*100)
