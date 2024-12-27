import re
from collections import Counter
from math import sqrt

# Function to preprocess the text
def preprocess(text):
    # Lowercase, remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize and remove stopwords
    stopwords = set(["and", "or", "the", "is", "in", "at", "of", "a", "to", "it", "on"])
    tokens = [word for word in text.split() if word not in stopwords]
    return tokens

# Function to create a vector representation
def text_to_vector(text):
    words = preprocess(text)
    return Counter(words)

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    # Dot product
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    # Magnitudes
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = sqrt(sum1) * sqrt(sum2)
    
    # Avoid division by zero
    if not denominator:
        return 0.0
    return numerator / denominator

# Input stories
story1 = """
Once upon a time, a young prince set out on a journey to rescue a princess locked in a tower.
"""
story2 = """
A brave knight embarked on a mission to save a princess trapped in a high castle.
"""

# Compute vectors
vector1 = text_to_vector(story1)
vector2 = text_to_vector(story2)

# Compute similarity
similarity = cosine_similarity(vector1, vector2)

# Output the result
print(f"Similarity between the two stories: {similarity:.2f}")
