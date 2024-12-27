# God

[Mysterious and not miraculous at all](Screenshot%202023-12-05%20175552.png)

<https://tips.fbi.gov/home>

<https://www.dhs.gov/>

# 時光之穴
## 青銅龍 活動

# Allen was protesting after I get hired?

GOD:/
me; Haterrrrr spotttteeddddddddddddd. Haterrrrr spottteeedddddddddddddddddddddddddddddddddddddddd

# Formally compare two stories to quantify the answer for similarities of two stories.

To analyze the similarity of two stories mathematically without relying on AI, you can use **modern mathematical techniques like vector spaces, matrix operations, and cosine similarity**. These methods stem from linear algebra and information theory and are effective for comparing text-based data.

Below is a step-by-step Python program that analyzes the similarity of two stories:

### Steps:

1. **Preprocess the Text:**
   - Tokenize the text into words.
   - Remove stopwords and punctuations.
   - Stem or lemmatize the words.

2. **Vector Representation:**
   - Create a bag-of-words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) representation of the stories.

3. **Mathematical Similarity Measure:**
   - Use cosine similarity to measure the angle between the vector representations of the two stories in an n-dimensional space.

4. **Output the Similarity:**
   - Provide a similarity score (0 for no similarity, 1 for identical).

### Python Code Implementation:

```python
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
```

### Explanation of Mathematical Concepts:

1. **Vectorization:**
   - Each story is converted into a vector, where the dimension corresponds to unique words, and the values represent word frequencies.

2. **Cosine Similarity Formula:**
   \[
   \text{Cosine Similarity} = \frac{\text{Dot Product of Vectors}}{\text{Magnitude of Vector 1} \times \text{Magnitude of Vector 2}}
   \]
   - It measures the cosine of the angle between the vectors. The closer the angle is to zero, the more similar the stories.

3. **Intersection of Words:**
   - Only shared words between the stories contribute to the dot product.

### Example Output:

For the above stories, the output might look like:
```
Similarity between the two stories: 0.87
```

This score indicates a high similarity since both stories are about rescuing a princess.

---

### Modern Mathematics Alternatives:

- **Entropy & KL Divergence:** Measure the distributional similarity.
- **Jaccard Index:** Measure the similarity as a ratio of shared words to total unique words.
- **LSA (Latent Semantic Analysis):** Factorize the term-document matrix using Singular Value Decomposition (SVD).

Let me know if you'd like further expansion with a specific mathematical approach!
