import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))

import os
import openai
from getpass import getpass

openai.api_key = getpass("OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key

from aimakerspace.openai_utils.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

# Example 1: Comparing similar concepts (puppies vs dogs)
puppy_sentence = "I love puppies!"
dog_sentence = "I love dogs!"
puppy_vector = embedding_model.get_embedding(puppy_sentence)
dog_vector = embedding_model.get_embedding(dog_sentence)
similarity_score = cosine_similarity(puppy_vector, dog_vector)
print(f"Similarity between '{puppy_sentence}' and '{dog_sentence}': {similarity_score:.4f}")

# Example 2: Comparing contrasting sentiments
puppy_sentence = "I love puppies!"
cat_sentence = "I dislike cats!"
puppy_vector = embedding_model.get_embedding(puppy_sentence)
cat_vector = embedding_model.get_embedding(cat_sentence)
similarity_score = cosine_similarity(puppy_vector, cat_vector)
print(f"Similarity between '{puppy_sentence}' and '{cat_sentence}': {similarity_score:.4f}")

# Example 3: Vector arithmetic (King - Man + Woman ≈ Queen)
king_vector = np.array(embedding_model.get_embedding("King"))
man_vector = np.array(embedding_model.get_embedding("man"))
woman_vector = np.array(embedding_model.get_embedding("woman"))
vector_calculation_result = king_vector - man_vector + woman_vector
queen_vector = np.array(embedding_model.get_embedding("Queen"))
similarity_score = cosine_similarity(vector_calculation_result, queen_vector)
print(f"Similarity between (King - Man + Woman) and Queen: {similarity_score:.4f}")
