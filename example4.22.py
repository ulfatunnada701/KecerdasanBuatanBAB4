
# Example 4.22 The NLTK.py Program
# https://www.nltk.org/

import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

sentence = """
Artificial intelligence (AI), is intelligence demonstrated by machines,
    unlike the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize its
    chance of successfully achieving its goals.[3] Colloquially, the term
    "artificial intelligence" is often used to describe machines (or computers)
    that mimic "cognitive" functions that humans associate with the human mind,
    such as "learning" and "problem solving".[4]
"""

# Tokenization
tokens = nltk.word_tokenize(sentence)
print("===== TOKENS =====")
print(tokens)

# Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("\n===== POS TAGGING =====")
print(tagged)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)
print("\n===== NAMED ENTITIES =====")
print(entities)
