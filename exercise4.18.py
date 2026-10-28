
# Exercise 4.18 - Text Analysis Using NLTK

import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Different text: Artificial Intelligence in Education
sentence = """
Artificial Intelligence is transforming education around the world.
Students use AI tools to understand difficult subjects and improve
their learning experience. Teachers can use intelligent systems
to create personalized lessons and evaluate student performance.
In the United States, organizations such as Google and Microsoft
are developing artificial intelligence technologies to support
digital learning. However, students should use AI responsibly
and continue developing their critical thinking skills.
"""

# Tokenization
tokens = nltk.word_tokenize(sentence)

print("\n===== TOKENIZATION =====")
print(tokens)

# Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)

print("\n===== POS TAGGING =====")
print(tagged)

# Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)

print("\n===== NAMED ENTITY RECOGNITION =====")
print(entities)

# Additional analysis
print("\n===== TEXT STATISTICS =====")
print("Total Tokens:", len(tokens))
print("Total Words:", sum(token.isalpha() for token in tokens))
print("Total Sentences:", len(nltk.sent_tokenize(sentence)))
