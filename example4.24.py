
# Example 4.24 - Transformers2.py

from transformers import pipeline

# Membuat model Question Answering
question_answerer = pipeline('question-answering')

# Pertanyaan dan konteks
result = question_answerer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
})

# Menampilkan hasil
print(result)
