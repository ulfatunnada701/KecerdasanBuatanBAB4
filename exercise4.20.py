
# Exercise 4.20 - Question Answering with Transformers
# Modified from Example 4.24

from transformers import pipeline

# Load Question Answering model
question_answerer = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

# Different text context
context = """
Artificial Intelligence (AI) is increasingly used in education
to improve the learning experience. Students can use AI tools
to understand difficult subjects and complete assignments.
Teachers use AI to create personalized learning materials
and evaluate student performance.

Google developed Google Classroom to support online learning.
The platform was launched in 2014. Microsoft also provides
educational technology through Microsoft Teams.

AI technology can help students learn more efficiently,
but students should still develop their critical thinking
and problem-solving skills.
"""

# Different questions
questions = [
    "What is AI used for in education?",
    "Who uses AI to create personalized learning materials?",
    "What platform did Google develop?",
    "When was Google Classroom launched?",
    "What skills should students develop?"
]

# Question Answering
print("\n===== QUESTION ANSWERING RESULTS =====")

for i, question in enumerate(questions, start=1):

    result = question_answerer(
        question=question,
        context=context
    )

    print(f"\nQuestion {i}: {question}")
    print(f"Answer    : {result['answer']}")
    print(f"Confidence: {result['score'] * 100:.2f}%")

print("\n===== PROCESS COMPLETED =====")
