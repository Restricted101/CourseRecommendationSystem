from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Enable CORS for cross-origin requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests from frontend

# Load the dataset with proper encoding (ISO-8859-1)
file_path = './cleaned_questions.csv'  # Make sure to update this path
departments = pd.read_csv(file_path, encoding='ISO-8859-1')  # Use ISO-8859-1 encoding to fix encoding issues

# Ensure dataset has required columns
required_columns = ['department', 'question', 'strand']
assert all(col in departments.columns for col in required_columns), f"Dataset must have {required_columns}"

# Vectorize department features (combining strand and department)
department_features = departments['strand'].fillna('') + ' ' + departments['department'].fillna('')
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(department_features)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the frontend (index.html)

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()

    # Get user inputs
    target_strand = data['strand'].strip().lower()
    entrance_exam_score = data['entrance_exam_score']
    desired_department = data['desired_department'].strip().lower()
    answers = data['answers']  # Answers to the shuffled questions

    # Initialize department scores
    department_scores = {dept: 0 for dept in departments['department'].unique()}

    # Content-based filtering using strand and desired course
    student_features = target_strand + ' ' + desired_department
    student_features_tfidf = vectorizer.transform([student_features])

    # Compute cosine similarity between user preferences and department features
    cosine_similarities = cosine_similarity(student_features_tfidf, tfidf_matrix)

    # Ask Randomized Questions and Count "Yes" Responses
    for idx, answer in enumerate(answers):
        if answer == 'yes':
            department_scores[departments['department'].iloc[idx]] += 1  # Increase score for the department

    # Bonus for Strand Match
    for dept in department_scores.keys():
        dept_strands = departments[departments['department'].str.lower() == dept]['strand'].iloc[0].lower().split(', ')
        if target_strand in dept_strands:
            department_scores[dept] += 1  # Add 1 for strand match

    # Bonus for Desired Course
    if desired_department in department_scores:
        department_scores[desired_department] += 1  # Add 1 for desired course match

    # Entrance Exam Consideration (Bonus if Passed)
    if entrance_exam_score >= 80:
        for dept in department_scores.keys():
            department_scores[dept] += 1  # Add 1 for all departments if the entrance exam score is above threshold

    # Integrating Cosine Similarity Scores
    for idx, dept in enumerate(department_scores.keys()):
        department_scores[dept] += cosine_similarities[0][idx]  # Cosine similarity adjustment

    # Sort by Highest Score
    sorted_departments = sorted(department_scores.items(), key=lambda x: x[1], reverse=True)

    # Return Top 3 Recommended Courses
    recommendations = [{"department": dept, "score": score} for dept, score in sorted_departments[:3]]

    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
