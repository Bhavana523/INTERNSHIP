import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_data
data = load_data("dataset.csv")
features = data[['python', 'ml', 'dl', 'score', 'time_spent']]
similarity_matrix = cosine_similarity(features)
def recommend_skills(student_index):

    similarity_scores = list(enumerate(similarity_matrix[student_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for i in similarity_scores[1:4]:

        student_data = data.iloc[i[0]]

        recommendations.append({
            "skill": student_data['next_skill'],
            "confidence": round(i[1] * 100, 2)
        })

    return recommendations
print(recommend_skills(0))