import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

flims = {
    'title': ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 'Iron Man', 'Titanic'],
    'genres': [
        'Sci-Fi Thriller',
        'Sci-Fi Adventure',
        'Action Crime Drama',
        'Action Superhero',
        'Action Superhero',
        'Romance Drama'
    ]
}

df = pd.DataFrame(flims)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(title, n=3):
    if title not in df['title'].values:
        return f"Movie '{title}' not found in database."
    
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_movies = [df['title'][i[0]] for i in sim_scores[1:n+1]]
    return top_movies

user_movie = input("Enter a movie you liked: ")
recommendations = recommend(user_movie, n=3)

print(f"\nSince you liked '{user_movie}', you may also like:")
print(recommendations)
