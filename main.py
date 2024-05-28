import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fonction pour lire le fichier CSV
def read_csv(file_path):
    return pd.read_csv(file_path)

# Fonction pour calculer la similarité et ajouter des colonnes
def add_similarity_columns(df):
    keywords = df.iloc[:, 1].tolist()  # Utiliser la deuxième colonne pour les mots-clés

    # Vectorisation
    vectorizer = TfidfVectorizer().fit_transform(keywords)
    vectors = vectorizer.toarray()
    cosine_sim_matrix = cosine_similarity(vectors)

    # Créer des colonnes pour les mots-clés similaires et les scores
    num_similar = 5  # Nombre maximum de mots-clés similaires à afficher par mot-clé
    for n in range(1, num_similar + 1):
        df[f'Similar Keyword {n}'] = ''
        df[f'Similarity Score {n}'] = ''

    for i in range(len(keywords)):
        similar_pairs = []
        scores = []
        for j in range(len(keywords)):
            if i != j and cosine_sim_matrix[i][j] > 0:
                similar_pairs.append((keywords[j], cosine_sim_matrix[i][j]))

        # Trier les paires par score décroissant
        similar_pairs = sorted(similar_pairs, key=lambda x: x[1], reverse=True)

        # Ajouter les mots-clés similaires et les scores dans les colonnes appropriées
        for n in range(min(num_similar, len(similar_pairs))):
            df.at[i, f'Similar Keyword {n+1}'] = similar_pairs[n][0]
            df.at[i, f'Similarity Score {n+1}'] = similar_pairs[n][1]

    return df

# Fonction pour écrire le DataFrame modifié dans un fichier CSV
def write_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

# Chemins pour les fichiers d'entrée et de sortie
input_file_path = 'motsclessimilarity.csv'  # Remplacez par votre chemin de fichier (copier-coller le chemin)
output_file_path = 'motsclessimilarity-export.csv'  # Remplacez par votre chemin de sortie

# Lire le fichier CSV
df = read_csv(input_file_path)

# Ajouter des colonnes de similarité
df = add_similarity_columns(df)

# Écrire le DataFrame modifié dans un nouveau fichier CSV
write_csv(df, output_file_path)

# Afficher le DataFrame modifié (optionnel)
df
