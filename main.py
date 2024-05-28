import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read the CSV file
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to calculate similarity and add columns
def add_similarity_columns(df):
    keywords = df.iloc[:, 1].tolist()  # Use the second column for keywords

    # Vectorization
    vectorizer = TfidfVectorizer().fit_transform(keywords)
    vectors = vectorizer.toarray()
    cosine_sim_matrix = cosine_similarity(vectors)

    # Create columns for similar keywords and scores
    num_similar = 5  # Maximum number of similar keywords to display per keyword
    for n in range(1, num_similar + 1):
        df[f'Similar Keyword {n}'] = ''
        df[f'Similarity Score {n}'] = ''

    for i in range(len(keywords)):
        similar_pairs = []
        scores = []
        for j in range(len(keywords)):
            if i != j and cosine_sim_matrix[i][j] > 0:
                similar_pairs.append((keywords[j], cosine_sim_matrix[i][j]))

        # Sort pairs by descending score
        similar_pairs = sorted(similar_pairs, key=lambda x: x[1], reverse=True)

        # Add similar keywords and scores to the appropriate columns
        for n in range(min(num_similar, len(similar_pairs))):
            df.at[i, f'Similar Keyword {n+1}'] = similar_pairs[n][0]
            df.at[i, f'Similarity Score {n+1}'] = similar_pairs[n][1]

    return df

# Function to write the modified DataFrame to a CSV file
def write_csv(df, output_file_path):
    df.to_csv(output_file_path, index=False)

# Paths for input and output files
input_file_path = 'motsclessimilarity.csv'  # Replace with your file path (copy paste path)
output_file_path = 'motsclessimilarity-export.csv'  # Replace with your output path

# Read the CSV file
df = read_csv(input_file_path)

# Add similarity columns
df = add_similarity_columns(df)

# Write the modified DataFrame to a new CSV file
write_csv(df, output_file_path)

# Display the modified DataFrame (optional)
df
