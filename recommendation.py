import sys
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from flask import Flask, request, jsonify

# MongoDB connection
try:
    uri = os.environ.get("MONGODB_URI", "mongodb+srv://pratikkhodka137:Khodka@cluster0.3nkyf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    connection = MongoClient(uri)
    db = connection.test
except Exception as e:
    print(f"MongoDB connection failed: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Fetch data from MongoDB
collection1 = db.userwithcats
collection2 = db.placeupdates

try:
    userList = pd.DataFrame(list(collection1.find()))
    placeList = pd.DataFrame(list(collection2.find()))
except Exception as e:
    print(f"Failed to fetch data from MongoDB: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Debug: Check raw data
print(f"User data sample: {userList.head(1).to_dict()}", file=sys.stderr)
print(f"Place data sample: {placeList.head(1).to_dict()}", file=sys.stderr)

# Preprocess user category (using 'category' field)
userList['category'] = userList['category'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
userList["category"] = userList["category"].str.split(", ")
userList = userList.explode("category")
userList["category"] = userList["category"].str.strip().str.lower().fillna("unknown")

# Preprocess place category
placeList["category"] = placeList["category"].str.strip().str.lower().fillna("unknown")

# Merge dataframes
merged_df = userList.merge(placeList, left_on="category", right_on="category", how="inner")
merged_df = merged_df[['username', 'email', 'password', 'category', 'heading', 'image', 'para']]

# Debug: Check merged data
print(f"Merged data sample: {merged_df.head(1).to_dict()}", file=sys.stderr)
print(f"Merged columns: {merged_df.columns.tolist()}", file=sys.stderr)

# Clean category field
merged_df["category"] = merged_df["category"].str.replace(" ", "", regex=True)

# Create tags with emphasis on category
merged_df['tags'] = (merged_df['category'] + ' ' + 
                     merged_df['category'] + ' ' + 
                     merged_df['para'] + ' ' + 
                     merged_df['username'] + ' ' + 
                     merged_df['email'] + ' ' + 
                     merged_df['heading'])
new_dataframe = merged_df[['email', 'heading', 'image', 'tags', 'para', 'category']].copy()
new_dataframe.loc[:, 'tags'] = new_dataframe['tags'].str.lower()

# Handle missing values in tags
new_dataframe['tags'] = new_dataframe['tags'].fillna("")

# Debug: Verify new_dataframe
print(f"Total entries in new_dataframe: {len(new_dataframe)}", file=sys.stderr)
print(f"new_dataframe sample: {new_dataframe.head(1).to_dict()}", file=sys.stderr)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_dataframe['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

def recommend(email):
    """Recommend up to 6 unique places based on email, ensuring para and heading are included."""
    print(f"Searching for email: {email}", file=sys.stderr)
    
    if email not in new_dataframe['email'].values:
        print(f"Email '{email}' not found in DataFrame", file=sys.stderr)
        return []
    
    # Get the user's index and category (first match)
    user_idx = new_dataframe[new_dataframe['email'] == email].index[0]
    user_category = new_dataframe.loc[user_idx, 'category']
    print(f"User category: {user_category}", file=sys.stderr)
    
    # Filter by category (optimization)
    category_matches = new_dataframe[new_dataframe['category'] == user_category]
    print(f"Category matches found: {len(category_matches)}", file=sys.stderr)
    
    # Get similarity scores
    distances = similarity[user_idx]
    sorted_places = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:]  # Skip self
    
    recommended_places = []
    seen_headings = set()
    
    # Collect recommendations
    for i, score in sorted_places:
        place = new_dataframe.iloc[i]
        print(f"Considering place: {place['heading']} with para: {place['para']}", file=sys.stderr)
        if pd.isna(place['heading']) or pd.isna(place['para']):
            print(f"Warning: Missing heading or para for index {i}", file=sys.stderr)
            continue
        if place['heading'] not in seen_headings:
            recommended_places.append({
                'heading': place['heading'],
                'image': place['image'] if not pd.isna(place['image']) else "default_image_url",
                'para': place['para'],
                'category': place['category']
            })
            seen_headings.add(place['heading'])
            if len(recommended_places) == 6:
                print("Returning 6 recommendations", file=sys.stderr)
                break
    
    print(f"Returning {len(recommended_places)} recommendations", file=sys.stderr)
    return recommended_places

# Flask application
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Recommendation Service! Use /recommend with POST or GET."}), 200

@app.route('/recommend', methods=['POST', 'GET'])
def recommend_endpoint():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email') if data else None
    else:  # GET
        email = request.args.get('email')

    if not email:
        return jsonify({"error": "No email provided"}), 400

    try:
        recommendations = recommend(email)
        if not recommendations:
            return jsonify({"message": "No recommendations found for this email"}), 200
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Main execution for command-line or Flask
if __name__ == "__main__":
    if len(sys.argv) > 1:  # Command-line mode
        email = sys.argv[1]
        try:
            recommendations = recommend(email)
            print(json.dumps(recommendations))  # Output JSON to stdout
        except Exception as e:
            print(json.dumps({"error": f"Recommendation failed: {str(e)}"}), file=sys.stderr)
            sys.exit(1)
    else:  # Flask mode
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)