import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import json
import plotly
import pandas as pd

from collections import Counter
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and clean text by removing stopwords and lemmatizing.
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        list: A list of cleaned and lemmatized words.
    """
    # Remove unwanted characters using regex
    text = re.sub(r"[.,'?#():!]", '', text)  # Remove commas, apostrophes, question marks, hashtags

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word.lower() not in stop_words]

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', con=engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.iloc[:,-36:]
    categories_counts = categories.sum().sort_values(ascending=False)
    categories_names = categories_counts.index

    # Tokenize all messages
    all_words = []
    for message in df.message:
        all_words.extend(tokenize(message))

    # Count word frequencies
    word_freq = Counter(all_words)

    # Get the top 100 most common words
    top_10_words = word_freq.most_common(10)

    # Separate words and their frequencies for plotting
    words, frequencies = zip(*top_10_words)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # First graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Second graph
                {
            'data': [
                Scatter(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Total of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        },

        #Third graph
                {
            'data': [
                Scatter(
                    x=words,
                    y=frequencies,
                    mode='markers',
                    marker=dict(
                        size=[f for f in frequencies],  # Scale the marker size for better visualization
                        color=frequencies,                  # Use frequencies for marker color
                        showscale=True,                      # Show color scale
                        colorscale='Viridis',                # Specify a color scale
                        sizemode='area',                     # Set size mode to area for better scaling
                        sizeref=2.*max(frequencies)/(100.**2), # Adjust sizeref for scaling
                    ),
                    text=words,  # Hover text to display word
                )
            ],

            'layout': {
                'title': 'Top 10 Words in Messages',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Word",
                    'automargin': True
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()