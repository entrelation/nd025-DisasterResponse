import sys

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Args:
        database_filepath (str): Filepath to the SQLite database.

    Returns:
        X (pd.Series): Features (messages).
        Y (pd.DataFrame): Target variables (categories).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', con=engine)
    X = df.message.values
    Y = df.iloc[:,-36:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text data.

    Args:
        text (str): Text to be tokenized.

    Returns:
        clean_tokens (list): List of cleaned tokens.
    """
    # Remove unwanted characters using regex
    text = re.sub(r"[.,'?#():!]", '', text)

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))

    clean_tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word.lower() not in stop_words]
    
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with predefined parameters.

    Returns:
        pipeline (Pipeline): A scikit-learn Pipeline object.
    """
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,1))),
                ('tfidf', TfidfTransformer())
            ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, min_samples_split=4)))
    ])

    return pipeline

# def build_model():
#     """
#     Build a machine learning pipeline.

#     Returns:
#         cv (GridSearchCV): Grid search cross-validation object with pipeline.
#     """
#     pipeline = Pipeline([
#         ('vect', CountVectorizer(tokenizer=tokenize)),
#         ('tfidf', TfidfTransformer()),
#         ('clf', MultiOutputClassifier(RandomForestClassifier()))
#     ])
    
#     # Define parameters for grid search
#     parameters = {
#         'vect__ngram_range': [(1, 1), (1, 2)],
#         'clf__estimator__n_estimators': [50, 100],
#         'clf__estimator__max_depth': [None, 10, 20],
#         'clf__estimator__min_samples_split': [2, 3],
#     }
    
#     cv = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=10, cv=3, verbose=2, n_jobs=-1, random_state=42)
    
#     return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model performance and print classification report.

    Args:
        model: Trained model.
        X_test (pd.Series): Test features.
        Y_test (pd.DataFrame): True labels for test data.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f'Category: {category}\n{classification_report(Y_test.iloc[:, i], Y_pred[:, i])}')



def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Args:
        model: Trained model.
        model_filepath (str): Filepath to save the pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()