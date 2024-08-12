# Disaster Response Pipeline Project

This project is a web application which purpose is to analyze disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages. 
It processes message data, trains a machine learning model, and provides a web interface for classifying new messages.

## Project Structure

- `README.md`: this file
- `requirements.txt`: list of packages to be installed to run the application

- **app**: Contains the Flask web application files.
  - `templates`: HTML templates for the web app.
  - `run.py`: Flask application to run the web app.

- **data**: Contains the dataset and scripts for data processing.
  - `disaster_categories.csv`: Dataset with categories.
  - `disaster_messages.csv`: Dataset with messages.
  - `process_data.py`: Script to process the data and save to a database.

- **models**: Contains the machine learning model and training script.
  - `train_classifier.py`: Script to train the classifier.

## Getting Started

### Prerequisites

- Python 3.x
- nltk
- Flask
- Pandas
- Plotly
- Scikit-learn
- SQLAlchemy


### Installing

1. Clone the repository:
    
   `git clone https://github.com/entrelation/nd025-DisasterResponse.git`
   `cd nd025-DisasterResponse`

3. Install dependencies:

    `pip install -r requirements.txt`


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database:

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    
This command creates a new file in the `data/` folder:
    
    `DisasterResponse.db`: SQLite database to save cleaned data.
        
- To run ML pipeline that trains classifier and saves in a pickle file:

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

This command creates a new file in the `models/` folder:

    `classifier.pkl`: Serialized model saved as a pickle file.

2. Run the following command in the app's directory to run your web app.

    `python run.py`

3. Go to http://0.0.0.0:3001/
