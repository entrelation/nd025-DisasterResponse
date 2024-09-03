import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge datasets.

    Parameters:
    messages_filepath (str): Filepath for the messages CSV file.
    categories_filepath (str): Filepath for the categories CSV file.

    Returns:
    DataFrame: Merged DataFrame containing messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on the 'id' column
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Cleans the DataFrame by processing the categories column and removing duplicates.

    Parameters:
    df (DataFrame): DataFrame containing the merged messages and categories data.

    Returns:
    DataFrame: Cleaned DataFrame with categories split into individual columns.
    """
    # Split the categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Select the first row of the categories DataFrame
    row = categories.iloc[0]
    
    # Extract a list of new column names for categories
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # Set each value to be the last character of the string and convert to numeric
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
        
        # Convert the numeric value to a binary string
        categories[column] = categories[column].apply(lambda x: format(x, 'b'))
    
    # Drop the original categories column from `df`
    df.drop(['categories'], inplace=True, axis=1)
    
    # Concatenate the original DataFrame with the new `categories` DataFrame
    df = pd.concat([df, categories], axis=1)
    
    # Drop the rows in 'related' category with different values than 0 or 1
    df = df[df['related'].isin(['0','1'])]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filepath):
    """
    Saves the DataFrame to an SQLite database.

    Parameters:
    df (DataFrame): The DataFrame to be saved.
    database_filepath (str): The file path for the SQLite database.
    
    Returns:
    None
    """
    # Create a database engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Save the DataFrame to a SQL table
    df.to_sql('Message', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
