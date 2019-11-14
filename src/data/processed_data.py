import os
import numpy as np
import pandas as pd


def read_data():
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    
    train_df = pd.read_csv(train_file_path, index_col = 'PassengerId')
    test_df = pd.read_csv(test_file_path, index_col = 'PassengerId')
    test_df['Survived'] = -100
    
    df = pd.concat([train_df, test_df], sort=-False, axis=0)
    return df


def process_data(df):
    return(df
           .assign(Title = lambda x: x.Name.map(get_title))
           .pipe(fill_missing_values)
           .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low', 'low', 'high', 'very_high']))
           .assign(AgeState = lambda x: np.where(x.Age >= 18, 'Adult', 'Child'))
           .assign(FamilySize = lambda x: x.Parch + x.SibSp + 1)
           .assign(IsMother = lambda x: np.where(((x.Age > 18) & (x.Parch > 0) & (x.Title != 'Miss') & (x.Sex == 'female')), 1,0))
           .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin))
           .assign(Deck = lambda x: x.Cabin.map(get_deck))
           .assign(IsMale = lambda x: np.where(x.Sex == 'male', 1, 0))
           .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
           .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
           .pipe(reorder_columns)
          )
    

# modify the function to reduce number of titles and return more meaningful functions
def get_title(name):
    title_map = {
        'mr': 'Mr',
        'mrs': 'Mrs',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'miss': 'Miss',
        'mlle': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'sir': 'Sir',
        'jonkheer': 'Sir',
        'dr': 'Officer',
        'major': 'Officer',
        'capt': 'Office',
        'col': 'Officer',
        'lady': 'Lady',
        'the countess': 'Lady',
        'dona': 'Lady'
    }
    first_name_with_title = name.split(',')[1]
    raw_title = first_name_with_title.split('.')[0]
    title = raw_title.strip().lower()
    return title_map[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def fill_missing_values(df):
    #Embarked
    df.Embarked.fillna('C', inplace=True)
    
    # Fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    
    #Age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df


def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')
    
    df.loc[df.Survived != -100].to_csv(write_train_path)
    
    columns = [column for column in df.columns if column != 'Survived']
    df.loc[df.Survived == -100][columns].to_csv(write_test_path)
    
if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
