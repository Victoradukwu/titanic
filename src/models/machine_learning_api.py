import os
import json
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request

app = Flask(__name__)

# model_path = os.path.join(os.path.pardir, os.path.pardir, 'models')
model_path = 'models'
model_filepath = os.path.join(model_path, 'lr_model.pkl')
scaler_filepath = os.path.join(model_path, 'lr_scaler.pkl')

scaler = pickle.load(open(scaler_filepath, 'rb'))
model = pickle.load(open(model_filepath, 'rb'))

columns = ['Age', 'Fare', 'FamilySize', 'IsMother', 'IsMale', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', \
           'Deck_E', 'Deck_F', 'Deck_G', 'Deck_Z', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady', \
           'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir', \
           'Fare_Bin_very_low', 'Fare_Bin_low','Fare_Bin_high','Fare_Bin_very_high', 'Embarked_C', 'Embarked_S', \
           'Embarked_Q', 'AgeState_Adult', 'AgeState_Child'
          ]

@app.route('/api', methods=['POST'])
def make_predictions():
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    
    passenger_ids = df['PassengerId'].ravel()
    actuals = df['Survived'].ravel()
    
    X = df[columns].values.astype('float')
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    df_response = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': predictions, 'Actual': actuals})
    return df_response.to_json()

if __name__ == '__main__':
    app.run(debug=True)
