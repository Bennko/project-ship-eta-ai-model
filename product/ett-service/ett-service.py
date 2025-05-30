from flask import Flask, jsonify, request
from joblib import load
import numpy as np

app = Flask(__name__)

meta_model_hamburg = load('./models_RotHam/meta_model.joblib')
model1_hamburg = load('./models_RotHam/model1.joblib')
model2_hamburg = load('./models_RotHam/model2.joblib')
model3_hamburg = load('./models_RotHam/model3.joblib')

meta_model_rotterdam = load('./models_FelRot/meta_model.joblib')
model1_rotterdam = load('./models_FelRot/model1.joblib')
model2_rotterdam = load('./models_FelRot/model2.joblib')
model3_rotterdam = load('./models_FelRot/model3.joblib')

@app.route('/hamburg', methods=['POST'])
def ettHamburg():
    data = request.get_json()
    features = [data['latitude'], data['longitude'], data['sog'], data['cog'], data['th'], data['shiptype'], data['endLongitude'], data['endLatitude'], data['pastTravelTime']]
    prediction1 = model1_hamburg.predict([features])[0] # 'lat' 'long' 'sog' "COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime"(seconds)
    prediction2 = model2_hamburg.predict([features])[0]
    prediction3 = model3_hamburg.predict([np.array(features, dtype=object)])[0]

    modelStack = np.column_stack((prediction1, prediction2, prediction3))
    prediction = meta_model_hamburg.predict(modelStack)    
    return jsonify({'ett': int(prediction)})


@app.route('/rotterdam', methods=['POST'])
def ettRotterdam():
    data = request.get_json()
    features = [data['latitude'], data['longitude'], data['sog'], data['cog'], data['th'], data['shiptype'], data['endLongitude'], data['endLatitude'], data['pastTravelTime']]
    prediction1 = model1_rotterdam.predict([features])[0] # 'lat' 'long' 'sog' "COG", "TH", "shiptype", "EndLongitude", "EndLatitude", "pastTravelTime"(seconds)
    prediction2 = model2_rotterdam.predict([features])[0]
    prediction3 = model3_rotterdam.predict([np.array(features, dtype=object)])[0]

    modelStack = np.column_stack((prediction1, prediction2, prediction3))
    prediction = meta_model_rotterdam.predict(modelStack)
    return jsonify({'ett': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)