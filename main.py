from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict')
def pre():

    return render_template('predict.html')


@app.route('/Predicted', methods=['POST', 'GET'])
def predicted():
    item_weight = float(request.form['item_weight'])
#      item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(
        request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_weight,  # item_fat_content,
                  item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    scaler_path = 'sc1.sav'

    sc = joblib.load(scaler_path)

    X_std = sc.transform(X)

    model_path = 'lr1.sav'
    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)

    return render_template('output.html', pred=Y_pred, i_t=int(item_weight),  # i_f_t=item_fat_content ,
                           i_w=item_type,
                           i_m=item_mrp, o_y=int(outlet_establishment_year), o_z=outlet_size, o_l=outlet_location_type, o_t=outlet_type)


@app.route('/explore')
def exp():
    return render_template('explore.html')


@app.route('/contact')
def cont():
    return render_template('contact.html')


@app.route('/product')
def prod():
    return render_template('product.html')


@app.route('/veg')
def veg():
    return render_template('veg.html')


@app.route('/fruit')
def fruit():
    return render_template('fruit.html')


@app.route('/dairy')
def dairy():
    return render_template('dairy.html')


@app.route('/meat')
def meat():
    return render_template('meat.html')


if __name__ == '__main__':
    app.run(debug=True, port=1234, host='0.0.0.0')
