import numpy as np
import os
import joblib
from flask import Flask, request, render_template, make_response


app = Flask(__name__, static_url_path='/static')
model = joblib.load('model.pkl')


@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/verificar', methods=['POST'])
def verificar():
	hair = request.form['hair']
	feathers = request.form['feathers']
	eggs = request.form['eggs']
	milk = request.form['milk']
	airborne = request.form['airborne']
	aquatic = request.form['aquatic']
	predator = request.form['predator']
	venomous = request.form['venomous']
	toothed = request.form['toothed']
	breathes = request.form['breathes']
	fins = request.form['fins']
	legs = request.form['legs']
	tail = request.form['tail']
	domestic = request.form['domestic']
	backbone = request.form['backbone']

	teste = np.array([[hair,feathers,eggs,aquatic,milk,airborne,predator,venomous,toothed,breathes,fins,legs,tail,domestic,backbone]])
	
	print(":::::: Dados de Teste ::::::")
	print("Cabelo: {}".format(hair))
	print("Penas: {}".format(feathers))
	print("Ovos: {}".format(eggs))
	print("Nada: {}".format(aquatic))
	print("Leite: {}".format(milk))
	print("Voa: {}".format(airborne))
	print("Predador: {}".format(predator))
	print("toothed: {}".format(toothed))
	print("breathes: {}".format(breathes))
	print("fins: {}".format(fins))
	print("legs: {}".format(legs))
	print("tail: {}".format(tail))
	print("domestic: {}".format(domestic))
	print("Venenoso: {}".format(venomous))
	print("Venenoso: {}".format(venomous))
	print("\n")

	classe = model.predict(teste)[0]
	print("Classe Predita: {}".format(str(classe)))

	return render_template('template.html',classe=str(classe))

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5500))
        app.run(host='127.0.0.1', port=port)

