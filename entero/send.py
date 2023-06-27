from flask import Flask, render_template, request
from archivomodelo import realizar_prediccion

app = Flask(__name__, template_folder='templates', static_folder='staticFiles')

premisas =['Young players engage in the sport of Water polo while others watch.','Two doctors perform surgery on patient.','Two women are embracing while holding to go packages.']


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/pregunta1.html')
def pregunta1():
    return render_template("pregunta1.html")

@app.route('/pregunta2.html')
def pregunta2():
    return render_template("pregunta2.html")

@app.route('/pregunta3.html')
def pregunta3():
    return render_template("pregunta3.html")

@app.route('/respuesta',methods=['POST'])
def respuesta():
    premisa = request.form['premisa']
    hipotesis = request.form['hipotesis']

    premisa = int(premisa)
    
    clasificacion = realizar_prediccion(premisas[premisa], hipotesis)
    #return "<h1>" + clasificacion + "</h1>"
    return render_template('respuesta.html', clasificacion = clasificacion)

if __name__ == '__main__':
    app.run(debug=True)