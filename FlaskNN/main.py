from cnn import CNN
from flask import Flask, jsonify

app = Flask(__name__)
cnn = CNN([(5, 5, 1, 20), (5, 5, 20, 20)], [(500, 300)], [300, 7])

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/train")
def train():
    global cnn
    return jsonify(cnn.salute())

if __name__ == '__main__':
    app.run()
