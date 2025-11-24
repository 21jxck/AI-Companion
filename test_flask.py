from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    """
    Legge un parametro in get e lo ritorna sottoforma di JSON
    """
    return jsonify({
        "content" : "Ciao ciao!",
        "language" : "Italian",
        "date" : "2025-10-13"
    })


if __name__ == '__main__':
    app.run(port=9000, debug=False)