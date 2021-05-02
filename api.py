from flask import Flask, render_template, request
from flask_cors import CORS
from predictComp import expLatex

app = Flask(__name__)
CORS(app)

@app.route('/imgdata/<data>', methods=['GET'])
def apiclientCall(data):
    finalVal = data.replace("SLASH", "/")
    # print("got : ", data)
    # print("replaced", finalVal)
    return expLatex(finalVal)

if __name__ == "__main__":
    app.run(debug=True)