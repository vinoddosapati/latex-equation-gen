from flask import Flask, render_template, request
from flask_cors import CORS

from calculator import latexEval
from predictComp import expLatex

import latex2mathml.converter

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/imgdata/<data>', methods=['GET'])
def apiclientCall(data):
    finalVal = data.replace("SLASH", "/")
    print("got : ", data)
    print("replaced", finalVal)
    return expLatex(finalVal)

@app.route('/cal/<data>', methods=['GET'])
def getCalValue(data):
    finalVal = data.replace("FSLASH", "\\")
    finalVal = finalVal.replace("SLASH", "/")
    print("--",finalVal,"---")
    result = latexEval(finalVal)
    print("res ", result)
    return str(result)

@app.route('/mathml/<equation>', methods=['GET'])
def getMathMlValue(equation):
    mathml_output = ""
    print("---eq ", equation)
    finalVal = equation.replace("FSLASH", "\\")
    finalVal = finalVal.replace("SLASH", "/")
    mathml_output = latex2mathml.converter.convert(finalVal)
    print("mathml out, ", mathml_output)
    return mathml_output
    
@app.route('/0.0.0.0:8080/imgdata/<data>', methods=['GET'])
def apiclientCall1(data):
    finalVal = data.replace("SLASH", "/")
    print("got : ", data)
    print("replaced", finalVal)
    return expLatex(finalVal)

@app.route('/0.0.0.0:8080/cal/<data>', methods=['GET'])
def getCalValue1(data):
    finalVal = data.replace("FSLASH", "\\")
    finalVal = finalVal.replace("SLASH", "/")
    print("--",finalVal,"---")
    result = latexEval(finalVal)
    print("res ", result)
    return str(result)

@app.route('/0.0.0.0:8080/mathml/<equation>', methods=['GET'])
def getMathMlValue1(equation):
    mathml_output = ""
    print("---eq ", equation)
    finalVal = equation.replace("FSLASH", "\\")
    finalVal = finalVal.replace("SLASH", "/")
    mathml_output = latex2mathml.converter.convert(finalVal)
    print("mathml out, ", mathml_output)
    return mathml_output


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)