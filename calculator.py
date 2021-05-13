import math

def preProcess(s):
    # For Mul
    s = s.replace("\\times", '*')
    s = s.replace("x", '*')
    # For power
    s = s.replace("^", "**")
    # For square-root
    s = s.replace("\sqrt", "math.sqrt")
    # For opening bracket
    s = s.replace("{", "(")
    # For closing bracket
    s = s.replace("}", ")")
    # for left(
    s = s.replace("\left", "(")
    # for right)
    s = s.replace("\\right", ")")
    # for div
    s = s.replace("\div", "/")
    return s

def preProcessFrac(s):
    frac = s.find("\\frac")
    if(frac > -1):
        start = s.find('(', frac) + 1
        count = 1
        string = "("
        while(count):
            if(s[start] == '('):
                count += 1
            elif(s[start] == ')'):
                count -= 1
            string += s[start]
            start += 1
        val = latexEval(string)
        rep = "\\frac" + string
        s = s.replace(rep, str(val) + '/')
    return s

def latexEval(s):
    print(s)
    s = s.replace(" ", "")
    s = preProcess(s)
    # For Fraction
    s = preProcessFrac(s)
    return eval(s)

# print(latexEval("3/2"))
# print(latexEval("\\frac{3+\\frac{1}{2}+6}{\sqrt{4}+1}"))