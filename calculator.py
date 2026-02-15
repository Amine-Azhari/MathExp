def arrange(exp):
    expression = []
    number = ""
    for c in exp:
        if c in "+-*/=":
            if number != "":
                expression.append(number)
                number = ""
            expression.append(c)
        else : 
            number += c
    if number != "": expression.append(number)
    return expression

def helperEq(value, old:int, op):
    if (op=='' or op==None) : return value
    elif (op == '+') : return (old + value)
    elif (op == '-') : return (old - value)
    elif (op == '*') : return (old * value)
    elif (op == '/') : return (old / value)



def calc(expression):
    value = 0
    old = 0
    op = ''
    for e in expression :
        if (e=='+' or e=='-' or e=='*' or e=='/'):
            value = helperEq(value, old, op)
            old = value 
            value = 0 
            op = e 
        else :
            value = int(e)
    value = helperEq(value, old, op)
    return value

def evaluate(exp):
    expression = arrange(exp)
    return calc(expression)
            


if __name__ == "__main__":
    exp = "10+30-9"
    print(evaluate(exp))


