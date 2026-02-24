import sys

# simple arithmetic script expects three arguments:
#   python l1.py <num1> <operator> <num2>
# where <operator> is one of + - * /
if len(sys.argv) != 4:
    print("Usage: python l1.py <num1> <operator> <num2>")
    sys.exit(1)

try:
    num1 = int(sys.argv[1])
    operator = sys.argv[2]
    num2 = int(sys.argv[3])
except ValueError:
    print("Error: first and third arguments must be integers")
    sys.exit(1)

if operator == "+":
    print(num1+num2)
elif operator == "-":
    print(num1-num2)    
elif operator == "*":
    print(num1*num2)
elif operator == "/":
    if num2 == 0:
        print("Error: Division by zero")
    else:
        print(num1/num2)  
else:    print("Error: Invalid operator")
