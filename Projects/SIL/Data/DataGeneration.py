"""

The aim of this class is the creation of all necessary data.

"""


def some_func():
    print("hi!")
# Igualamos var a una función
var = some_func
var()
print(var)

def retorna_algo():
    return 3
var = retorna_algo
print(var)

var = retorna_algo()
print(var)