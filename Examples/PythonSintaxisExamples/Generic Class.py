class Objeto(object): #clase genérica
    nombre = None
    edad = None
    def __init__(self,nombre,edad):
        self.nombre = nombre
        self.edad = edad

    def printAtributos(self):
        print("El objeto con nombre " + self.nombre +
              " tiene " + str(self.edad) +
              " años de edad y " + str(self.ojos) + " ojos")

class Animal(Objeto): #clase no genérica
    ojos = None
    def __init__(self,nombre,edad,ojos):
        Objeto.__init__(self, nombre,edad)
        self.ojos = ojos

class Humano(Animal): #clase no genérica
    inteligencia = None
    nacionalidad = None
    def __init__(self,nombre,edad,ojos,inteligencia,nacionalidad):
        Animal.__init__(self,nombre,edad,ojos)
        self.inteligencia = inteligencia

a = Animal("hola", 24, 3)
a.printAtributos()

maria = Humano("María",25,2,90,"Española")
maria.printAtributos( )