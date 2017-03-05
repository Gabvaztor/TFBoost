class Objeto(object): #clase genérica
    nombre = None
    edad = None
    def __init__(self,nombre,edad):
        self.nombre = nombre
        self.edad = edad

    def printAtributos(self):
        print("El objeto con nombre " + self.nombre +
              " tiene " + self.edad + " años de edad y " + self.ojos + " ojos")

class Animal(Objeto): #clase no genérica
    ojos = None
    def __init__(self,nombre,edad,ojos):
        super(Objeto,self).__init__(nombre,edad)
        self.ojos = ojos

class Humano(Animal): #clase no genérica
    inteligencia = None
    nacionalidad = None
    def __init__(self,nombre,edad,ojos,inteligencia,nacionalidad):
        super(Animal, self).__init__(nombre,edad,ojos)
        self.inteligencia = inteligencia

maria = Humano("María",25,2,90,"Española")
maria.printAtributos( )