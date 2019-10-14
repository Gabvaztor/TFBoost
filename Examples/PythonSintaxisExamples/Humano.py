class Humano:

    ojos = 2
    piernas = 2
    inteligencia = None
    masaMuscular = 50
    longitudDeBiceps = 20
    fuerzaBrazo = masaMuscular*longitudDeBiceps
    nombre = None

    def __init__(self, nombre,gjasfhjgal,masaMuscular):
        self.nombre = nombre
        self.ojos = gjasfhjgal
        self.masaMuscular = masaMuscular

    def mostrarNombre(self):
        print("El nombre del humano es: " + self.nombre + ", ojos = " +  str(self.ojos) + str(self.masaMuscular))

maria = Humano("2", 2,58181817)
maria.mostrarNombre()








