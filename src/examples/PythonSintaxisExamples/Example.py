class Persona():

    def __init__(self, edad, nombre, sexo, nivel):
        self.edad = edad
        self.nombre = nombre
        self.sexo = sexo
        self.nivel = nivel

    def imprimir(self):
        print(self.edad // self.nivel)

    def nombrar(self):
        if self.nombre == "Jesus":
            print("Nada")
        if self.nombre == "Gabriel":
            print(self.nivel)
        if 1==0:
            pass
        else:
            print("No es ninguna de las anteriores")

gabriel = Persona(3, "Gabriel", "Femenino", 1)
jesus = Persona(4, "Jesus", "Masculino", 2)

gabriel.nombrar()