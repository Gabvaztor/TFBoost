class Clase:

    property_1 = 2
    property_2 = ""
    property_3 = None

    def __init__(self,nombre = "Sin nombre", piernas = "Sin piernas", tipo = "Sin tipo"):
        self.nombre = nombre
        self.piernas = piernas
        self.tipo = tipo

    def nombrar(self,property = "Snooby"):
        self.property_2 = property

    def print(self):
        print("Ojos = " + str(self.property_1) +
              " y Nombre = " + self.property_2)

mascota = Clase()
mascota2 = Clase()

mascota2.nombrar("Raichu")
mascota2.print()
mascota.nombrar()
mascota.print()

persona = Clase("Nombre",2,"Persona")
persona.print()

print("Tengo una mascota que se llama %s y"
      " tiene %s ojos"%(mascota.property_2,mascota.property_1))

