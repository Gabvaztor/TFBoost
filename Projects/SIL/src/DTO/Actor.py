class Actor(object):
    """
    Abstract Actor object
    """
    def __init__(self, user, name, identifier, email, surname, phone):
        self.user = user
        self.name = name
        self.surname = surname
        self.identifier= identifier
        self.phone = email
        self.phone= phone