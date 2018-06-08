from Projects.SIL.src.DTO.Actor import Actor
from Projects.SIL.src.DTO.Authority import Authority
class ExpertClient(Actor):
    """
    ClientExpert class
    """
    def __init__(self, user, name, identifier, email, surname, phone):
        Actor.__init__(self, user, name, identifier, email, surname, phone)
        self.authority = Authority().EXPERT_CLIENT