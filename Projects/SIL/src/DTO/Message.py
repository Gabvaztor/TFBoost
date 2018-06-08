class Message():
    """
    Message class
    """
    def __init__(self, client, folder, moment, subject, body):
        self.client = client
        self.folder = folder
        self.moment = moment
        self.subject = subject
        self.body = body