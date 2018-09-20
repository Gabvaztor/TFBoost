import ftplib
from ftplib import FTP
import traceback

class FtpConnection():
    server = None
    def connect(self, ip, port, user, password):
        """
        Connect to FTP server
        Args:
            ip:
            port:
            user:
            password:
        """
        print("Trying to connect...")
        try:
            self.server = FTP("ftp.debian.org")
            #self.server.login(user='username', passwd = 'password')
            self.server.login()
            print("Connected")
            #self.server.connect(', 21)
            #self.server.login(user='username', passwd='password')
            # You don't have to print this, because this command itself prints dir contents
            self.server.dir()

        except Exception:
            print("Can not connect because...")
            traceback.print_exc()
    def folder_to_download(self, ftp_folder, local_folder):
        # TODO (@gabvaztor) Finish
        pass

    def delete_files_from_folder(self, tfp_folder):
        # TODO (@gabvaztor) Finish
        pass

    def close_connection(self):
        """

        Close the connection

        """
        if self.server:
            self.server.close()

def main():
    ip, port, user, password = "", 12, "", ""
    folder_to_download  = ""
    local_folder = ""

    ftp_connection = FtpConnection().connect(ip, port, user, password)
    #ftp_connection.folder_to_download(ftp_folder=folder_to_download,
    #                                  local_folder=local_folder)
    #  TODO Finish
    pass

main()