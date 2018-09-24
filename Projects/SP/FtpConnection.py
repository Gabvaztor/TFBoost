"""
    Steps:
        - Create a new array connection in the end of file
        - Set "current_data" variable with the name of your new array connection

        - You must have Python3 installed.
        NOTE: All lines that must be modified have "TODO" word
              You must have installed python3

    Array connections (position information):
         1: ftp_name
         2: port, (if None, get ftp default)
         3: user name
         4: password
         5: ftp folder where files are located
         6: local folder where files will be downloaded

    If any downloaded file has "gz" extension, it will be uncompressed in the same folder and, after that,
    will be deleted
"""
from ftplib import FTP
import traceback

class FtpConnection():
    server = None
    ip = None
    port = 12
    user = None
    password = None
    attempts = 0
    actual_dir = ""
    compressed_files = []

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
            if not port:
                self.server = FTP(ip)
            else:
                self.server = FTP()
                self.server.connect(host=ip, port=port)
            self.server.login(user=user, passwd=password)
            #self.server.login()
            print("Connected")
            #self.server.connect(', 21)
            #self.server.login(user='username', passwd='password')
            # You don't have to print this, because this command itself prints dir contents
            self.server.dir()
        except Exception:
            print("Can not connect because...")
            traceback.print_exc()
    def folder_to_download(self, ftp_folder, local_folder, unzip=False):
        """
        Connect to a folder and download all files
        Args:
            ftp_folder:
            local_folder:

        Returns:

        """
        self.server.cwd(ftp_folder)
        self.actual_dir = ftp_folder
        filenames = self.server.nlst()  # get filenames within the directory
        print(filenames)
        for filename in filenames:
            print("Downloading: " + filename + " ...")
            local_filename = os.path.join(local_folder, filename)
            file = open(local_filename, 'wb')
            self.server.retrbinary('RETR ' + filename, file.write)
            if local_filename[-2:] == "gz":
                self.compressed_files.append(local_filename)
            file.close()
        #self.server.quit()


    def delete_files_from_ftp_folder(self, ftp_folder):
        """
        Delete files from FTP folder
        Args:
            tfp_folder:
        """
        try:
            if self.actual_dir != ftp_folder:
                self.server.cwd(ftp_folder)
            print("Folder status before delete")
            filenames = self.server.nlst()  # get filenames within the directory
            print("Deleting files from FTP folder if any...")
            for filename in filenames:
                self.server.delete(filename)
            print("Actual folder status after delete")
            filenames = self.server.nlst()  # get filenames within the directory
            print(filenames)
        except Exception:
            traceback.print_exc()
            self.attempts += 1
            if self.reset_connection():
                self.delete_files_from_ftp_folder(ftp_folder=ftp_folder)

    def reset_connection(self):
        success = False
        if self.attempts < 3:
            if self.server:
                self.close_connection()
                if self.ip and self.port and self.password and self.user:
                    self.connect(self.ip, self.port, self.password, self.user)
                    success = True
                else:
                    print("Can not reconnect")
            else:
                if self.ip and self.port and self.password and self.user:
                    self.connect(self.ip, self.port, self.password, self.user)
                    success = True
                else:
                    print("Can not reconnect")
        else:
            print("After 3 attempts, it can not connect")
        return success

    def close_connection(self):
        """

        Close the connection

        """
        if self.server:
            #self.server.quit()  # This is the “polite” way to close a connection
            self.server.close()
            print("Connection closed")

    def unzip_files(self):
        if self.compressed_files:
            try:
                import gzip
                import shutil
                import os
                for compressed_f in self.compressed_files.copy():
                    with gzip.open(compressed_f, 'rb') as f_in:
                        with open(compressed_f[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(compressed_f)
            except Exception:
                print("Files could not be unzipped")
                print("Files has not been deleted from server and local for prevention")
                traceback.print_exc()
                return False
        return True

def main(ip, port, user, password, ftp_folder, local_folder):

    ftp_connection = FtpConnection()
    ftp_connection.connect(ip, port, user, password)
    ftp_connection.folder_to_download(ftp_folder=ftp_folder,
                                      local_folder=local_folder)
    if ftp_connection.unzip_files():
        ftp_connection.delete_files_from_ftp_folder(ftp_folder=ftp_folder)
    if ftp_connection.server:
        ftp_connection.close_connection()

if __name__ == "__main__":
    """
    Array connections (position information):
         1: ftp_name
         2: port, (if None, get ftp default)
         3: user name
         4: password
         5: ftp folder where files are located
         6: local folder where files will be downloaded
         
    If any downloaded file has "gz" extension, it will be uncompressed in the same folder and, after that, 
    will be deleted
    """

    prism_connection_data = ["prism.nacse.org", None, "anonymous", "email@email.com",
                             "monthly/ppt/1895/", "Downloads\\"]
    localhost_connection = ["localhost", 9898, "", "", "", "Downloads\\"]
    # TODO Create a new one array connection here:
    new_connection = ["ftp_name",
                      port number,
                      "user_name",
                      "password",
                      "ftp_folder_name/",
                      "local_folder"]

    "GZIp --> Unzip: 1: All files will be downloaded, unzipped and, after that, that will be deleted"

    # TODO To change when change connection (in the right, set the name of your array)
    current_data = localhost_connection

    ip = current_data[0]
    port = current_data[1]
    user = current_data[2]
    password = current_data[3]
    ftp_folder = current_data[4]
    local_folder = current_data[5]

    main(ip=ip, port=port, user=user, password=password,
         ftp_folder=ftp_folder, local_folder=local_folder)

