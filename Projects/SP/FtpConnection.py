"""
    Steps:
        - Create a new array connection in the end of file
        - Set "current_data" variable with the name of your new array connection
        - Execute in command line: python3 filePath

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
import os

class FtpConnection():
    server = None
    ip = None
    port = None
    user = None
    password = None
    attempts = 0
    actual_dir = ""
    compressed_files = []
    ftp_folder_name = ""
    local_folder = ""
    not_to_delete_ftp = []  # Files on FTP folder that will not be deleted

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

        if not self.ip:
            self.ip = ip
        if not self.port:
            self.port = port
        if not self.user:
            self.user = user
        if not self.password:
            self.password = password
        try:
            if not port:
                self.server = FTP(self.ip)
            else:
                self.server = FTP()
                self.server.connect(host=self.ip, port=self.port)
            self.server.login(user=self.user, passwd=self.password)
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

        if not self.ftp_folder_name:
            self.ftp_folder_name = ftp_folder
        if not self.local_folder:
            self.local_folder = local_folder
        self.server.cwd(self.ftp_folder_name)
        self.actual_dir = self.ftp_folder_name
        filenames = [x for x in self.server.nlst() if self.is_file(x)]  # get filenames within the directory
        print(filenames)

        for filename in filenames:
            print("Downloading: " + filename + " ...")
            local_filename = os.path.join(self.local_folder, filename)
            file = open(local_filename, 'wb')
            self.server.retrbinary('RETR ' + filename, file.write)
            if local_filename[-2:] == "gz":
                self.compressed_files.append(local_filename)
            elif local_filename[-2:] == "ok":
                self.not_to_delete_ftp.append(filename)
            file.close()

        #self.server.quit()

    def is_file(self, filename):
        is_file = True
        try:
            filesize = self.server.size(filename)
            if type(filesize) == type(None):
                is_file = False
        except:
            is_file = False
        return is_file

    def delete_files_from_ftp_folder(self, ftp_folder):
        """
        Delete files from FTP folder
        Args:
            tfp_folder:
        """
        try:
            #if self.local_folder != self.actual_dir:
            #    self.server.cwd(self.local_folder)
            print("Folder status before delete")
            filenames = self.server.nlst()  # get filenames within the directory
            print("Deleting files from FTP folder if any...")
            for filename in filenames:
                if filename not in self.not_to_delete_ftp:
                    self.server.delete(filename)
            print("Actual folder status after delete")
            filenames = self.server.nlst()  # get filenames within the directory
            print(filenames)
        except Exception:
            traceback.print_exc()
            self.attempts += 1
            if self.reset_connection():
                self.delete_files_from_ftp_folder(ftp_folder=self.ftp_folder_name)

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

    def load_json(self):
        # Change with your setting file if necessary
        import json
        with open("info.json") as json_data:
            settings = json.load(json_data)
            self.ip=settings["ftp_name"]
            self.port=settings["port_number"]
            self.user=settings["user_name"]
            self.password=settings["password"]
            self.ftp_folder_name=settings["ftp_folder_name"]
            self.local_folder=settings["local_folder"]
            return settings

def main_manual(ip=None, port=None, user=None, password=None,
         ftp_folder=None, local_folder=None, avoid_set=False, delete_ftp_files=False):

    if avoid_set:
        prism_connection_data = ["prism.nacse.org", None, "anonymous", "email@email.com",
                                 "monthly/ppt/1895/", "Downloads\\"]
        localhost_connection = ["localhost", 9898, "", "", "", "Downloads\\"]
        # TODO Create a new one array connection here:
        new_connection = ["ftp_name",
                          21,
                          "user_name",
                          "password",
                          "ftp_folder_name/",
                          "local_folder"]

        "GZIp --> Unzip: 1: All files will be downloaded, unzipped and, after that, that will be deleted"

        # TODO To change when change connection (in the right, set the name of your array)

        current_data = new_connection

        ip = current_data[0]
        port = current_data[1]
        user = current_data[2]
        password = current_data[3]
        ftp_folder = current_data[4]
        local_folder = current_data[5]
    else:

        ftp_connection = FtpConnection()
        ftp_connection.connect(ip, port, user, password)
        ftp_connection.folder_to_download(ftp_folder=ftp_folder,
                                          local_folder=local_folder)
        if ftp_connection.unzip_files() and delete_ftp_files:
            ftp_connection.delete_files_from_ftp_folder(ftp_folder=ftp_folder)
        if ftp_connection.server:
            ftp_connection.close_connection()

def main_json(delete_ftp_files=False):
    ftp_connection = FtpConnection()
    dicts = ftp_connection.load_json()
    print("dicts", str(dicts))

    ip = dicts["ftp_name"]
    port = dicts["port_number"]
    user = dicts["user_name"]
    password = dicts["password"]
    ftp_folder = dicts["ftp_folder_name"]
    local_folder = dicts["local_folder"]

    main_manual(ip=ip, port=port, user=user, password=password,
         ftp_folder=ftp_folder, local_folder=local_folder, delete_ftp_files=delete_ftp_files)

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
    #main(ip=ip, port=port, user=user, password=password,
    #     ftp_folder=ftp_folder, local_folder=local_folder)
    main_json()
