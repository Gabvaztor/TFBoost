import sqlite3
from sqlite3 import Error
from Projects.SIL.src.UsefulTools.UtilsDecorators import *

@for_all_methods(exception(logger=logger()))
class SqlLite():
    connection = None
    def create_connection(self, db_file):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)

        return None

    def select_all_tasks(self, conn):
        """
        Query all rows in the tasks table
        :param conn: the Connection object
        :return:
        """
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks")

        rows = cur.fetchall()

        for row in rows:
            print(row)

    def select_task_by_priority(self, conn, priority):
        """
        Query tasks by priority
        :param conn: the Connection object
        :param priority:
        :return:
        """
        cur = conn.cursor()
        cur.execute("SELECT * FROM tasks WHERE priority=?", (priority,))

        rows = cur.fetchall()

        for row in rows:
            print(row)

    def insert(self, sql):
        # Get cursor
        cursor = self.connection
        # Insert a row of data
        cursor.execute(sql)

        # Save (commit) the changes
        self.connection.commit()

    def close(self):
        if self.connection:
            self.connection.close()

    def connect(self, database_dir="C:\Software\SqlLite\SIL.db"):
        database = database_dir

        # create a database connection
        self.connection = self.create_connection(database)
        with self.connection:
            print("1. Query task by priority:")
            select_task_by_priority(conn, 1)

            print("2. Query all tasks")
            select_all_tasks(conn)
