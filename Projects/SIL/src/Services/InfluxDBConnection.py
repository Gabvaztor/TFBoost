import argparse
from influxdb import InfluxDBClient
from Projects.SIL.src.UsefulTools.UtilsDecorators import *

@for_all_methods(exception(logger=logger()))
class InfluxDB():
    """

    """
    # -*- coding: utf-8 -*-
    """Tutorial on using the InfluxDB client."""

    def main(host='192.168.1.200', port=8086):
        """Instantiate a connection to the InfluxDB."""
        # admin
        # admin@localhost

        user = 'admin'
        password = 'gabriel'
        dbname = 'miranda_db'
        dbuser = 'smly'
        dbuser_password = 'my_secret_password'
        query = 'select * from ;'
        json_body = [
            {
                "measurement": "cpu_load_short",
                "tags": {
                    "host": "server01",
                    "region": "us-west"
                },
                "time": "2009-11-10T23:00:00Z",
                "fields": {
                    "Float_value": 0.64,
                    "Int_value": 3,
                    "String_value": "Text",
                    "Bool_value": True
                }
            }
        ]

        client = InfluxDBClient(host, port, user, password, dbname)

        str(client.get_list_database())

        print("Querying data: " + query)
        result = client.query(query)

        print("Result: {0}".format(result))
        """
        print("Create database: " + dbname)
        client.create_database(dbname)

        print("Create a retention policy")
        client.create_retention_policy('awesome_policy', '3d', 3, default=True)

        print("Switch user: " + dbuser)
        client.switch_user(dbuser, dbuser_password)

        print("Write points: {0}".format(json_body))
        client.write_points(json_body)

        print("Switch user: " + user)
        client.switch_user(user, password)

        print("Drop database: " + dbname)
        client.drop_database(dbname)
        """

if __name__ == '__main__':
    influxDB = InfluxDB()
    influxDB.main()