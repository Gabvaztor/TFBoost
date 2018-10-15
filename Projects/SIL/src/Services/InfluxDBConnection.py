import argparse
from influxdb import InfluxDBClient
from Projects.SIL.src.UsefulTools.UtilsDecorators import *

@for_all_methods(exception(logger=logger()))
class InfluxDB():
    """

    """
    # -*- coding: utf-8 -*-
    """Tutorial on using the InfluxDB client."""

    def main(self, host='localhost', port=3000):
        """Instantiate a connection to the InfluxDB."""
        # admin
        # admin@localhost
        host = "192.168.1.220"
        port = 8086
        user = 'gabriel'
        user = 'gabriel.vazquez@smartiotlabs.com'
        user = ''
        password = 'fq1MvWvJFyREZI1wq1xyXMClrPhf7W'
        password = ''
        dbname = 'miranda'
        dbuser = 'smly'
        dbuser_password = 'my_secret_password'
        query = 'select top 100 value from mysensors where device=1 and sensor=4'
        api_key = "eyJrIjoiT2R5ZkJ5V0tpMEVYMHF5SWJCTWZyaHNDVldLajB5eW8iLCJuIjoic21hcnRpb3RsYWJzIiwiaWQiOjF9"
        http_header = 'curl -H "Authorization: Bearer ' \
                      'eyJrIjoiT2R5ZkJ5V0tpMEVYMHF5SWJCTWZyaHNDVldLajB5eW8iLCJuIjoic21hcnRpb3RsYWJzIiwiaWQiOjF9" ' \
                      'http://192.168.1.220:3000/api/dashboards/home'


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

        db_client = InfluxDBClient(host, port, user, password, dbname)
        #db_client = InfluxDBClient(host, port)
        #db_client = InfluxDBClient(host, port)

        print("Trying ping to server...")
        print(db_client.ping())
        print("Ping done!")

        print("Switch user: " + user)
        db_client.switch_user(user, password)

        print("Listing databases")
        print(str(db_client.get_list_database()))

        print("Querying data: " + query)
        result = db_client.query(query)

        print("Result: {0}".format(result))
        """
        print("Create database: " + dbname)
        db_client.create_database(dbname)

        print("Create a retention policy")
        db_client.create_retention_policy('awesome_policy', '3d', 3, default=True)

        print("Switch user: " + dbuser)
        db_client.switch_user(dbuser, dbuser_password)

        print("Write points: {0}".format(json_body))
        db_client.write_points(json_body)

        print("Switch user: " + user)
        db_client.switch_user(user, password)

        print("Drop database: " + dbname)
        db_client.drop_database(dbname)
        """

if __name__ == '__main__':
    influxDB = InfluxDB()
    influxDB.main()