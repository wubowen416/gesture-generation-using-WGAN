import socket
import sys
import time
sys.path.append(".")
from tools.Config import JsonConfig


def alert(msg):
    print(msg)
    sys.exit(1)

class DataClient:

    def __init__(self, config=None):

        if not config:
            config = JsonConfig("./tools/commu/config/tcpip.json")
        
        self.server_ip = config.Data.server_ip
        self.port = config.Data.port
        self.encoding = config.Data.encoding
        self._connect_to_server()

    def __del__(self):
        print("Close Data client.")
        self.client.close()
        
    def _connect_to_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        try:
            self.client.connect((self.server_ip, self.port))
            print("Connected to Data server...")
        except:
            alert('Failed to connect Data: ' + str(self.server_ip) + ": " + str(self.port))

    def receive(self, size):
        return self.client.recv(size).decode(self.encoding)


class CommuClient:

    def __init__(self, config=None):

        if not config:
            config = JsonConfig("./tools/commu/config/tcpip.json")
        
        self.server_ip = config.Commu.server_ip
        self.port = config.Commu.port
        self.encoding = config.Commu.encoding
        self.interval = config.Commu.interval
        self._connect_to_server()

    def __del__(self):
        print("Close Commu client.")
        self.client.close()
        
    def _connect_to_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client.connect((self.server_ip, self.port))
            print("Connected to Commu server...")
        except:
            alert('Failed to connect Commu: ' + str(self.server_ip) + ": " + str(self.port))

    def reset_pose(self):
        command = "/movemulti 2 90 10 3 0 10 4 -90 10"
        self.client.send(command.encode(self.encoding))

    def send(self, line: str):
        self.client.send(line.encode(self.encoding))

    def sendall(self, lines: list):
        for line in lines:
            self.send(line)
            print(line)
            time.sleep(self.interval)
