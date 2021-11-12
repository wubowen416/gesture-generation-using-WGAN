import socket
import sys
sys.path.append(".")
from tools.Config import JsonConfig

def alert(msg):
    print(msg)
    sys.exit(1)

class GeneratorClient:

    def __init__(self, config=None):

        if not config:
            config = JsonConfig("./tools/commu/config/tcpip.json")
        
        self.server_ip = config.server_ip
        self.port = config.port
        self.encoding = config.encoding
        self._connect_to_server()

    def __del__(self):
        print("Close client.")
        self.client.close()
        
    def _connect_to_server(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Connect to server...")
        try:
            self.client.connect((self.server_ip, self.port))
        except:
            alert('Failed to connect ' + self.server_ip + ": " + self.port)

    def receive(self, size):
        return self.client.recv(size).decode(self.encoding)
