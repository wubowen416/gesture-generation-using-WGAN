import socket
import sys
import time
import numpy as np
import socket
import time
sys.path.append(".")
from tools.Config import JsonConfig
import wave
from threading import Thread


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
        except Exception as e:
            print(e)
            alert('Failed to connect Commu: ' + str(self.server_ip) + ": " + str(self.port))

    def reset_pose(self):
        command = "/movemulti 1 0 1 2 90 10 3 0 10 4 -90 10 5 0 10 0 0 1 7 0 1 8 0 1\n"
        self.client.send(command.encode(self.encoding))

    def send(self, line: str):
        print(line, end="")
        self.client.send(line.encode(self.encoding))

    def sendall(self, lines: list):
        for line in lines:
            self.send(line)
            time.sleep(self.interval)


class WavClient(Thread):
    def __init__(self):
        Thread.__init__(self)
        config = JsonConfig("./tools/commu/config/tcpip.json")
        self.host = config.Wav.server_ip
        self.port = config.Wav.port
        self.interval = config.Wav.interval
        self.sock = self._connect_to_server()

        self.header = -1
        self.volume = 1
        self.delay = 0
        self.interval = 10.0 / 1000.0

    def __del__(self):
        print("Close Wav client.")
        self.sock.close()
        self.wav_file.close()

    def _connect_to_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
            print("Connected to Wav server...")
        except Exception as e:
            print(e)
            alert('Failed to connect Wav: ' + str(self.host) + ": " + str(self.port))
        return sock

    def load_wavfile(self, path):
        self.wav_file = wave.open(path, 'rb')
        # オーディオプロパティ
        self.CHANNELS = self.wav_file.getnchannels()
        SAMPLE_WIDTH = self.wav_file.getsampwidth()
        self.RATE = self.wav_file.getframerate()
        FRAMES = self.wav_file.getnframes()
        self.CHUNK = 160
        print("Channel num : ", self.CHANNELS)
        print("Sample width : ", SAMPLE_WIDTH)
        print("Sampling rate : ", self.RATE)
        print("Frame num : ", FRAMES)
        self.wav_file.readframes(self.RATE*self.delay)

    def run(self):
        base_time = time.time()
        while True:
            wav_data = self.wav_file.readframes(self.CHUNK)

            if wav_data == b'':
                break

            self.sock.send(self.mix_sound(wav_data, self.CHANNELS, self.CHUNK, 1))

            next_time = ((base_time - time.time()) % self.interval) or self.interval
            time.sleep(next_time)

    def mix_sound(self, data, channels, frames_per_buffer, volume):
        # デコード
        decoded_data = np.frombuffer(data, np.int16).copy()
        # データサイズの不足分を0埋め
        decoded_data.resize(channels * frames_per_buffer, refcheck=False)
        #音量調整 & エンコード
        self.header = self.header + 1
        return np.array([self.header]).astype(np.int32).tobytes() + (decoded_data * volume).astype('<i2').tobytes()
