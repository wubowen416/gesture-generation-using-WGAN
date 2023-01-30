import numpy as np
import wave
# import pyaudio
import socket
import threading
import time

#https://qiita.com/tokoroten-lab/items/f82babc96e05a80b810a

class MixedSoundStreamClient(threading.Thread):
    def __init__(self, server_host, server_port, wav_filenames, delay=0):
        threading.Thread.__init__(self)
        self.SERVER_HOST = server_host
        self.SERVER_PORT = int(server_port)
        self.WAV_FILENAMES = wav_filenames
        self.interval = 10.0 / 1000.0
        self.volume = 1
        self.delay = delay
        self.header = -1

    def run(self):
        # audio = pyaudio.PyAudio()

        wav_number = 0
        # 音楽ファイル読み込み
        wav_file = wave.open(self.WAV_FILENAMES[wav_number], 'rb')
        wav_number += 1

        # オーディオプロパティ
        # FORMAT = pyaudio.paInt16
        CHANNELS = wav_file.getnchannels()
        SAMPLE_WIDTH = wav_file.getsampwidth()
        RATE = wav_file.getframerate()
        FRAMES = wav_file.getnframes()
        CHUNK = 160
        print("Channel num : ", CHANNELS )
        print("Sample width : ", SAMPLE_WIDTH)
        print("Sampling rate : ", RATE)
        print("Frame num : ", FRAMES)
        # サーバーに接続
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.SERVER_HOST, self.SERVER_PORT))

            base_time = time.time()
            wav_file.readframes(self.delay*RATE)
            
            # メインループ
            while True:
                # 音楽ファイルとマイクからデータ読み込み
                wav_data = wav_file.readframes(CHUNK)
                if not len(wav_data) == 2 * CHUNK:
                    if not wav_number == len(self.WAV_FILENAMES):
                        wav_file = wave.open(self.WAV_FILENAMES[wav_number], 'rb')
                        wav_data += wav_file.readframes(int(CHUNK - len(wav_data)/2))
                        wav_number += 1
                    else:
                        break

                # サーバに音データを送信
                sock.send(self.mix_sound(wav_data, CHANNELS, CHUNK, 1))

                next_time = ((base_time - time.time()) % self.interval) or self.interval
                time.sleep(next_time)
                #print("FPS: ", 1.0 / (time.time() - hoge))

        # audio.terminate()

    def mix_sound(self, data, channels, frames_per_buffer, volume):
        # デコード
        decoded_data = np.frombuffer(data, np.int16).copy()
        # データサイズの不足分を0埋め
        decoded_data.resize(channels * frames_per_buffer, refcheck=False)
        #音量調整 & エンコード
        self.header = self.header + 1
        return np.array([self.header]).astype(np.int32).tobytes() + (decoded_data * volume).astype('<i2').tobytes()


if __name__ == '__main__':
    ip_address = "172.27.174.94"
    port = 4002

    mss_client = MixedSoundStreamClient(ip_address, port, ["data/takekuchi/source/split/dev/inputs_16k/audio1129.wav"])
    # mss_client = MixedSoundStreamClient(ip_address, port, ["20171226-164500-000-c03.wav"])
    mss_client.start()