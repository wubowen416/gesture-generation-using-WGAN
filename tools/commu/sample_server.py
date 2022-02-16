# -*- coding : UTF-8 -*-

# 0.ライブラリのインポートと変数定義
import socket
import sys
import argparse
sys.path.append(".")
from tools.Config import JsonConfig

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="tools/commu/sample_data/20210317-095527-729-rms.txt")
args = parser.parse_args()

config = JsonConfig("./tools/commu/config/tcpip.json")
listen_num = 1
# buffer_size = 1024

# 1.ソケットオブジェクトの作成
tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2.作成したソケットオブジェクトにIPアドレスとポートを紐づける
tcp_server.bind((config.Data.server_ip, config.Data.port))

# 3.作成したオブジェクトを接続可能状態にする
tcp_server.listen(listen_num)

# load data
with open(args.data_path) as f:
    lines = f.readlines()

# 4.ループして接続を待ち続ける
while True:
    # 5.クライアントと接続する
    print("Wait for client...")
    client, address = tcp_server.accept()
    print("[*] Connected!! [ Source : {}]".format(address))

    print("Send data...")

    frame_number = list(reversed([str(i) for i in range(len(lines))]))

    for i, line in enumerate(lines[1:]):

        line = frame_number[i] + "\t" + line
        client.send(line.encode("utf-8"))

        # print(line)

    client.send("0\n".encode("utf-8"))
    
    # break
    client.close()
    
