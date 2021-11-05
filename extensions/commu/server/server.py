# -*- coding : UTF-8 -*-

# 0.ライブラリのインポートと変数定義
import socket

server_ip = "localhost"
server_port = 8081
listen_num = 1
# buffer_size = 1024

# 1.ソケットオブジェクトの作成
tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2.作成したソケットオブジェクトにIPアドレスとポートを紐づける
tcp_server.bind((server_ip, server_port))

# 3.作成したオブジェクトを接続可能状態にする
tcp_server.listen(listen_num)

# load data
with open('audio1109-rms.txt') as f:
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

    client.close()
