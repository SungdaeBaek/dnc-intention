# -*- coding: utf-8 -*-
import sys
import signal
from grpc_wrapper.client import create_client


def signal_handler(sig, frame):
    print('terminating...')
    #client.send({"roomState": 1, "roomId": 999, "sentence": "null"})
    sys.exit(0)


def run():
    #roomState = 0  # 0: open, 1: close, 2: chat
    #roomId = 999
    sentence = "null"  # is used with roomState value 2

    # init
    #client.send({"roomState": 0, "roomId": roomId, "sentence": sentence})
    client.send({"sentence": sentence})

    while(True):
        sys.stdout.write('>>')
        sys.stdout.flush()
        sentence = sys.stdin.readline().strip()
        #roomState = 2
        #result = client.send({"roomState": roomState, "roomId": roomId, "sentence": sentence})
        result = client.send({"sentence": sentence})
        print('response : ', str(result))


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    client = create_client("localhost", 50069)
    run()
