# -*- coding: utf-8 -*-

"""@package docstring
Documentation for this module.
 
This package is client code.
"""

import sys
import signal
from grpc_wrapper.client import create_client


def signal_handler(sig, frame):
    """Documentation for a function.
 
    This function handles a exit event.
    """
    print('terminating...')
    #client.send({"roomState": 1, "roomId": 999, "sentence": "null"})
    sys.exit(0)


def run():
    """Documentation for a function.
 
    This function makes connection with server.
    """
    roomState = 0  # 0: open, 1: close, 2: chat
    roomId = 999
    sentence = "null"  # is used with roomState value 2

    # init
    client.send({"roomState": 0, "roomId": roomId, "sentence": sentence})

    while(True):
        sys.stdout.write('>>')
        sys.stdout.flush()
        sentence = sys.stdin.readline().strip()
        roomState = 2
        result = client.send({"roomState": roomState, "roomId": roomId, "sentence": sentence})
        print('response : ', str(result))


if __name__ == '__main__':
    """Documentation for a function.
 
    This main function starts client program with given both IP address and port number about server.
    """
    signal.signal(signal.SIGINT, signal_handler)
    client = create_client("127.0.0.1", 50051)
    run()
