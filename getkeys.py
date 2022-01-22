import win32api as wapi
import time


keyList = [0x26,0x28,0x54]
# for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
#     keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(key):
            keys.append(key)
    return keys