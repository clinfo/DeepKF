# coding: utf-8
import numpy as np 
from socket import *
import time

HOST = gethostname()
PORT = 34512
BUFSIZE = 1024
#ADDR = (gethostbyname(HOST), PORT)
ADDR = ("127.0.0.1", PORT)
USER = 'Client'
 
udpClntSock = socket(AF_INET, SOCK_DGRAM)
arr=np.array([0,2],dtype=np.float32)
#data=np.getbuffer(arr)
#data = raw_input('%s > ' % USER) # 標準入力からのデータ入力
for t in range(1000):
	#arr[1]=np.sin(t*0.03)*100+200
	if (t//100)%2==0:
		arr[1]=0
	else:
		arr[1]=10
	#arr[1]=np.sin(t*0.03)*100+200
	print(arr)
	data=memoryview(arr)
	tt=t%0x100
	udpClntSock.sendto((tt).to_bytes(1,byteorder='little')+data, ADDR) # データ送信
	time.sleep(0.03)
udpClntSock.close()
