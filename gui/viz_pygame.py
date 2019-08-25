# coding: utf-8
# 
import numpy as np 
import json
from socket import *
import select
import pygame
from pygame.locals import *
import sys


HOST = gethostname()
PORT = 1113
BUFSIZE = 1024
ADDR = ("127.0.0.1", PORT)
USER = 'Server'
INTERVAL=0.01
VEROCITY=100
LIFETIME=1000
#Window.fullscreen=True

class DataReceiver:
	def init(self):
		self.label_data={}
		self.t={}
		self.udpServSock = socket(AF_INET, SOCK_DGRAM) #IPv4/UDP
		self.udpServSock.bind(ADDR)
		self.udpServSock.setblocking(0)
		enabled_labels = [0,1,2]
		labels = {"0":"aaa","1":"bbb","2":"ccc"}
		
	def update(self):
		data=None
		draw_data={}
		try:
			while True:
				data, addr = self.udpServSock.recvfrom(BUFSIZE)
				if data is not None and len(data)>0:
					print('...received from and returned to:', addr)
					print(len(data))
					while len(data)>=6:
						idx=int.from_bytes(data[0:1],byteorder='little')
						l=int.from_bytes(data[1:2],byteorder='little')
						arr=np.frombuffer(data[2:6],dtype=np.float32)
						data=data[6:]
						y=arr[0]
						draw_data[l]=y
						print(idx,l,y)
		except:
			pass

		if len(draw_data)>0:
			for l,y in draw_data.items():
				# set label color
				label=int(l)
				if label not in self.label_data:
					self.label_data[label]=np.zeros((100,))
					self.label_data[label][:]=np.nan
					self.t[label]=0
				self.label_data[label][self.t[l]]=y
		for l,v in self.label_data.items():
			self.t[l]+=1
			self.t[l]=self.t[l]%100
			self.label_data[l][self.t[l]]=np.nan
	def stop(self):
		self.udpServSock.close()

color_list=[(255,0,0),(0,255,0),(0,0,255),]
def main():
	dr=DataReceiver()
	pygame.init() # 初期化
	screen = pygame.display.set_mode((600, 400))
	pygame.display.set_caption("Pygame Test")
	dr.init()
	t=0
	dt=10
	while(True):
		cnt=0
		for event in pygame.event.get(): # 終了処理

				if event.type == QUIT:
					pygame.quit()
					sys.exit()
		dr.update()
		screen.fill((0,0,0,)) # 背景色の指定。RGBだと思う
		for k,v in dr.label_data.items():
			cur=dr.t[k]
			prev=np.nan
			for i in range(len(v)):
				y=v[(cur-i)%len(v)]
				#if (not np.isnan(y)) and (not np.isnan(prev)):
				#	pygame.draw.line(screen, (255,255,255), (i*dt,prev), ((i+1)*dt,y),5)
				if (not np.isnan(y)):
					pygame.draw.circle(screen, color_list[k], (i*dt,int(y)), 3)
					cnt+=1
				prev=y
		pygame.display.update() # 画面更新
		pygame.time.wait(30) # 更新間隔。多分ミリ秒
		t+=1
if __name__ == "__main__":
    main()
