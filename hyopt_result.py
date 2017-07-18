
import os
import sys
import glob
import json

VERB=True
if len(sys.argv)>1:
	if sys.argv[1]=="no-verb":
		VERB=False

data=[]
for filename in glob.glob("hyopt/hyparam*.result.json"):
	fp = open(filename, 'r')
	obj=json.load(fp)
	res=obj["evaluation"]
	val=res["validation_cost"]
	data.append((filename,val))
	#"evaluation_output": "hyopt/hyparam00029.result.json",
data=sorted(data,key=lambda x: x[1])
print("## Ranking")
for el in data:
	print(el[0],el[1])
print("## Top:",data[0][0])
