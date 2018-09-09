from math import ceil
l=list(range(400))

feed = list()
size=len(l)

if size>20:
	cycles = ceil(size/20)
	count=0
	while cycles>count:
		feed.append(l[count*20:(count+1)*20])
		count+=1
		
	
else:
	feed.append(l)
	
for i in feed:
	print(i)