
from tqdm import tqdm
path = './Data/visda-2017/train/image_list.txt'
file1 = open(path, 'r')
lines = file1.readlines()
file1.close()
k=0
length = len(lines)

for i in tqdm(range(length)):
	line=lines[i].strip()
	if int(line[-2:])>=10:
		line = line[:-3]
	else:
		line = line[:-2]
	try:
		image_path = './Data/visda-2017/train/'+line
		f = open(image_path,'r')
	except:
		print('NOT FOUND!!!')
		print(line)
		k+=1
print(k)