from tqdm import tqdm
import random
# path = './Data/visda-2017/train/image_list.txt'
path = './Data/image_list.txt'
file1 = open(path, 'r')
lines = file1.readlines()
file1.close()
k=0
length = len(lines)
dict_domain = {}
print(length)
for i in tqdm(range(length)):
	line=lines[i].strip()
	if int(line[-2:])>=10:
		line = line[:-3]
	else:
		line = line[:-2]
	names = line.split('/')
	if dict_domain.get(names[0]) == None:
		dict_domain[names[0]]=[i]
	else:
		dict_domain[names[0]].append(i)
# path_new='./Data/visda-2017/train/image_new_list.txt'
path_new='./Data/image_new_list.txt'
file2 = open(path_new, 'w')
print(dict_domain.keys())
for k,v in dict_domain.items():
	index = random.choices(v, k=50)
	for j in tqdm(range(len(index))):
		file2.write(lines[index[j]])

file2.close()

