
import re
from string import digits
orgReg = ['(',')','（','）','有限','公司']
def filter():
    reader = open('/Users/hecy/workdir/s_title.data','r',encoding = 'utf-8')
    line = reader.readline()
    data = []
    while True:
        line = reader.readline()
        if not line:
            break
        line =line.replace('/','').replace('-','')  #line.replace(" ","").replace('(','').replace(')','').replace('（','').replace('）','').replace('有限','').replace('公司','')
        remove = str.maketrans('','',digits)
        line = line.translate(remove)
        if line not in data:
            data.append(line)
    reader.close()     
    str1 = ''.join(data)
    f = open('/Users/hecy/workdir/s_title.data','w')
    f.write(str1)
    f.close()

label = ['B-PRO', 'I-PRO', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
def combination():
    locReader = open('/Users/hecy/workdir/location.data','r',encoding = 'utf-8')
    orgReader = open('/Users/hecy/workdir/s_name_org.data','r',encoding = 'utf-8')
    proReader = open('/Users/hecy/workdir/s_title.data','r',encoding = 'utf-8')
    loc = []
    org = []
    pro = []
    while True:
        line = locReader.readline()
        if not line:
            break
        loc.append(line.strip())
    while True:
        line = orgReader.readline()
        if not line:
            break
        org.append(line.strip())
    while True:
        line = proReader.readline()
        if not line:
            break
        pro.append(line.strip())
    data = []
    for i in loc:
        f = 0
        temp = []
        for str in i:
            if f == 0:
                temp.append(str+' '+'B-LOC\n')
            else:
                temp.append(str+' '+'I-LOC\n')
            f += 1
        for j in range(137330,137340):
            p = 0
            for str in org[j]:
                if p == 0:
                    temp.append(str+' B-ORG\n')
                else:
                    temp.append(str+' I-ORG\n')
                p += 1
            for k in range(34442,34452):
                q = 0
                data.extend(temp)
                for str in pro[k]:
                    if q == 0:
                        data.append(str+' B-PRO\n')
                    else:
                        data.append(str+' I-PRO\n')
                    q += 1
                data.append('\n')
    f = open('ner/data/test_search.data','w')
    f.write(''.join(data))
    f.close()

combination()
# filter()