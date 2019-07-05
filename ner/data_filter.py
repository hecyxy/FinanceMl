
import re
def filter():
    reader = open('ner/data/org_name.data','r',encoding = 'utf-8')
    line = reader.readline()
    data = []
    while True:
        line = reader.readline()
        if not line:
            break
        if('virtual_' not in line and 'deprecated' not in line):
            data.append(line.replace(" ",""))
    reader.close()     
    str = ''.join(data)
    f = open('ner/data/name.data','w')
    f.write(str)
    f.close()

filter()
    


