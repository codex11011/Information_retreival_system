import json 
from collections import defaultdict
import math  

filename = './relevance_judgement.txt'
dict1 = defaultdict(dict)
  
with open(filename) as fh: 
  
    for line in fh: 
        command, description = line.strip().split(None, 1) 
        if command in dict1:    
            dict1[command]['related_document'].append(description.strip()) 
        else:
            dict1[command] = {
                'category':math.floor(int(description.strip())/100),
                'related_document':[description.strip()]   
            }
        

out_file = open("relevance_judgement.json", "w") 
json.dump(dict1, out_file, indent = 4, sort_keys = False) 
out_file.close()
