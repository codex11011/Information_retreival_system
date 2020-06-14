 
import json 
  
def write_to_json_file(json_object,filename):
    filepath = '../result_benchmark/'+filename+'.json'    
    json_object = json.dumps(json_object, indent = 4) 
    
    with open(filepath, "w") as outfile: 
        outfile.write(json_object)