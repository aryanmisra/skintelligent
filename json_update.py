import json
import os
json_path = '../json/RS_Clean-images/Split_updated/'
img_dir = '../images/'

def fileList(source): #creating a list of all json files in directory
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            if filename.endswith(('.json')):
                matches.append(os.path.join(root, filename))
    return matches

def updateJsonFile(json_file):
    jsonFile = open(json_file, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Working with buffered content
    tmp = data["filename"] 
    path = "RS_Clean-images/Split_updated-naming-convention/" + tmp
    data["filename"] = path

    ## Save our changes to JSON file
    jsonFile = open(json_file, "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()

jsons = fileList(json_path)
print(len(jsons))
for file in jsons:
    updateJsonFile(file)
    print(file)
