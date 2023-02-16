import re
from pymongo import MongoClient

a_file = open("sample_out.txt",encoding='utf8')

file = open("sample_out.txt",encoding='utf8')

#inputstring = 'wfowefnowefn "Hello" ewofnewofnewofnewofn "Hello hi how are you"'
Counter = 0
  
# Reading from file
Content = file.read()
Content = str(Content)
CoList = Content.split("\n")
  
for i in CoList:
    if i:
        Counter += 1
          
#print("This is the number of lines in the file")
#print(Counter)
for i in range(Counter):
    line = a_file.readline()

    test_str = str(line)
    #print(test_str)
    #print(type(re.findall('"([^"]*)"', test_str)[1:]))

    client = MongoClient('localhost', 27017)
    db = client['IR_Project']
    
    collection = db['IR_Project_Key']
    #print(collection)
    d = re.findall('"([^"]*)"', test_str)[1:]
    res_dict = {}
    res_dict = {re.findall('"([^"]*)"', test_str)[0]: d for i in range(0, len(d))}
    print(res_dict)
    
	
    collection.insert_one(res_dict)
