import json 
from deepface import DeepFace

#face matching
result = DeepFace.verify(img1_path = "database\Sagar_Shukla.jpg", img2_path = "database\Sagar2.jpg")
print(json.dumps(result, indent=2))

#Finding a face in the database 
dfs = DeepFace.find(img_path = "database\Sagar2.jpg", db_path = "database")
print(dfs)

#Face Analysis 
objs = DeepFace.analyze(img_path = "database\Sagar3.jpg", actions = ['age', 'gender', 'race', 'emotion'])
print(json.dumps(objs,indent =2))