import json 
from deepface import DeepFace

result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")
print(json.dumps)