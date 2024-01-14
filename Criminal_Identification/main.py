import json 
from deepface import DeepFace

result = DeepFace.verify(img1_path = "Sagar Shukla.jpg", img2_path = "Sagar3.jpg")
print(json.dumps(result,))