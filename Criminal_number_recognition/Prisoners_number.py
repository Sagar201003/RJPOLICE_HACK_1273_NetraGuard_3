import cv2
import easyocr
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import ImageFont, ImageDraw, Image
import pandas as pd


reader = easyocr.Reader(['en'])
cap = cv2.VideoCapture('Kaidi602.mp4')

numbers = []

while True:
    ret, frame = cap.read()

    results = reader.readtext(frame)
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(results)

    # Display the frame with the results
    cv2.imshow('Prison Number', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
