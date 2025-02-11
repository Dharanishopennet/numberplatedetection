import cv2#library for the number plate detection
import matplotlib.pyplot as plt #library for the marking around the number plate
import pytesseract#library for the convertion of the text 
import numpy as np#increases the accuracy of the text
#OCR path 
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\vkdha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
#cascade that using the pretrained code for the number plate detection
#This cascade classifier for the ML for number plate detection 
#giving the path to the opencv
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
#function to increases the accuracy
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)# convert to the gray scale
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)#enlarge the image 2x2 for clarity
    gray = cv2.medianBlur(gray, 3)# remove the noise 
    #adjust the light condition for clarity
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    #remove the small gaps and holes
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed
#function for the process
def detect_number_plate(image_path):
    img = cv2.imread(image_path)#it will read the image file
    #it will convert the image to grayscale to increase the speed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     #it is used to detect the object in the image
    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)
     #for draw rectangle x,y top left corner
    #w,h is width and height
    #and to convert the text
    for (x, y, w, h) in plates:
        plate_img = img[y:y+h, x:x+w]
        #passing the number plate to make accuracy
        processed_plate = preprocess_plate(plate_img)
        cv2.imwrite("processed_plate.jpg", processed_plate)
        #converting the text of the new accuracy image
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(processed_plate, config=custom_config).strip()
        #draw rectangel in the image //bottom right corner // color // thickness
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#convert to the RGB
        plt.imshow(img_rgb)#show image with plot
        plt.axis('off')
        plt.show()#show image
        return text  

    return "No number plate detected"

image_path = 'download1.jpg'# giving the path of the image 
detected_text = detect_number_plate(image_path) #passing the image to the function
print(f"Detected Number Plate: {detected_text}")#print the text 
