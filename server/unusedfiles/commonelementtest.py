from PIL import Image,ImageShow
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
PATH= os.path.dirname(__file__)



def joinpath(folder):
    return os.path.join(PATH,folder)

matrices=[]
def getmatriximg(*args):
    for a in args:
        file=os.path.join(PATH,a)
        img=Image.open(file)
        matrices.append(preprocess(np.asarray(img)))

def preprocess(numpyimg):
    grey=(cv2.cvtColor(numpyimg, cv2.COLOR_BGR2GRAY))
    blur= cv2.GaussianBlur(grey,(3,3), 0)
    ret, thresh = cv2.threshold(blur,3, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    struct=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilate=cv2.dilate(thresh,struct,iterations=4)
    
    
matrix1="storePDF\SSRN-id3052318\page5.jpg"
matrix2="storePDF\SSRN-id3052318\page6.jpg"
matrix3="storePDF\SSRN-id3052318\page7.jpg"

getmatriximg(matrix1,matrix2,matrix3)
# Assuming you have a list of matrices
#matrices = [matrix1, matrix2, matrix3]

#print(matrices[1])
# Normalize the matrices
normalized_matrices = []
max_value = np.max(matrices)
min_value = np.min(matrices)


#plt.show()

for matrix in matrices:
    normalized_array = (matrix - min_value) / (max_value - min_value)
    normalized_matrices.append(normalized_array)
    
#plt.imshow(Image.fromarray(normalized_matrices[0],'RGB'))
#plt.show()
# Determine the canvas size based on the matrices' dimensions
num_rows = len(normalized_matrices[0])
num_cols = len(normalized_matrices[0][0])

canvas = Image.new('RGB', (num_cols, num_rows), color='black')

# Layer the matrices onto the canvas
for matrix in normalized_matrices:
    
    for i in range(num_rows):
        for j in range(num_cols):
            pixel_value = int(matrix[i][j] * 255)  # Scale the value to the range 0-255
            current_pixel = canvas.getpixel((j, i))
            new_pixel = tuple([max(pixel_value[i],[j], current_pixel[k]) for k in range(3)])  # Layer the values
            canvas.putpixel((j, i), new_pixel)

# Display the resulting composite image
canvas.show()




def fitzfoothead(self):



        name=joinpath(os.path.join('testPDF','s44155-023-00042-4.pdf'))

        doc = fitz.open(name)
        num_sections = 10
        counts = [0] * num_sections

        for page_num in range(doc.page_count):
            page = doc[page_num]

            text_dict = page.get_text('dict')

            blocks = text_dict['blocks']

            for block in blocks:
                x, y, w, h = block['bbox']

                section = int(y / page.rect.height * num_sections)

                counts[section] += 1

        counts = np.array(counts)

        header_section = np.argmin(counts[:num_sections//2])
        footer_section = np.argmin(counts[num_sections//2:]) + num_sections//2

        header_threshold = header_section / num_sections * page.rect.height
        footer_threshold = footer_section / num_sections * page.rect.height

        #print(doc.embfile_info())

        #print('Header threshold:', header_threshold)
        #print('Footer threshold:', footer_threshold)
        return header_threshold,footer_threshold
    
