import re
import fitz  
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image,ImageShow
SIMILARITY = 0.5
PATH= os.path.dirname(__file__)
print(PATH)


def joinpath(folder):
    return os.path.join(PATH,folder)


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    doc.close()
    return text




def extract_table_data(pdf_path):
    doc = fitz.open(pdf_path)
    table_data = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        table_blocks = page.get_text("blocks", clip=page.rect)

        for block in table_blocks:
            if block[4] == 0:  # Check if it's a horizontal text line (potential table row)
                row_data = []
                for line in block[5]:
                    row_data.append(line[4])
                table_data.append(row_data)

    doc.close()
    return table_data










def cutoffHeight(self):
        page=self.pages[1:-1,:,:] 
        header_height = 0
        max_similarity=SIMILARITY
        similarity=1
        cut=page.shape[0]/0.1
        while similarity > max_similarity:
            meansim=[]
            for i in range(cut,page.shape[0],np.math.floor((page.shape[0]-cut))):
                
                #similarity = (page[1,header_height,:]==page[i,header_height,:]).all(axis=(0,1)).mean()
                
                similarity =1- np.mean((page[0,header_height,:] - page[i,header_height,:])**2)
                meansim.append(similarity)
            

            if np.asarray(meansim).mean() < max_similarity:
                break
            header_height+=1
        self.header=header_height-5
    
 
def cutoffFooter(self):
    page=self.pages[1:-1,:,:] #case of covers
    page_height=page.shape[1]-1
    footer_height = 0
    max_similarity=SIMILARITY
    similarity=1
    cut=page.shape[0]/0.1
    
    while similarity > max_similarity:
        meansim=[]
        for i in range(cut,page.shape[0]-cut,np.math.floor((page.shape[0]-cut))):
            
            #similarity = (page[1,(page_height-footer_height),:]==page[i,(page_height-footer_height),:]).all(axis=(0,1)).mean()
            matrix=np.mean((page[1,page_height-footer_height,:] - page[i,page_height-footer_height,:])**2)
            similarity =1- matrix
            
            meansim.append(similarity)
        

        if np.asarray(meansim).mean() < max_similarity:
            break
        footer_height+=1
    self.footer=footer_height-5


import fitz
import numpy as np
import os
from pathlib import Path

def fitzheader():
    name = joinpath(os.path.join('testPDF','s44155-023-00042-4.pdf'))
    doc = fitz.open(name)
    num_sections = 10
    counts = [0] * num_sections
    hashes = []
    unique_sections = {}
    sections = []

    for page_num in range(doc.page_count):
        page = doc[page_num]

        text_dict = page.get_text('dict')

        blocks = text_dict['blocks']

        for block in blocks:
            x, y, w, h = block['bbox']

            section = int(y / page.rect.height * num_sections)

            text = block['text']
            hash_value = hash(text)

            if hash_value not in unique_sections:
                unique_sections[hash_value] = text
                sections.append(hash_value)

            counts[section] += 1

    counts = np.array(counts)

    footer_sections = []

    for section in sections:
        section_counts = [counts[i] for i in range(len(counts)) if hash(unique_sections[section]) in hashes[i]]
        section_percentage = (sum(section_counts) / len(section_counts)) / doc.page_count * 100

        if section_percentage > 40:
            footer_sections.append(section)

    header_section = np.argmin(counts[:num_sections//2])
    footer_section = np.argmin(counts[num_sections//2:]) + num_sections//2

    header_threshold = header_section / num_sections * page.rect.height
    footer_threshold = footer_section / num_sections * page.rect.height

    print('Header threshold:', header_threshold)
    print('Footer threshold:', footer_threshold)

    footer_sections_text = '\n'.join([unique_sections[section] for section in footer_sections])
    print('Footer sections:', footer_sections_text)



def fitzheader1():
    name=joinpath(os.path.join('testPDF','Generatedpdfpopulated.pdf'))
    doc = fitz.open(name)
    num_sections = 10
    counts = [0] * num_sections

    for page_num in range(doc.page_count):
        page = doc[page_num]

        text_dict = page.get_text('dict')

        blocks = text_dict['blocks']

        for block in blocks:
            x, y, w, h = block['bbox']
            #section = int(y / page.rect.height * num_sections)

            counts[block] += 1
    

    counts = np.array(counts)   





def fitzheader2():
    name=joinpath(os.path.join('testPDF','Generatedpdfpopulated.pdf'))
    # Open the PDF file
    doc = fitz.open(name)

    # Find repeated headers and footers
    repeated_headers = {}
    repeated_footers = {}

    for page in doc:
        # Extract the text from the page
        text = page.get_text("text")

        # Extract the header and footer text using regular expressions
        header = re.search(r"(?<=\n\n).*(?=\n\n)", text)
        footer = re.search(r"(?<=\n\n\n).*(?=\n\n\n)", text)

        if header:
            header_text = header.group().strip()
            if header_text in repeated_headers:
                repeated_headers[header_text] += 1
            else:
                repeated_headers[header_text] = 1

        if footer:
            footer_text = footer.group().strip()
            if footer_text in repeated_footers:
                repeated_footers[footer_text] += 1
            else:
                repeated_footers[footer_text] = 1

    # Filter the repeated headers and footers based on the number of occurrences
    repeated_headers = {header: count for header, count in repeated_headers.items() if count > 1}
    repeated_footers = {footer: count for footer, count in repeated_footers.items() if count > 1}

    return repeated_headers, repeated_footers
    #header_section = np.argmin(counts[:num_sections//2])
    #footer_section = np.argmin(counts[num_sections//2:]) + num_sections//2

    #header_threshold = header_section / num_sections * page.rect.height
    #footer_threshold = footer_section / num_sections * page.rect.height


    #print('Header threshold:', header_threshold)
    #print('Footer threshold:', footer_threshold)


    #doc.close()
    #print(doc.get_toc())
    #print(text_dict)
    #plt.imshow(Image.fromarray(doc,'RGB'))
    #plt.show()
    #print("break")


if __name__ == "__main__":
    print(extract_text_from_pdf( joinpath(os.path.join('testPDF','Generatedpdfpopulated.pdf'))))