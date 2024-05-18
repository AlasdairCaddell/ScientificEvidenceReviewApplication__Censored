
import csv
import pickle
import platform
import random
import shutil
import statistics
import subprocess
import threading
import webbrowser
import numpy as np
import os
import fitz  
import numpy as np
import os
import PIL
from PIL import Image
import cv2
import regex as re
import pdf2image as p2i
#from server.classify import datacontroller
from visualise import scroll,learnfrom,learnfromPAGE
from tkinterscrollable import ScrollableImage
import pytesseract  

from matplotlib import pyplot as plt
from visualise import *
from classify import *
import pytesseract

#debugcommands
def joinpath(folder):
    return os.path.join(PATH,folder)

def store_filepath_or_input():
    try:
        filepath=joinpath('tesseractpath.txt')
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        else:    
            with open(filepath, 'r') as file:
                stored_path = file.read().strip()
                if stored_path!='' and os.path.exists(stored_path):
                    return stored_path
                
        return None
    except Exception as e:
        print(e)
        return None
    
def get_filepath(new_path):    
    filepath=joinpath('tesseractpath.txt') 
    with open(filepath, 'w') as file:
        file.write(new_path)
    return new_path
    

PATH= os.path.dirname(__file__)
pytesseract.pytesseract.tesseract_cmd = r'C:\\User\\Users\\Envs\\proj10\\Lib\\site-packages\\tesseract.exe'
pytesseract_lock = threading.Lock()

#For any public deployment convert to private for security
 
    
def add_to_dict(dictionary, key, value):
    if key in dictionary:
        if isinstance(dictionary[key], list):
            dictionary[key].append(value)
        else:
            dictionary[key] = [dictionary[key], value]
    else:
        dictionary[key] = value


class Savepickleobject():
    def __init__(self,documentobject,ifboundingbox) :
        self.object=documentobject
        #self.ifdisplayimgfortraining=ifdisplayimgfortraining
        self.ifboundingbox=ifboundingbox
         

def savepickle(obj,path):
    try:
        #print("saving pickle")
        pickle.dump(obj,open(path,"wb"),pickle.HIGHEST_PROTOCOL)
        return 1
    except Exception as e:
        print(e)
        return 0
    
    
def unpickle(path):
    try:
        obj=pickle.load(open(path,"rb"))
        return obj.object,obj.ifboundingbox
    except Exception as e:
        print(e)
        return None



    
#helper functions
def numpyshow(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()



def preprocessdriver(folder):
    processpath=folder+'\\processsing'
    os.mkdir(processpath)
    return processpath
    
def getstorepage(name):
    return os.path.join(PATH,'storePDF',name)

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def checkiffile(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except OSError:
        print("Error creating directory"+path)  

def checkencoding(image):
    if image.dtype!=np.uint8:
        return image.astype(np.uint8)
    return image

def boundingbox(img,location,color=(0,0,255)):
    thickness = 3
    cv2.rectangle(img, (location[1],location[0]), (location[1]+location[2],location[0]+location[3]), color, thickness)
    #cv2.imshow('image',img)
    return img

def thresholding(image,blur=None,threshold=None,rect=None,dilate=6,step=4):
    image=checkencoding(image)
    #print(image.shape)
    #numpyshow(image)
    if step>=0:
        grey=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if step>=1:
            blurred= cv2.GaussianBlur(grey,blur, 0)
            if step>=2:
                ret, thresh = cv2.threshold(blurred,threshold, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
                if step>=3:
                    struct=cv2.getStructuringElement(cv2.MORPH_RECT,rect)
                    dilate=cv2.dilate(thresh,struct,iterations=dilate)
                    return struct,dilate 
                return thresh
            return blurred
        return grey
    return None

def read_files(path):
    files = os.listdir(path)
    re_pattern = re.compile('.+?(\d+)\.([a-zA-Z0-9+])')
    files_ordered = sorted(files, key=lambda x: int(re_pattern.match(x).groups()[0]) if re_pattern.match(x) else -1)
    print(files_ordered)
    return files_ordered

def imagefromfile(path):
    return cv2.imread(path)

def greyimagefromfile(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def check(lst): 
    from itertools import repeat
    repeated = list(repeat(lst[0], len(lst)))
    return repeated == lst   

def sectionfromlocation(page,locations):
    sections=[]
    for location in locations:
        y,x,w,h=location
        sections.append(page[y:y+h, x:x+w,:])
    return sections

def findmoheight(height):
    return max(set(height), key = height.count)


def get_letter_category(letter):
    global lettercategories
    for category, letters in lettercategories.items():
        if letter in letters:
            return category
    return None

    

def section_image(pages,blur,threshold,rect,debugpath=None,dilationiterations=8):
    pages=checkencoding(pages)
    #struct,dilate=thresholding(pages,blur=blur,threshold=threshold,rect=rect)
    #blur=cv2.GaussianBlur(pages,blur, 0)
    ret, thresh = cv2.threshold(pages,threshold, 255,cv2.THRESH_BINARY)
    struct=cv2.getStructuringElement(cv2.MORPH_RECT,rect)
    dilate=cv2.dilate(thresh,struct,iterations=dilationiterations)
    contours, h = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        pass#scroll(thresh)

    sections = []
    locations=[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  
        if w > 5 and h > 5:
            section = pages[y:y+h, x:x+w]
            locations.append([y,x,w,h])
            sections.append(section)
            if debug:
                numpyshow(section)
            
    #cv2.imshow('con',sections[1])   
    #numpyshow(sections)  
    
    #https://stackoverflow.com/questions/66490374/how-to-merge-nearby-bounding-boxes-opencv
    
    
    
    
    return sections,locations

def readtext(section):
    img=PIL.Image.fromarray(section)
    return pytesseract.image_to_string(img, config=("-c tessedit""_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"" --psm 10"" -l osd"" "))

def readsinglechar(section):
    img = PIL.Image.fromarray(section)
    with pytesseract_lock:
        return pytesseract.image_to_string(img, config='--psm 10')




def replace_last_entry_in_path(path, new_entry):
    components = path.split(os.path.sep)
    components.pop()
    components.append(new_entry)
    modified_path = os.path.sep.join(components)
    return modified_path


#block information within page




#add to block




def loadheaderfootermodel():
    datapath=joinpath("sectionimg\\data2.csv")
    names=['ylocation','xlocation','sectionwidth','sectionheight','pageheight','pagewidth','class','parentPDF']
    model=predictmatrix(names,datapath,ignore=2,graph=modelgraph)
    return model




def classifyheaderfooter(model,information):
    return model.predict(np.asarray(information).reshape(1,-1))


#class textfeature:
 #   def __init__(self,letter):
  #      category=get_letter_category(letter)
   #     
    #    self.category_heights = {
     #       "short_letter": [],
      #      "tall_letter":[] ,
       #     "medium_letter":[] ,
        #    "below_line_letter": [],
         #   "medium_below_line_letter":[] ,
          #  "tall_uppercase": [],
           # "punctuation"[]: 
           # }
        #self.type=None

        
        
        
class SubBlock:
    def __init__(self, y, x, width, height):
        
        
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = None
        self.classification=None
        self.information={}
        self.relationship=None
        
        
    

    def returninformation(self):
        return [self.y,self.x,self.width,self.height]
    
    def addclassification(self,addclassification):
        self.classification=addclassification
        
    def returnarea(self,img):
        return img[self.y:self.y+self.height, self.x:self.x+self.width]
    def addallinformation(self, **kwargs):
        for key, value in kwargs.items():
            self.information[key] = value
            
    def trainclassifysubsection(self,img):
        self.classification=learnfromPAGE(self.returnarea(img))
    
    
    def getinformation(self):
        info=list(self.information.values())
        return info
    
    def classifysubsection(self,model):
        self.addclassification(model.predict(np.asarray(self.getinformation()).reshape(1,-1)))
        print(self.classification)
    
    def writeallinformation(self,writer,name):
        
        info=self.getinformation()
        info.append(self.classification)
        info.append(name)
        writer.writerow(info)


class Block:
    def __init__(self, y, x, width, height):
        
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.classification=None
        self.subblocks=[]

    def returninformation(self):
        return [self.y,self.x,self.width,self.height]
    
    def setclassification(self,classification):
        return self.classification==classification
    
    def returnarea(self,img):
        return img[self.y:self.y+self.height, self.x:self.x+self.width]
      
    def addsubblock(self,subblock):
        self.subblocks.append(subblock)
        
    def getgarbagelocations(self):
        return [subblock.returninformation() for subblock in self.subblocks if subblock.classification in ['garbage','garbageheader','garbagefooter','graph','graphpartial','figure','figurepartial','image','imagepartial']]
    
    def trainclassifysubblock(self,img):
        [subblock.trainclassifysubsection(img) for subblock in self.subblocks]
    def classifysubblock(self,model):
        [subblock.classifysubsection(model) for subblock in self.subblocks]

    
    
    def writeallsubblocks(self,writer,name):
        [subblock.writeallinformation(writer,name) for subblock in self.subblocks]
    
    
    
    def process_block(self,section,pageinformation):
        global xblur,yblur,thresholdglobal,train,equation_characters,dilationiteration   
        print("processing block",pageinformation[0])    
        struct, dilate = thresholding(section, blur=(xblur, yblur), threshold=thresholdglobal, rect=(3, 3),dilate=dilationiteration)
        contours, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(h)
        heights=[]
        for i,cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 3 and h > 3:
                subsection=section[y:y+h,x:x+w]
                data=pytesseract.image_to_data(subsection, output_type=pytesseract.Output.DICT)
                uniquecolour = len(np.vstack({tuple(r) for r in subsection.reshape(-1,3)})) #should be greyscaled
                filtered_word_list = [word for word in data['text'] if len(word) >= 2]
                numberofwords=len(filtered_word_list)
                numofchars=np.sum([len(word) for word in data['text']])
                gray_image = cv2.cvtColor(subsection, cv2.COLOR_BGR2GRAY)
                nonzero=cv2.countNonZero(gray_image)
                equationcharacters=0
                linesoftext=len(data['text'])
                #isencapsulated=h
                for i, word in enumerate(data['text']):
                    if (char in equation_characters for char in word):
                        equationcharacters+=1
                    if word != '':
                        heights.append(data['height'][i])
                
                textheight=np.mean(heights)
                if np.isnan(textheight):
                    textheight=0
                if len(heights)>0:        
                    stdheights=np.std(heights)
                else:
                    stdheights=0
                    
                    
                subblock=SubBlock(self.y+y,self.x+x, w, h)
                subblock.addallinformation(numofchars=numofchars,stdheight=stdheights,textheight=textheight,numberofwords=numberofwords,nonzero=nonzero,uniquecolour=uniquecolour,equationcharacters=equationcharacters,linesoftext=linesoftext)
                #subblock.text=data['text']
                self.addsubblock(subblock)
                
                

#page information within document


class Page:
    def __init__(self,pagenum,path):
        self.file=path
          
        #self.colour = colour
        self.pagenum = pagenum
        self.blocks=[]
        self.width=None
        self.height=None

        self.text=""
        self.textheights=[]
        
        self.data={}
        
    def getinformation(self):
        return [self.pagenum,self.width,self.height,len(self.blocks)]
    
    def get_text(self):
        return self.text
    
    
    def addtextheights(self,heights):
        for height in heights:
            if height not in self.textheights:
                self.textheights.append(height)
                
                    
    def pageorientation(self):
        if self.x>self.y:
            islandscape=0  
        elif self.x==self.y:
            islandscape=1
        else:
            islandscape = 2
        return islandscape
    
    def getdimensions(self):
        with open (self.file,'r') as f:
            image= Image.open(self.file)
            self.width,self.height=image.size
        f.close()
        
    def preprocess(self,blur,threshold,debugpath=None):
        with open (self.file,'r') as f:
            image= Image.open(self.file)
            activearea=thresholding(np.asarray(image),blur=blur,threshold=threshold,step=2)
            if debug:
                path=self.file.split('\\')[:-1]
                Image.fromarray(activearea).save(os.path.join(debugpath,'preprocess'+str(self.pagenum)+'.jpg'))
        f.close()
        return activearea
    
    
    #remove            
    def blockfromlayout(layout):
        block=[]
        for element in layout:
            if element.isinstance(Block):
                block.append(element)
        return block

    
    def pageblocks(self,locations,classifications):
        for i,location in enumerate(locations):
            y,x,w,h=location
            block=Block(y,x,w,h)
            block.classification=classifications[i]
            self.blocks.append(block)

    def overritenonconformingpages(self,name):
        global train,dilationiteration,thresholdglobal,xblur,yblur
        self.blocks=[] #wipe blocks
        img=imagefromfile(self.file)
        #img=cv2.bitwise_not(img)
        #sections,locations=section_image(img,(xblur,yblur),5,(5,5),dilationiterations=dilationiteration)
        struct,dilate=thresholding(img,blur=(xblur,yblur),threshold=thresholdglobal,rect=(3,3),dilate=dilationiteration)
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        model=loadheaderfootermodel()
        
        
        with open(joinpath('sectionimg\\data2.csv'), mode='a',newline='\n') as datafile:
            writer = csv.writer(datafile)    
        
            for cnt in contours:
                x, y, w, h =cv2.boundingRect(cnt)
                
                location=[y,x,w,h]
            
                if train :
                    sectiontype=learnfromHeaderFooter(img[y:y+h, x:x+w])
                    if sectiontype!=None and sectiontype!="unknown":
                            tempdata=[]
                            tempdata.append(location[0])
                            tempdata.append(location[1])
                            tempdata.append(location[2])
                            tempdata.append(location[3])
                            tempdata.append(self.height)
                            tempdata.append(self.width)
                            tempdata.append(sectiontype)
                            tempdata.append(name)
                            writer.writerow(tempdata)
            
            
                information=[location[0],location[1],location[2],location[3],self.height,self.width]
                classification=classifyheaderfooter(model,information)

                block=Block(y,x,w,h)
                block.classification=classification
                self.blocks.append(block)
        datafile.close()  
         
    def refineclassifications(self):
          pass
      
    def extractdata(self,flag=0,name=None,writer=None,check=True):
        global train
        #letterdict=categorize_letters_by_height()
        print("page",self.pagenum)
        
        if flag==0:
            blocks=self.blocks
            filtered_blocks = [block for block in blocks if block.classification == 'paragraph']

        elif flag==1:
            self.overritenonconformingpages(name)
            blocks=self.blocks
            filtered_blocks = [block for block in blocks if block.classification == 'paragraph']
        
        
        img=imagefromfile(self.file)
        #print(img.shape)

        pageinfo=self.getinformation()
        #constrain to content only
        for block in filtered_blocks:
            
            blockimg=block.returnarea(img)
            block.process_block(blockimg,pageinfo)
            
            
        if train:
            for block in filtered_blocks:
                block.trainclassifysubblock(img)
                block.writeallsubblocks(writer,name)
                #with concurrent.futures.ThreadPoolExecutor() as executor:
                #    executor.map(process_block, filtered_blocks,sections,model)
                #sectionsdata = [future for future in futures]    
        print("processing done",self.pagenum)    

        
    def tostring(self):
        return "x: "+str(self.x)+" y: "+str(self.y)+" pagenum: "+str(self.pagenum)+" file: "+str(self.file)+" footer: "+str(self.footer)+" header: "+str(self.header)+" text: "+str(self.text)+" data: "+str(self.data)+"\n"
            
    def addblocks(self,blocks):
        self.blocks.append(blocks)
    
    def getblockinformation(self):
        blockinfo=[(block.classification,block.returninformation()) for block in self.blocks]
        return blockinfo
            
    def returnblocks(self):
        return self.getblockinformation
    
    def gettextfrompage(self):      
        img=imagefromfile(self.file)
        garbagelocations=[]
        for block in self.blocks:
            if block.classification=='paragraph':
                garbagelocations.append(block.getgarbagelocations())
            else:
                y,x,w,h=block.returninformation()
                img[y:y+h,x:x+w]=(255,255,255)
        
        for location in garbagelocations:
            
            for sublocation in location:
                y,x,w,h=sublocation
                img[y:y+h,x:x+w]=(255,255,255)
        
        text=pytesseract.image_to_string(img)
        self.text=text
        return text
    
    def predictblocks(self,model):
        for block in self.blocks:
            block.classifysubblock(model)   
#Document information
    
  #  def secondaryprocess(self,writer=None,check=False,name=None):
 #       blocks=self.blocks
 #       filtered_blocks = [block for block in blocks if block.classification == 'paragraph']
 #       
 #       
 #       proxthreshold=5
 #       
 #       block_relationships = {}
 #       for block in filtered_blocks:
  #          
   #         subblocks=[subblock for subblock in block.subblocks if subblock.classification=='paragraph']
      #      
     #       for i, subblock1 in enumerate(subblocks):
     #           x1, y1, _, _ = subblock1.returninformation()
     #       
          #      for j, subblock2 in enumerate(subblocks):
           ##         if i != j:
          #              x2, y2, _, _ = subblock2.returninformation()
         #               distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
        #                
       #                 if distance <= proxthreshold:
      #                      if i not in block_relationships:
     #                           block_relationships[i] = []
       #                     block_relationships[i].append(j)
    #    for block_id, neighbors in block_relationships.items():
   #         print(f"Block {block_id} is next to blocks: {neighbors}")
  #          if train:
 #               #block.classifysubblock()
#
#                block.writeallsubblocks(writer,name)


class Document:
    #name = name of the document
    #folder = folder where the document is stored
    #pages = list of pages in the document
    #processingfiles = folder where images used for processing are stored
    def __init__(self,path,overwrite=False):
        pages,self.storepage=self.loadpdf(path,overwrite)
        if debug:
            components = self.storepage.split(os.path.sep)
            #components.pop()
            components.append('debug')
            modified_path = os.path.sep.join(components)
            if not os.path.exists(modified_path):
                print("creating debug folder")
                os.mkdir(modified_path)
            else:
                print("debug folder exists")
            self.debugpath=modified_path
          
        self.pages=[Page(i,c) for i,c in enumerate(pages)]
        self.name=path.split(os.path.sep)[-1].replace('.pdf','')
        self.pdfpath=self.copyoriginalpdf(path)
        #self.pages=pages
        self.dimensions=None
        self.setdimensions()
        self.numpages=len(self.pages)
        self.header,self.footer,self.leftboundary,self.rightboundary=None,None,None,None
        #self.processingfiles=preprocessdriver(self.folder)
        self.layouttemplate=None
        self.references={}
        self.templateimg=None
        
    
    def copyoriginalpdf(self,originpath):
        import shutil
        try:
            filepath=os.path.join(self.storepage,originpath.split(os.path.sep)[-1])
            
            if not os.path.exists(filepath):
                shutil.copy2(originpath, filepath)

            return filepath
                    
        except Exception as e:
            print(e)
            return None
    
        
    def generatesamplepage(self):
        return self.loadpage(random.randint(1,self.numpages-1)).file
    
    
    def boundingbox(self,location,color=(0,0,255)):
        thickness = 3
        img=imagefromfile(self.generatesamplepage())
        cv2.rectangle(img, (location[1],location[0]), (location[1]+location[2],location[0]+location[3]), color, thickness)
        #cv2.imshow('image',img)
        return img
    
    
    def numlines(datafilepath):
        with open(datafilepath, "a", newline='\n') as datafile:
                flines=datafile.readlines()
                length=len(flines)
        datafile.close()
        return length
    
    def checktrainheaderfooter(self):
        return datacontroller(self.name)
    

    
    def trainheaderfooter(self):
        self.checktemplate()
        arr=checkencoding(self.templateimg)

        arr=cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


        locations=self.layouttemplate[1]
        #cut=self.numpages
        datafilepath=joinpath("sectionimg\\data2.csv")
        with open(datafilepath, "a", newline='\n') as datafile:
            writer = csv.writer(datafile)
            sections=sectionfromlocation(arr,locations)

            for i,section in enumerate(sections):
                if trainhelp:
                    try:
                        boximg=self.boundingbox(locations[i])
                        if not os.path.exists(os.path.join(PATH,'debug')):
                            os.mkdir(os.path.join(PATH,'debug'))
                        #print (inspect.stack()[1])      
                        Image.fromarray(boximg).save(os.path.join(PATH,'debug\\')+inspect.stack()[1][3]+'.jpg','JPEG')
                    except:
                        print("error")
                        pass
                    #scroll(np.asarray(boximg))
                sectiontype=learnfromHeaderFooter(section)
                if sectiontype!=None and sectiontype!="unknown":
                    
                        # create the csv writer
                        
                        

                        # write a row to the csv file
                        tempdata=[]
                        
                        tempdata.append(locations[i][0])
                        tempdata.append(locations[i][1])
                        tempdata.append(locations[i][2])
                        tempdata.append(locations[i][3])
                        tempdata.append(self.dimensions[0])
                        tempdata.append(self.dimensions[1])
                        tempdata.append(sectiontype)
                        tempdata.append(self.name)
                        writer.writerow(tempdata)
        datafile.close()
        
        

    
    def headerfooterdriver(self):
        model=loadheaderfootermodel()
        self.checktemplate()
        classification=[]
        headercuttoff=0
        footercuttoff=self.dimensions[0]
        
        paragraphheadercutoff=0
        paragraphfootercutoff=self.dimensions[0]
        
        for location in self.layouttemplate[1]:
            #print(location)
            information=[location[0],location[1],location[2],location[3],self.dimensions[0],self.dimensions[1]]
            classi=(classifyheaderfooter(model,information))
            #print(classi)
            classification.append(classi)
            if classi=="footer" and (footercuttoff>(location[0])):

                footercuttoff=location[0]
            elif classi=="header" and (headercuttoff<(location[0]+location[3])):

                headercuttoff=location[0]+location[3]
                
            if classi=="paragraph" and (paragraphfootercutoff>(location[0]+location[3])):

                paragraphfootercutoff=location[0]+location[3]
                
            #print(headercuttoff)    
            if classi=="paragraph" and (paragraphheadercutoff<(location[0])):

                paragraphheadercutoff=location[0]
            #location.classification=classification

        if checkheaderfooter:    
            numpyshow(imagefromfile(self.loadpage(3).file)[headercuttoff:footercuttoff,:,:])
        
        if headercuttoff>footercuttoff:
            headercuttoff=0
            footercuttoff=self.dimensions[0]
        
        if headercuttoff!=0:
            self.header=headercuttoff
        elif paragraphheadercutoff!=0:
            self.header=paragraphheadercutoff
        else:
            self.header=0
            
        if footercuttoff!=self.dimensions[0]:
            self.footer=footercuttoff
        elif paragraphfootercutoff!=self.dimensions[0]:
            self.footer=paragraphfootercutoff
        else:
            self.footer=self.dimensions[0]   
        #print(classification) 
        #do sides        
        return self.layouttemplate[1],classification
    
    def testtemplatetopage(self):
        pages=[]
        for i,page in enumerate(self.pages):
            pageimg=greyimagefromfile(page.file)
            for location in self.layouttemplate[1]:
                pageimg[location[0]:location[0]+location[3],location[1]:location[1]+location[2]]=255
            pageimg=checkencoding(pageimg)  
            r,pageimg=cv2.threshold(pageimg,10, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)  
            
            if cv2.countNonZero(pageimg)>0:
                #numpyshow(pageimg)
                pages.append(i)
            #numpyshow(pageimg)
        return pages


    
    #def train(self,datafile):    
     #   self.checktemplate()
      #  self.headerfooterdriver()
#
 #       classifications=learnfromPAGE(self.layouttemplate[0])
  #      datapath=joinpath("sectionimg\\data2.csv")
   #     names=['numofcharacters','mocharheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','relitivelocation','class','parentPDF']
        
        #model=classify.predictmatrix(names,datapath,graph=False)
            
        #return header,footer,locations
    
    
    
    def definepagesections(self):
        #self.checktemplate()
        locations,classifications=self.headerfooterdriver()
        for page in self.pages:
            page.pageblocks(locations,classifications)

    def loadpage(self,pagenum):
        #print(len(self.pages))
        return self.pages[pagenum]
        
 
    def setdimensions(self):
        self.dimensions=self.checkdimensions()
        
    def checkdimensions(self):
        dimentionlist=[]
        for page in self.pages:
            page.getdimensions()
            dimentionlist.append((page.height,page.width))
        numdimentions=np.unique(dimentionlist,axis=0)    
        if len(numdimentions)==1:
            #print(numdimentions)
            #print("all pages have the same dimensions")
            return numdimentions[0]
        else:
            print("Error: pages have different dimensions")
            pass
            #handle case where pages have different dimensions will not cover this case for now

            
    def createtemplate(self):
        #print(self.dimensions)
        arr=np.zeros(self.dimensions)
        
        if debug:
            for page in self.pages:
                arr+=page.preprocess((3,3),20,self.debugpath)/self.numpages#possible floating point error
                
        else:
            for page in self.pages:
                arr+=page.preprocess((3,3),20)/self.numpages#possible floating point error
                
           
        sections,locations=section_image(arr,(15,15),(2/self.numpages)*255,(5,5))
        #arr=arr.astype(np.uint8) 
        if debug:       
            (Image.fromarray(arr)).convert('L').save(joinpath('storePDF\\'+self.name+'\\debug\\test.jpg'))
        
        
        self.layouttemplate=sections,locations
        self.templateimg=arr

    
    def checktemplate(self):
        if self.layouttemplate==None:
            self.createtemplate()
        else: return 
        
    def tostring(self):
        return "name: "+str(self.name)+" folder: "+str(self.folder)+" pages: "+str(self.pages)+" processingfiles: "+str(self.processingfiles)+"\n"
    
    
    def get_page_number_for_text(self, text):
        for i, page in enumerate(self.pages):
            if text in page.get_text():
                return i + 1  
        return None

    def get_page_path(self, page_number):
        if 1 <= page_number <= len(self.pages):
            return f"Path to page {page_number}"
        return None
    
    
    
    def loadpdf(self,startpath,overwrite,userpassword=None,debug=False):
        try:   
            pagepaths=[]
            split= startpath.split(os.path.sep)[-1].replace('.pdf','')
            #print(split)
            storepath= joinpath('storePDF\\'+split)
            #print(storepath)
            
            if os.path.exists(storepath) and overwrite==True:#makeshift overwrite
                delete_folder(storepath)
            
            if os.path.exists(storepath):
                for img in read_files(storepath):
                    if img.endswith('.jpg'):
                        pagepaths.append(os.path.join(storepath,img))
                        if debug:
                            print(img+" exists")
                return pagepaths,storepath
            #handle write errors later
            else:
                #testpath=joinpath('testPDF')
                info = p2i.pdfinfo_from_path(startpath, userpassword, poppler_path=None)
                maxPages = info["Pages"]
                checkiffile(storepath)
                count=0
                for i in range(maxPages+1):
                    p2i.convert_from_path(startpath,first_page=count,last_page=i,output_folder=storepath,userpw=userpassword,fmt="jpg",output_file="page",poppler_path=None,strict=False,thread_count=4)
                    count+=1
                    if debug:
                        print("created page"+str(count)+" to"+str(maxPages))
                for img in read_files(storepath):
                    #print(img)
                    pagepaths.append(os.path.join(storepath,img))
                    
                return pagepaths,storepath
            
        
        except Exception as e:
            print("exception")
            print(e)




    def addblockstopages(self,locations,classifications):
        for page in self.pages:
            page.pageblocks(locations,classifications)



    def documenttext(self):
        text=""
        for page in self.pages:
            text+=page.gettextfrompage()
        return text
    
    def checkremovefile(self,datafilepath,override):
        if datacontroller(self.name):
            print("data already exists")
            if override:
                removedata(self.name,datafilepath)
            else:
                return

    
    
def driver(startpath,overwrite=False,userpassword=None,debug=False):
    
    doc=Document(startpath)
    doc.trainheaderfooter(override=True)
    locations,classifications=doc.headerfooterdriver()
    doc.addblockstopages(locations,classifications)
    return doc





from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageEnhance, ImageTk
from visualise import *


def enforcepath(path):
    return path.replace("/","\\")

def colourfromclass(classification):
    #print(classification)
    if classification== "header":
        return (255,0,0)
    elif classification==   "footer":
        return (0,255,0)
    elif classification==  "paragraph":
        return (0,0,255)
    elif classification== "rightsidetext":
        return (255,255,0)
    elif classification== "leftsidetext":
        return (0,255,255)
    
    
def confirmation(message):
    print("")
    msg_box = messagebox.askquestion('Confirmation', message, icon='warning')
    if msg_box == 'yes':
        return True
    else:
        return False
    
def show_popup(message):
    messagebox.showinfo("Warning", message)    
    
    
def update_image():
    global interfaceimg,interfacedoc,interfacecurrentpage,adapted_img,xblur,yblur,dilationiteration,thresholdglobal,displayimgfortraining,ifboundingbox

    if interfacecurrentpage==0:
        previous_button.config(bg="gray")
    else:
        previous_button.config(bg=default_color)
        
    if interfacecurrentpage==interfacedoc.numpages:
        next_button.config(bg="gray")
    else:
        next_button.config(bg=default_color)
        
    if displayimgfortraining:    
        struct,dilate=thresholding(np.asarray(interfaceimg),blur=(xblur,yblur),threshold=thresholdglobal,rect=(3,3),dilate=dilationiteration)
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colourdilate=cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            x, y, w, h =cv2.boundingRect(cnt)
            adapted_img=boundingbox(colourdilate,[y,x,w,h],(200,10,200))
       
    else:
        adapted_img = np.asarray(interfaceimg)
    
    
        
    if ifboundingbox:
        for item in (interfacedoc.loadpage(interfacecurrentpage).getblockinformation()):
            classification,location=item
            adapted_img=boundingbox(np.asarray(adapted_img),location,colourfromclass(classification))
    
    
    adapted_img=Image.fromarray(adapted_img)    
    adapted_img=ImageTk.PhotoImage(image=adapted_img)
    image_container.create_image(0, 0, image=adapted_img, anchor=NW)
    image_container.config(scrollregion=image_container.bbox("all"))



def new_document():
    global interfaceimg, interfacedoc, interfacecurrentpage,displayimgfortraining,ifboundingbox
    try:
        pdf = filedialog.askopenfilename(title="Select file", filetypes=(("pdf files", "*.pdf"), ("all files", "*.*")))
        if pdf:
            pdf = enforcepath(pdf)
            interfacedoc = Document(pdf,overwrite=True)
            interfacecurrentpage=0
            pageobj = interfacedoc.pages[0]
            interfaceimg= Image.open(pageobj.file)
            ifboundingbox=None
            save_button.config(bg=default_color)
            load_button.config(bg="gray")
            previous_button.config(bg=default_color)
            next_button.config(bg=default_color)
            xblur_slider.config(bg="gray",label="X Blur")
            yblur_slider.config(bg="gray",label="Y Blur")
            dilation_slider.config(bg="gray",label="Dilation")
            threshold_slider.config(bg="gray",label="Threshold")
            launch_processing_button.config(bg="gray")
            launch_text_processing_button.config(bg="gray")
        update_image()
    except Exception as e:
        print(e)
        


def previous():
    global interfaceimg,interfacedoc,interfacecurrentpage
    if interfaceimg and interfacecurrentpage>0:
        interfacecurrentpage-=1
        interfaceimg = Image.open(interfacedoc.pages[interfacecurrentpage].file)
        #interfaceimg = interfaceimg.transpose(Image.FLIP_LEFT_RIGHT)
        update_image()


def next():
    global interfaceimg,interfacedoc,interfacecurrentpage
    if interfaceimg and interfacecurrentpage<interfacedoc.numpages-1:
        try:
            interfacecurrentpage+=1
            interfaceimg = Image.open((interfacedoc.pages[interfacecurrentpage]).file)
            update_image()
        except:
            interfacecurrentpage-=1


def load():
    global interfaceimg,interfacedoc,interfacecurrentpage,ifboundingbox
    name = filedialog.askopenfilename(initialdir="Untitled", title="Select file",
                                        filetypes=(("pickle files", "*.pkl"), ("all files", "*.*")))
    if name:
        interfacedoc,ifboundingbox = unpickle(name)
        interfacecurrentpage=0
        pageobj = interfacedoc.pages[0]
        interfaceimg= Image.open(pageobj.file)
        update_image()

def save():
    global interfaceimg,interfacedoc,ifboundingbox,displayimgfortraining
    if interfaceimg:
        name = filedialog.askdirectory()
        filename=(interfacedoc.name)+".pkl"
        savename=os.path.join(name,filename)
        saveobj=Savepickleobject(interfacedoc,ifboundingbox)
        if name:
            savepickle(saveobj,savename)
  
def save_text_to_file(filename, text):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to '{filename}' successfully.")
    except Exception as e:
        print(f"An error occurred while saving the text: {e}")  
            
def launch_processing():
    global interfaceimg,interfacedoc,ifboundingbox,train
    
    

    if interfaceimg:
        
        if train:
            checkifoverride=interfacedoc.checktrainheaderfooter()
            check=False
            show_popup("Resizing the window will cause the program to crash. Please do not resize the window.")
            #print("checkoverride: ",checkifoverride)    
            if checkifoverride:
                check=confirmation("You have already trained the header and footer. Do you want to override?")
            if check == checkifoverride:
                interfacedoc.checkremovefile(joinpath("sectionimg\\data2.csv"),check)            
                interfacedoc.checkremovefile(joinpath("sectionimg\\data.csv"),check)  
            

                interfacedoc.trainheaderfooter()

        locations,classification=interfacedoc.headerfooterdriver()
        interfacedoc.addblockstopages(locations,classification)

        
        unlikelypages=interfacedoc.testtemplatetopage()
        #interfacedoc.addblockstopages(locations,classification)
        
        filename= joinpath("sectionimg\\data.csv")
        

        if train and check:
            with open(filename, 'a') as datafile:
                
                writer=csv.writer(datafile)
                
                for likelypage in interfacedoc.pages:
                    if likelypage.pagenum in unlikelypages:
                        likelypage.extractdata(flag=1,name=interfacedoc.name,writer=writer,check=check)
                    else:
                        likelypage.extractdata(name=interfacedoc.name,writer=writer,check=check)
            
                
            datafile.close()
        
        else:
            #names=['numofcharacters','stdheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','class','parentPDF']
            #model=predictmatrix(names,filename,2)
            for likelypage in interfacedoc.pages:
                if likelypage.pagenum in unlikelypages:

                    likelypage.extractdata(flag=1)
                else:
                    likelypage.extractdata()
        #subblock.addallinformation(numofchars=numofchars,stdheight=stdheights,numberofwords=numberofwords,nonzero=nonzero,uniquecolour=uniquecolour,equationcharacters=equationcharacters,linesoftext=linesoftext)
            names=['numofcharacters','stdheight','textheight','numberofwords','nonzero','uniquecolour','equationcharacters','linesoftext','class','parentPDF']
            model=predictmatrix(names,filename,2)
            
            for likelypage in interfacedoc.pages:
                likelypage.predictblocks(model)



        launch_text_processing_button.config(bg=default_color)

        ifboundingbox=True

        
        update_image()





def launch_text_processing():
    global interfaceimg,interfacedoc,ifboundingbox,train,ifboundingbox
    if ifboundingbox and interfacedoc:
        entiretext=interfacedoc.documenttext()
        save_text_to_file(os.path.join(interfacedoc.storepage,"entiretext.txt"),entiretext)
        import processTEXT as processTEXT
        
        guidelinefile = filedialog.askopenfilename(initialdir="Untitled", title="Select Guideline file",
                                        filetypes=(("text files", "*.txt"), ("all files", "*.*")))
        success=processTEXT.driver(entiretext,guidelinefile,documentobject=interfacedoc)
        if success:
            

            webbrowser.open_new(success)
            #system = platform.system().lower()
            #E:\Lvl4Project\ScientificEvidenceReviewApplication\server\storePDF\Generatedpdfpopulated\Generatedpdfpopulatedreport.pdf
            #print(success)
            #if system == 'darwin':  # macOS
           #     subprocess.run(["open", success], check=True)
           # elif system == 'linux':
        #        subprocess.run(["xdg-open", success], check=True)
        #    elif system == 'windows':
       #         subprocess.run(["start", success], check=True)
     #       else:
     #           print("Unsupported operating system")
        else:
            messagebox.showerror("Error", "text processing failed")
    update_image()
            
            
def change_xblur(var):
    global interfaceimg,xblur
    xblur =int(var)
    update_image()

def change_yblur(var):
    global interfaceimg,yblur
    yblur =int(var)
    update_image()
    
def change_dilation(var):
    global interfaceimg,dilationiteration
    dilationiteration =int(var)
    update_image()
    
def change_threshold(var):
    global interfaceimg,thresholdglobal
    thresholdglobal =int(var)
    update_image()

def tickbox():
    global train
    train = not train

def toggleimg():
    global displayimgfortraining
    displayimgfortraining = not displayimgfortraining
    if interfaceimg:
        update_image()
    

def interface():
    global equation_characters,root,default_color, image_container, open_button, previous_button, next_button, xblur_slider, yblur_slider, dilation_slider, threshold_slider, save_button, load_button, displayimgfortraining, ifboundingbox, interfaceimg, interfacedoc, interfacecurrentpage,contrast_val,train,debug,pdpassword,userpassword,trainhelp,xblur,yblur,dilationiteration,thresholdglobal,checkheaderfooter,modelgraph,lettercategories,launch_processing_button,launch_text_processing_button
    

    root = Tk()
    root.title("Image Editor")
    root.geometry('1900x900')
    default_color = root.cget('bg')

    displayimgfortraining=False
    interfaceimg = None
    ifboundingbox=None
    contrast_val = 50
    train=False
    debug=False
    pdpassword=None
    userpassword=None
    train=False
    modelgraph=False
    checkheaderfooter=False
    trainhelp=False
    xblur=5
    yblur=5
    dilationiteration=8
    thresholdglobal=5
    
    tesslocation=store_filepath_or_input()
    if not tesslocation:
        messagebox.showerror("Error", "Pytesseract location not found. Please enter the location of pytesseract.exe in the following window")
        tesslocation=get_filepath(filedialog.askopenfilename(initialdir="Untitled", title="Select file",filetypes=(("pickle files", "*.exe"), ("all files", "*.*"))))
    pytesseract.pytesseract.tesseract_cmd = r'%s' %tesslocation
    lettercategories = {
        "short_letter": "acemnorsuvwxz",
        "tall_letter": "bdfhklABCDEFGHIJKLMNOPRSTUVWXYZ123456789",
        "medium_letter": "ti",
        "below_line_letter": "gpqy",
        "medium_below_line_letter": "j",
        "tall_uppercase": "Q"
    }
    equation_characters = "+-="#add how pytesseract reads the equation characters here
    
    
    
    open_button = Button(text='New Document', font=('Arial', 20), command=new_document)
    previous_button = Button(text='Previous Page', font=('Arial', 10), command=previous, bg="gray",
                                    width=15)
    next_button = Button(text='Next Page', font=('Arial', 10), command=next, bg="gray", width=15)

    xblur_slider = Scale(from_=1, to=75, orient=HORIZONTAL,resolution=2, bg="gray", command=change_xblur)
    yblur_slider = Scale(from_=1, to=75, orient=HORIZONTAL,resolution=2, bg="gray", command=change_yblur)
    dilation_slider = Scale(from_=1, to=20, orient=HORIZONTAL, bg="gray", command=change_dilation)
    threshold_slider=Scale(from_=1, to=255, orient=HORIZONTAL, bg="gray", command=change_threshold)
    toggleimgview = Checkbutton(text='Toggle Image View', font=('Arial', 20), command=toggleimg)
    save_button = Button(text='Save', font=('Arial', 20), command=save, bg="gray")
    load_button = Button(text='Load', font=('Arial', 20), command=load, bg="gray")
    canvas_frame = Frame(root)
    canvas_frame.pack(fill="both", expand=True)
    image_container = Canvas(canvas_frame, borderwidth=5, relief="groove", width=300, height=300)
    image_container.pack(fill="both", expand=True, side=LEFT)

    vertical_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=image_container.yview)
    vertical_scrollbar.pack(fill="y", side=RIGHT)
    horizontal_scrollbar = ttk.Scrollbar(root, orient="horizontal", command=image_container.xview)
    horizontal_scrollbar.pack(fill="x", side=BOTTOM)

    train_tickbox = Checkbutton(text='Train', font=('Arial', 20),command=tickbox)

    launch_processing_button=Button(text='Launch Processing', font=('Arial', 20), bg="gray",command=launch_processing)
    launch_text_processing_button=Button(text='Launch Text Processing', font=('Arial', 20), bg="gray",command=launch_text_processing)

    image_container.config(yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizontal_scrollbar.set)


    open_button.pack(anchor='nw', side=LEFT)
    save_button.pack(anchor='nw', side=LEFT)
    load_button.pack(anchor='nw', side=LEFT)
    xblur_slider.pack(anchor='w', side=LEFT)
    yblur_slider.pack(anchor='w', side=LEFT)
    dilation_slider.pack(anchor='w', side=LEFT)
    threshold_slider.pack(anchor='w', side=LEFT)
    toggleimgview.pack(anchor='w', side=LEFT)

    previous_button.pack(anchor='w', side=LEFT)
    next_button.pack(anchor='w', side=LEFT)

    train_tickbox.pack(anchor='w', side=LEFT)
    launch_processing_button.pack(anchor='w', side=LEFT)
    launch_text_processing_button.pack(anchor='w', side=LEFT)
    root.mainloop()




    
if __name__ == "__main__":
    

    interface()
    #test code
    #pages,storepath=loadpdf(joinpath("server\\storePDF\\2022-17092-001"))
    
    #doc=Document(joinpath("testPDF\\2022-17092-001.pdf"))

    #doc=Document(joinpath("testPDF\\s44155-023-00042-4"))
    #doc=Document(joinpath("testPDF\SSRN-id3052318.pdf"))
    #doc.trainheaderfooter(override=False)
    #locations,classifications=doc.headerfooterdriver()
    #doc.addblockstopages(locations,classifications)
    
    #img=imagefromfile(doc.pages[3].file)[locations[0][0]:locations[0][0]+locations[0][3],locations[0][1]:locations[0][1]+locations[0][2]]
    
    #est_height,ocrresult=estimate_text_height(img)
    #numpyshow(img)
    #print(est_height)
    #print(ocrresult)
    #unlikelypages=doc.testtemplatetopage()
    #dilationiteration=20
    #for likelypage in doc.pages:
        #if likelypage.pagenum in unlikelypages:
            #print(likelypage.pagenum, "page unlikely to conform to template")
            #likelypage.extractdata(flag=1,name=doc.name)
       # else:
           # likelypage.extractdata()

    
    #doc.definepagesections()
    #print(doc.tostring())
    #print(do