import csv
from PyPDF2 import *
import pandas as pd
from nltk import *
from re import *
from numpy import *
import os
import PIL
import cv2
import scipy
from pdf2image import *
import threading
#import tensorflow as tf
import tkinter as tk
import math
#print("TensorFlow version:", tf.__version__)
from treeNode import HeadNode,SectionNode,ReferenceNode,MeaningNode
from visualise import classify,scroll,learnfrom
from tkinterscrollable import ScrollableImage
import pytesseract  

import logging
import threading
import time
import visualise
from sklearn.metrics.pairwise import cosine_similarity
from pageobj import Page,Document
#config
pytesseract.pytesseract.tesseract_cmd = r'C:\User\Users\Envs\proj10\Lib\site-packages\tesseract.exe'
PATH= os.path.dirname(__file__)
DEBUG=False
SECTIONS=False
TRAIN=True
SIMILARITY=0.9
TITLES=["Title","Author","Publisher","ISBN","Edition","Year","Pages","Price","Language","Format","Category","Description","Table of Contents","Index","Notes","Bibliography","Appendix","Glossary","Preface","Acknowledgements","Dedication","Epilogue","Foreword","Introduction","Prologue","Epigraph","Summary"]
BLUR=(5,5)
RECT=(5,5)
NAMES=['numofcharacters','mocharheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','class','parentPDF']

def joinpath(folder):
    return os.path.join(PATH,folder)

#https://stackoverflow.com/questions/62876241/os-listdirfolder-function-printing-out-the-wrong-order-of-files
def read_files(path):
    files = os.listdir(path)
    re_pattern = re.compile('.+?(\d+)\.([a-zA-Z0-9+])')
    files_ordered = sorted(files, key=lambda x: int(re_pattern.match(x).groups()[0]))
    return files_ordered
    
def loadpdf(startpath):
    try:   
        pagepaths=[]
        split= startpath.split('\\')[-1].replace('.pdf','')
        storepath= joinpath('storePDF/'+split)
        
        if os.path.exists(storepath):
            for img in read_files(storepath):
                pagepaths.append(os.path.join(storepath+'/',img))
            if DEBUG:
                print(img+" exists")
            return pagepaths
        #handle write errors later
        else:
            #testpath=joinpath('testPDF')
            info = pdfinfo_from_path(startpath, userpw=None, poppler_path=None)
            maxPages = info["Pages"]
            
            os.mkdir(storepath)
            count=0
            for page in range(1,maxPages+1,4):
                pages=convert_from_path(startpath,thread_count=4,output_folder=storepath)
                if DEBUG:
                    print("created page"+str(count)+"to"+maxPages)
        # for i in range(len(pages)):
            #    pages[i].save(os.path.join(storepath+'/','page'+ str(i) +'.jpg'), 'JPEG')
            for img in read_files(storepath):
                pagepaths.append(os.path.join(storepath+'/',img))
                
            return pagepaths
    except Exception as e:
        print(e+" error in pdftoimage")







def cutoffHeight(pages):
    page=pages[1:-1,:,:] #case of covers
    header_height = 0
    max_similarity=SIMILARITY
    similarity=1
    while similarity > max_similarity:
        meansim=[]
        for i in range(2,page.shape[0],math.floor(page.shape[0]/6)):
            
            #similarity = (page[1,header_height,:]==page[i,header_height,:]).all(axis=(0,1)).mean()
            
            similarity =1- mean((page[0,header_height,:] - page[i,header_height,:])**2)
            meansim.append(similarity)
        

        if asarray(meansim).mean() < max_similarity:
            break
        header_height+=1
    return header_height-5
 
def cutoffFooter(pages):
    page=pages[1:-1,:,:] #case of covers
    page_height=pages.shape[1]-1
    footer_height = 0
    max_similarity=SIMILARITY
    similarity=1
    
    while similarity > max_similarity:
        meansim=[]
        for i in range(2,page.shape[0],math.floor(page.shape[0]/6)):
            
            #similarity = (page[1,(page_height-footer_height),:]==page[i,(page_height-footer_height),:]).all(axis=(0,1)).mean()
            similarity =1- mean((page[1,page_height-footer_height,:] - page[i,page_height-footer_height,:])**2)
            meansim.append(similarity)
       

        if asarray(meansim).mean() < max_similarity:
            break
        footer_height+=1
    return footer_height-5

def scanheader(pages,footer_height = 50):
    arr=[]
    for i in range(pages.shape[0]):

        grey=cv2.cvtColor(pages[i], cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey,3, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
        #print(thresh.shape)
        arr.append(thresh)
    arr=asarray(arr)
    
    
    header_height=pages.cutoffHeight(arr)
    footer_height=pages.cutoffFooter(arr)
    if DEBUG:
        print(arr.shape)
        print(grey.shape)
        print(header_height)
        print(footer_height)
    #scroll(pages[:,header_height:-footer_height,:,:][10])
    return pages[:,header_height:-footer_height,:,:]
    

def section_image(pages,blurthres=BLUR,rect=RECT):
    grey=(cv2.cvtColor(pages, cv2.COLOR_BGR2GRAY))
    blur= cv2.GaussianBlur(grey,(blurthres), 0)
    ret, thresh = cv2.threshold(blur,3, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    
    struct=cv2.getStructuringElement(cv2.MORPH_RECT,(rect))
    dilate=cv2.dilate(thresh,struct,iterations=4)
    contours, h = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        scroll(dilate)
    #cv2.imshow('debug',dilate)
    #cv2.waitKey(0) & 0xFF
    sections = []
    locations=[]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  
        if w > 20 and h > 20:
            section = pages[y:y+h, x:x+w]
            locations.append([x,y,w,h])
            sections.append(section)
    #cv2.imshow('con',sections[1])     
    return sections,locations


def readtext(section):
    img=PIL.Image.fromarray(section)
        
  
    return pytesseract.image_to_string(img)





def extractdata(sections):
    
    sectionsdata=[]
    
    
    print(len(sections))
    for section in sections:
        
        
        chars=readtext(section)
        numofchars=len(chars)

        
        height=[]
        grey=(cv2.cvtColor(section, cv2.COLOR_BGR2GRAY))
        ret, thresh = cv2.threshold(grey,5, 255,cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
        contours, he = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        secheight,secwidth=section.shape[0],section.shape[1]
        
        nonzero=cv2.countNonZero(thresh)

        uniquecolour = len(vstack({tuple(r) for r in section.reshape(-1,3)}))
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt) 
            height.append(h)
            
    
    #add to loop for performance
        moheight= max(set(height), key = height.count)      
        if DEBUG:
            print(f"most often letter pixel height: %s"%moheight)
            print(uniquecolour)
        
        sectionsdata.append([numofchars,moheight,nonzero,uniquecolour,secheight,secwidth])
    
    if DEBUG:
        print(sectionsdata)
    return sectionsdata






#synonim check
def checkiftitle(text,titles=TITLES):
    if text in titles:
        return text
    #for title in titles:
     #   if text in (x:=title):
      #      return title
    return False


def runPDF(filename):
    print("converting pdf to image")
    p,q=pdftoimage(filename)
    print("complete")
    
    print("Transforing pages")
    cropped=pageimg(asarray(p),True)
    shapepages=cropped.shape[1]*cropped.shape[0] 
    combined=reshape(cropped,(shapepages,cropped.shape[2],cropped.shape[3]))
    print("complete")
    
    print("sectioning image")
    sections,locations =section_image(combined)
    print("complete")
    
    print("extracting data")
    extracteddata=extractdata(sections)
    #sectionmap=dict(zip(sections,extractdata(sections)))
    #print(locations)
    if DEBUG:
        scroll(combined)
    
    if SECTIONS:
        classify(sections)
            
    sectiontype=[]
    
    print("training breakpoint")       
    if TRAIN:
        
        with open(os.path.join(PATH,'sectionimg/data.csv'), 'a') as f:
            for i in range(len(sections)):
                
                sectionclass=learnfrom(sections[i])
                if sectionclass:
                    
                        # create the csv writer
                        writer = csv.writer(f)

                        # write a row to the csv file
                        tempdata=extracteddata[i]
                        tempdata.append(sectionclass)
                        tempdata.append(filename)
                        
                        writer.writerow(tempdata)
            
            
    
    else:
        from classify import predictmatrix
        predictmodel=predictmatrix()
        #for i in range(len(sections)):
        sectiontypes=(predictmodel.predict(extracteddata))
        #if DEBUG:
        print(sectiontypes)
        removed =[]
        flag=0
        
        for i,t in enumerate(sectiontypes[::-1]):
            if t=="trash" or t=="figure"or t=="figurepartial"or t=="graph" or t=="graphpartial" or t=="image"or t=="imagepartial":
                x,y,w,h=locations[i]
                print("remove "+t)
                removed.append(combined[y:y+h,x-w:x,:])
                combined[y:y+h,x-w:x,:]=0
                #scroll(removed[i])
            else:
                sectiontext=readtext(sections[::-1][i])
                if len(sectiontext)<3 and (x:=checkiftitle(sectiontext)):
                    flag=1
        #if DEBUG:
        #scroll(combined)
        

            #learnmain(section)
    
#tuned for id4347886


    

if __name__ == '__main__':
    runPDF('s44155-023-00042-4.pdf',False)#combined img no headers






#scroll(cropped[-1])
     


#t=PIL.Image.fromarray(combined)






#cv2.imshow('Section', sections[0])



    



    

    
