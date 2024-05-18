import numpy as np
import cv2
from PIL import Image,ImageTk
import tkinter as tk
from tkinterscrollable import ScrollableImage
import os

PATH= os.path.dirname(__file__)


def classify(img):
   cv2.imshow('Section', img[0])
   key = cv2.waitKey(0) & 0xFF

 # KEYBOARD INTERACTIONS
   if key == ord('d'):
      cv2.destroyAllWindows()
      try:
         classify(img[1:])
      except Exception:
         print(Exception)

def learnfrom(img):
      
      cv2.imshow('Section', img)
      key = cv2.waitKey(0) & 0xFF
      if key == ord('t'):
            cv2.destroyAllWindows()
            return "trash"
            
      elif key == ord('g'):
            cv2.destroyAllWindows()
            return "graph"
      
      elif key == ord('h'):
            cv2.destroyAllWindows()
            return "graphpartial"
      
      elif key == ord('f'):
            cv2.destroyAllWindows()
            return "figure"
      
      elif key == ord('d'):
            cv2.destroyAllWindows()
            return "figurepartial"
      
      elif key == ord('i'):
            cv2.destroyAllWindows()
            return "image"
            
      elif key == ord('o'):
            cv2.destroyAllWindows()
            return "imagepartial"
      
      elif key == ord('t'):
            cv2.destroyAllWindows()
            return "imagepartial"
      
      elif key == ord('p'):
      # text 
            cv2.destroyAllWindows() 
            return "paragraph"
      
      elif key == ord('s'):
      # text 
            cv2.destroyAllWindows() 
            return "sentence"
      
      elif key == ord('l'):
      # text 
            cv2.destroyAllWindows() 
            return "loosetext"
      
      
      elif key == ord('e'):
      # text 
            cv2.destroyAllWindows() 
            return "equation"
      
      elif key == ord('r'):
      # text 
            cv2.destroyAllWindows() 
            return "equationpartial"



def learnfromPAGE(img):
      cv2.imshow('Subblock Classification', img)
      key = cv2.waitKey(0) & 0xFF
      if key == ord('t'):
            cv2.destroyAllWindows()
            return "table"
            
     # elif key == ord('h'):
    #        cv2.destroyAllWindows()
   #         return "textheader"
      
    #  elif key == ord('t'):
  #          cv2.destroyAllWindows()
  #          return "textblock"
      
      elif key == ord('i'):
            cv2.destroyAllWindows()
            return "image"
      elif key == ord('o'):
            cv2.destroyAllWindows()
            return "imagepartial"
      
      

      
      elif key == ord('f'):
            cv2.destroyAllWindows()
            return "figure"
      elif key == ord('d'):
            cv2.destroyAllWindows()
            return "figurepartial"      
      
      
      elif key == ord('p'):
            cv2.destroyAllWindows() 
            return "paragraph"
      
      elif key == ord('s'):
            cv2.destroyAllWindows() 
            return "sentence"
      
      elif key == ord('l'):
            cv2.destroyAllWindows() 
            return "loosetext"
      
      elif key == ord('h'):
            cv2.destroyAllWindows()
            return "Heading"
      
      elif key == ord('e'):
            cv2.destroyAllWindows() 
            return "equation"
      
      elif key == ord('p'):
            cv2.destroyAllWindows() 
            return "equationpartial"



def learnfromHeaderFooter(img):
      cv2.imshow('Block Classification', img)
      key = cv2.waitKey(0) & 0xFF
      if key == ord('h'):
            cv2.destroyAllWindows()
            return "header"
      elif key == ord('f'):
            cv2.destroyAllWindows()
            return "footer"
      elif key == ord('l'):
            cv2.destroyAllWindows()
            return "leftsidetext"
      elif key == ord('r'):
            cv2.destroyAllWindows()
            return "rightsidetext"
      elif key == ord('p'):
            cv2.destroyAllWindows()
            return "paragraph"      
      else:
            cv2.destroyAllWindows()
            return "unknown"      
            
            
            
            
            
            
            
            
            
            
            
            
            
            
     
            
            
            
import inspect
def scroll(combined):   
   
      if not os.path.exists(os.path.join(PATH,'debug')):
            os.mkdir(os.path.join(PATH,'debug'))
      #print (inspect.stack()[1])      
      Image.fromarray(combined).save(os.path.join(PATH,'debug\\')+inspect.stack()[1][3]+'.jpg','JPEG')      

      root = tk.Tk()
      #root.attributes('-fullscreen', True)
      
      
      
      img = ImageTk.PhotoImage(file=os.path.join(PATH,"debug\\")+inspect.stack()[1][3]+'.jpg')

      image_window = ScrollableImage(root, image=img, scrollbarwidth=6, 
                                    width=1600, height=1200)
      image_window.pack()

      root.mainloop()