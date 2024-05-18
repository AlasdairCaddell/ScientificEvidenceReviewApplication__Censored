# Import the required Libraries
import os
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
from pageobj import Document
from PIL import Image, ImageTk


#pdf page image window


import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from pageobj import *

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('700x500')
        self.title('Main Window')
        self.document = None
        # place a button on the root window
        ttk.Button(self,text='Show Document',command=self.open_pageimg).grid(row=0, column=5, sticky=W)
        ttk.Button(self, text="Load Document", command=lambda: self.open_file(('Folders', '*'))).grid(row=0, column=0, sticky=W)
        ttk.Button(self, text="New Document", command=lambda: self.new_file(('PDF', '*.pdf'))).grid(row=0, column=1, sticky=W)
        #(self, text="confirmation",command=lambda: self.confirmation("confirmation"))
        Text(self, height=10, width=30)
    

    def errortouser(self,message):
        tk.messagebox.showerror("Error", message)
    
    def open_pageimg(self):
        pageimg = Window(self)
        pageimg.resizable(0,0)
        pageimg.geometry("1080x720")
        pageimg.grab_set()
        
    def update_image():
        global modified_image
        brightness = tk.brightness_slider.get()
        contrast = tk.contrast_slider.get()

        # Apply brightness and contrast adjustments to the original image
        modified_image = (tk.image_data + brightness) * contrast

        # Clip pixel values to the range [0, 255]
        modified_image = np.clip(modified_image, 0, 255).astype(np.uint8)

        # Convert modified_image array to ImageTk.PhotoImage and update the label
        modified_photo = ImageTk.PhotoImage(image=Image.fromarray(modified_image))
        tk.image_label.config(image=modified_photo)
        tk.image_label.image = modified_photo  # Keep reference to prevent garbage collection

    def checkopen(self):
        if self.document is not None:
            overwrite= self.confirmation("A file is currently open. Are you sure you want to open a new one?")
            if overwrite:
                return True
            else:
                return False
            
    def new_file(self,file):
        try:
            file = filedialog.askopenfile(mode='r', filetypes=[file])#io incomplete object

            if self.checkopen():
                newfile = Document(file.name,overwrite)
                self.document = newfile 
            if file.name[-4:] == '.pdf':
                if file.name[-4:] in os.path.listdir("server\storePDF"):
                    overwrite= self.confirmation("File already exists. Do you want to overwrite? Any changes may lead to a different output.")
                    if overwrite:
                        newfile = Document(file.name,overwrite)
                        self.document = newfile
                    
                #processpdf(file)
                file.close()

        except AttributeError:
            pass
        except Exception as e:
            self.errortouser("Error in opening file" + str(e))

    def open_file(self,file):
        file = filedialog.askopenfile(mode='r', filetypes=[file])#io incomplete object 
        if file.name[-1] != '/':
            #processfolder(file)
            #loadfile(file)
            file.close()
            

    def confirmation(self,message):
        msg_box = tk.messagebox.askquestion('Confirmation', message, icon='warning')
        if msg_box == 'yes':
            return True
        else:
            return False










            
class Window(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.geometry('300x100')
        self.title('Toplevel Window')
        #create a text frame for the text box with scrollbar
        #text_frame = ttk.Text(self).pack()


class Textframe(tk.Frame):
    def __init__(self, parent):
        pass




class ViewDocument(Document):
    def __init__(self,path,overwrite=False):
        super().__init__(path,overwrite=overwrite)
        self.location=""
        self.current_page = 0
        self.text=""
        self.sections=[]
        

    
    def read_page(self, page_number):
        self.current_page = page_number
        self.text = self.sections[page_number]
        return self.text



def driver():
    app = App()
    app.mainloop()
    
if __name__ == '__main__':
    driver()
#img= (Image.open("download.png"))