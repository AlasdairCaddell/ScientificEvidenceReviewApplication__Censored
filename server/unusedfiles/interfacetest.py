from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageEnhance, ImageTk
from visualise import *
from pageobj import *

def enforcepath(path):
    return path.replace("/","\\")
def colourfromclass(classification):
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
    print(msg_box)
    if msg_box == 'yes':
        return True
    else:
        return False
    
def show_popup(message):
    messagebox.showinfo("Warning", message)    
    
    
def update_image():
    global interfaceimg,interfacedoc,interfacecurrentpage,adapted_img,XBLUR,YBLUR,DILATIONITERATION,THRESHOLD,displayimgfortraining,ifboundingbox

    if interfacecurrentpage==0:
        previous_button.config(bg="gray")
    else:
        previous_button.config(bg=default_color)
        
    if interfacecurrentpage==interfacedoc.numpages:
        next_button.config(bg="gray")
    else:
        next_button.config(bg=default_color)
        
    if displayimgfortraining:    
        struct,dilate=thresholding(np.asarray(interfaceimg),blur=(XBLUR,YBLUR),threshold=THRESHOLD,rect=(3,3),dilate=DILATIONITERATION)
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colourdilate=cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            x, y, w, h =cv2.boundingRect(cnt)
            adapted_img=boundingbox(colourdilate,[y,x,w,h],(0,0,255))
       
    else:
        adapted_img = np.asarray(interfaceimg)
    
    
        
    if ifboundingbox:
        for item in interfacedoc.pages[interfacecurrentpage].getblockinformation():
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
            launch_processing_button.config(bg=default_color)
            previous_button.config(bg=default_color)
            next_button.config(bg=default_color)
            xblur_slider.config(bg="gray",label="X Blur")
            yblur_slider.config(bg="gray",label="Y Blur")
            dilation_slider.config(bg="gray",label="Dilation")
            threshold_slider.config(bg="gray",label="Threshold")
        show_popup("Please remember to use the sliders before processing, otherwise the default values will be used.")
        update_image()
    except:
        pass


def previous():
    global interfaceimg,interfacedoc,interfacecurrentpage
    if interfaceimg and interfacecurrentpage>0:
        interfacecurrentpage-=1
        interfaceimg = Image.open(interfacedoc.pages[interfacecurrentpage].file)
        #interfaceimg = interfaceimg.transpose(Image.FLIP_LEFT_RIGHT)
        update_image()


def next():
    global interfaceimg,interfacedoc,interfacecurrentpage
    if interfaceimg and interfacecurrentpage<interfacedoc.numpages+1:
        try:
            interfacecurrentpage+=1
            interfaceimg = Image.open((interfacedoc.pages[interfacecurrentpage]).file)
            update_image()
        except:
            pass


def load():
    global interfaceimg,interfacedoc,interfacecurrentpage
    name = filedialog.askopenfilename(initialdir="Untitled", title="Select file",
                                        filetypes=(("pickle files", "*.pkl"), ("all files", "*.*")))
    if name:
        interfacedoc = unpickle(name)
        interfacecurrentpage=0
        pageobj = interfacedoc.pages[0]
        interfaceimg= Image.open(pageobj.file)
        update_image()

def save():
    global interfaceimg,interfacedoc
    if interfaceimg:
        name = filedialog.askdirectory()
        filename=(interfacedoc.name)+".pkl"
        savename=os.path.join(name,filename)
        if name:
            savepickle(interfacedoc,savename)
            
def launch_processing():
    global interfaceimg,interfacedoc,ifboundingbox,TRAIN
    if interfaceimg:
        if TRAIN:
            checkifoverride=interfacedoc.checktrainheaderfooter()
            check=False
            show_popup("Resizing the window will cause the program to crash. Please do not resize the window.")
            print("checkoverride: ",checkifoverride)    
            if checkifoverride:
                check=confirmation("You have already trained the header and footer. Do you want to override?")
            
            interfacedoc.trainheaderfooter(override=check)

        locations,classification=interfacedoc.headerfooterdriver()
        interfacedoc.addblockstopages(locations,classification)

        
        unlikelypages=interfacedoc.testtemplatetopage()
        interfacedoc.addblockstopages(locations,classification)
        for likelypage in interfacedoc.pages:
            if likelypage.pagenum in unlikelypages:
                print("unlikelypage: ",likelypage.pagenum)
                likelypage.extractdata(flag=1)
            else:
                likelypage.extractdata()
        
        
        ifboundingbox=True

        
        update_image()
            
            
def change_xblur(var):
    global interfaceimg,XBLUR
    XBLUR =int(var)
    update_image()

def change_yblur(var):
    global interfaceimg,YBLUR
    YBLUR =int(var)
    update_image()
    
def change_dilation(var):
    global interfaceimg,DILATIONITERATION
    DILATIONITERATION =int(var)
    update_image()
    
def change_threshold(var):
    global interfaceimg,THRESHOLD
    THRESHOLD =int(var)
    update_image()

def tickbox():
    global TRAIN
    TRAIN = not TRAIN

def toggleimg():
    global displayimgfortraining
    displayimgfortraining = not displayimgfortraining
    if interfaceimg:
        update_image()
    

    
    
root = Tk()
root.title("Image Editor")
root.geometry('1700x900')
default_color = root.cget('bg')

displayimgfortraining=False
interfaceimg = None
ifboundingbox=None
contrast_val = 50
TRAIN=False



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
root.mainloop()