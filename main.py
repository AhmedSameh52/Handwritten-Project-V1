from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk, Image
from tkinter import ttk
import tkinter as tk
from PIL import ImageGrab
root = Tk()
root.title("White Board")
root.geometry ( "1050x570+150+50" )
root.configure (bg = "#f2f3f5")
root. resizable(False,False)

current_x = 0
current_y = 0
color = 'black'

def locate_xy(work):
    global current_x, current_y
    current_x = work.x
    current_y = work.y

def addLine(work):
    global current_x, current_y

    canvas.create_line((current_x,current_y,work.x,work.y),width = 10, fill = color, capstyle=ROUND, smooth=True)
    current_x, current_y = work.x, work.y

def show_color(new_color):
    global color
    color = new_color

def new_canvas():

    canvas.delete('all')
    display_pallete()

#icon
image_icon = ImageTk.PhotoImage(file = "gemy.png")
root.iconphoto(False, image_icon)

# color_box = ImageTk.PhotoImage(file = "boardColor.png")
# Label(root, image = color_box, bg = "#f2f3f5").place(x = 10, y = 20)

eraser = ImageTk.PhotoImage(file = "eraser.png")
Button(root, image = eraser, bg = "#f2f3f5", command = new_canvas).place(x = 750, y = 10)

colors = Canvas(root, bg = "#ffffff", width = 300, height = 37, bd = 0)
colors.place(x = 400, y = 10)

def display_pallete():
    id = colors.create_rectangle((10,10,30,30),fill = "black")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('black'))

    id = colors.create_rectangle((40,10,60,30),fill = "gray")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('gray'))

    id = colors.create_rectangle((70,10,90,30),fill = "brown4")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('brown4'))

    id = colors.create_rectangle((100,10,120,30),fill = "red")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('red'))

    id = colors.create_rectangle((130,10,150,30),fill = "orange")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('orange'))

    id = colors.create_rectangle((160,10,180,30),fill = "yellow")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('yellow'))

    id = colors.create_rectangle((190,10,210,30),fill = "green")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('green'))

    id = colors.create_rectangle((220,10,240,30),fill = "blue")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('blue'))

    id = colors.create_rectangle((250,10,270,30),fill = "purple")
    colors.tag_bind(id, '<Button-1>', lambda x: show_color('purple'))

display_pallete()

canvas = Canvas(root, width = 900, height = 400, background="white", cursor="hand2")
canvas.pack()
canvas.place(x=75,y=75)

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', addLine)



root.update()
# Create a new image with the same width and height as the canvas.
# Define the area you want to capture
x = root.winfo_rootx() + canvas.winfo_x()
y = root.winfo_rooty() + canvas.winfo_y()
x1 = x + canvas.winfo_width()
y1 = y + canvas.winfo_height()
capture_region = (x, y, x1, y1)
print(x)




# Define the function to capture the snapshot and send it to the machine learning model
def capture_and_send():
    # Use the canvas's postscript() method to render the canvas to the new image.
    # Capture the snapshot of the defined area
    img = ImageGrab.grab(bbox=capture_region)
    
    # Save the snapshot to disk
    img.save("captured_snapshot.jpg")

button = tk.Button(root, text="Capture Snapshot", command=capture_and_send)

button.pack()
button.place(x = 525, y = 500)

root. mainloop()

