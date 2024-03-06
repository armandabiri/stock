from tkinter import *
import tkinter.font
from tkinter.colorchooser import *
from tkinter import simpledialog
from PIL import Image as im
from PIL import ImageTk,ImageDraw
import os
from tkinter.filedialog import *
from sklearn.preprocessing import MinMaxScaler

root=Tk()
root.geometry("800x600")
root.title("Paint")

icon_path=os.path.abspath('python/paint_icon3.png')
icon_img=im.open(icon_path)
icon=ImageTk.PhotoImage(icon_img)
root.iconphoto(False,icon)

class PaintApp:
    text_font=StringVar()
    text_size=IntVar()
    bold_text=StringVar()
    italic_text=StringVar()

    drawing_tool=StringVar()

    stroke_size=IntVar()
    fill_color=StringVar()
    stroke_color=StringVar()

    left_but='up'
    x_pos,y_pos=None,None

    x1_line_pt,y1_line_pt,x2_line_pt,y2_line_pt=None,None,None,None
    image=None

    my_image=im.new("RGB",(800,600),(255,255,255))
    draw=ImageDraw.Draw(my_image)

    @staticmethod
    def quit():
        root.destroy()

    def save_file(self,event=None):

        my_file=asksaveasfile(mode='w',defaultextension='.png',filetypes=[('png files','*.png')],
                              initialdir='/',parent=root)
        if my_file is not None:
            path=os.path.abspath(my_file.name)
            self.my_image.save(path)

    def open_file(self,event=None):
        my_file_name=askopenfilename(parent=root,filetypes=[('png files','*.png'),])
        if my_file_name:
            my_pic=im.open(my_file_name)
            self.image=ImageTk.PhotoImage(my_pic)
            self.drawing_area.create_image(0,0,anchor='nw',image=self.image)


    def make_menu_bar(self):
        the_menu=Menu(root)

        #File#
        file_menu=Menu(the_menu,tearoff=0)
        file_menu.add_command(label='Open',command=self.open_file)
        file_menu.add_command(label='Save',command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label='Quit',command=self.quit)

        the_menu.add_cascade(label='File',menu=file_menu)
        #File#

        #Font#
        font_menu=Menu(the_menu,tearoff=0)

        font_type_submenu=Menu(font_menu,tearoff=0)
        font_type_submenu.add_radiobutton(label='Times',variable=self.text_font,value='Times')
        font_type_submenu.add_radiobutton(label='Courier',variable=self.text_font,value='Courier')
        font_type_submenu.add_radiobutton(label='Arial',variable=self.text_font,value='Arial')
        font_menu.add_cascade(label='Font Type',menu=font_type_submenu)

        font_size_submenu=Menu(font_menu,tearoff=0)
        font_size_submenu.add_radiobutton(label='10',variable=self.text_size,value=10)
        font_size_submenu.add_radiobutton(label='15',variable=self.text_size,value=15)
        font_size_submenu.add_radiobutton(label='20',variable=self.text_size,value=20)
        font_size_submenu.add_radiobutton(label='25',variable=self.text_size,value=25)
        font_menu.add_cascade(label='Font Size',menu=font_size_submenu)

        font_menu.add_checkbutton(label='Bold',variable=self.bold_text, onvalue='bold',offvalue='normal')
        font_menu.add_checkbutton(label='Italic',variable=self.italic_text, onvalue='italic',offvalue='roman')

        the_menu.add_cascade(label='Font',menu=font_menu)

        #Tool#
        tool_menu=Menu(the_menu,tearoff=0)
        tool_menu.add_radiobutton(label='Pencil',variable=self.drawing_tool,value='pencil')
        tool_menu.add_radiobutton(label='Line',variable=self.drawing_tool,value='line')
        tool_menu.add_radiobutton(label='Arc',variable=self.drawing_tool,value='arc')
        tool_menu.add_radiobutton(label='Oval',variable=self.drawing_tool,value='oval')
        tool_menu.add_radiobutton(label='Rectangle',variable=self.drawing_tool,value='rectangle')
        tool_menu.add_radiobutton(label='Text',variable=self.drawing_tool,value='text')

        the_menu.add_cascade(label='Tool',menu=tool_menu)
        #Tool#

        #Color#
        color_menu=Menu(the_menu,tearoff=0)
        color_menu.add_command(label='Fill',command=self.pick_fill)
        color_menu.add_command(label='Stroke',command=self.pick_stroke)
        stroke_width_submenu=Menu(color_menu,tearoff=0)
        stroke_width_submenu.add_radiobutton(label='1',variable=self.stroke_size,value=1)
        stroke_width_submenu.add_radiobutton(label='2',variable=self.stroke_size,value=2)
        stroke_width_submenu.add_radiobutton(label='3',variable=self.stroke_size,value=3)
        stroke_width_submenu.add_radiobutton(label='4',variable=self.stroke_size,value=4)
        stroke_width_submenu.add_radiobutton(label='5',variable=self.stroke_size,value=5)
        color_menu.add_cascade(label='Stroke Size',menu=stroke_width_submenu)

        the_menu.add_cascade(label='Color',menu=color_menu)
        #Color#

        root.config(menu=the_menu)

    def pick_fill(self,event=None):
        fill_color=askcolor(initialcolor='black',parent=self.drawing_area,title='Pick Fill Color')
        if None not in fill_color:
            self.fill_color.set(fill_color[1])

    def pick_stroke(self,event=None):
        stroke_color=askcolor(initialcolor='black',parent=self.drawing_area,title='Pick Stroke Color')
        if None not in stroke_color:
            self.stroke_color.set(stroke_color[1])

    def left_but_down(self,event=None):
        self.left_but='down'
        self.x1_line_pt=event.x #WRT CANVAS WIDGET
        self.y1_line_pt=event.y
        self.x_pos=self.x1_line_pt
        self.y_pos=self.y1_line_pt

    def left_but_up(self,event=None):
        self.left_but='up'
        self.x2_line_pt=event.x #WRT CANVAS WIDGET
        self.y2_line_pt=event.y

        if(self.drawing_tool.get()=='line'):
            self.line_draw(event)
        elif(self.drawing_tool.get()=='pencil'):
            self.pencil_draw(event)
        elif(self.drawing_tool.get()=='oval'):
            self.oval_draw(event)
        elif(self.drawing_tool.get()=='rectangle'):
            self.rectangle_draw(event)
        elif(self.drawing_tool.get()=='text'):
            self.text_draw(event)
        elif(self.drawing_tool.get()=='arc'):
            self.arc_draw(event)

    def motion(self,event=None):
        if(self.drawing_tool.get()=='pencil'):
            self.pencil_draw(event)

    def pencil_draw(self,event=None):
        if(self.left_but=='down'):
            event.widget.create_line(self.x_pos,self.y_pos,event.x,event.y,
                                     fill=self.stroke_color.get(),width=self.stroke_size.get(),smooth=True)

            self.draw.line([(self.x_pos,self.y_pos),(event.x,event.y)],fill=self.stroke_color.get())
            self.x_pos=event.x
            self.y_pos=event.y


    def line_draw(self,event=None):
        if None not in (self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt):
            event.widget.create_line(self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt,
                                     smooth=True,fill=self.stroke_color.get(),width=self.stroke_size.get())

            self.draw.line([(self.x1_line_pt,self.y1_line_pt),(self.x2_line_pt,self.y2_line_pt)],
                            fill=self.stroke_color.get())

    def rectangle_draw(self,event=None):
        if None not in (self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt):
            event.widget.create_rectangle(self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt,
                                     outline=self.stroke_color.get(),fill=self.fill_color.get(),
                                     )

            self.draw.rectangle([(self.x1_line_pt,self.y1_line_pt),(self.x2_line_pt,self.y2_line_pt)],
                                outline=self.stroke_color.get(),fill=self.fill_color.get())

    def oval_draw(self,event=None):
        if None not in (self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt):
            event.widget.create_oval(self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt,
                                     outline=self.stroke_color.get(),fill=self.fill_color.get(),
                                     )

            self.draw.ellipse([(self.x1_line_pt,self.y1_line_pt),(self.x2_line_pt,self.y2_line_pt)],
                                outline=self.stroke_color.get(),fill=self.fill_color.get())

    def text_draw(self,event=None):
        if None not in (self.x1_line_pt,self.y1_line_pt):
            text_font=tkinter.font.Font(family=self.text_font.get(),size=self.text_size.get(),
                                        weight=self.bold_text.get(),
                                        slant=self.italic_text.get())
            user_text=simpledialog.askstring('Input','Enter Text',parent=root)
            if user_text is not None:
                event.widget.create_text(self.x1_line_pt,self.y1_line_pt,fill=self.fill_color.get(),
                                         font=text_font,text=user_text)

    def arc_draw(self,event=None):
        if None not in (self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt):
            event.widget.create_arc(self.x1_line_pt,self.y1_line_pt,self.x2_line_pt,self.y2_line_pt,
                                    fill=self.stroke_color.get(),
                                    width=self.stroke_size.get(),extent=150,start=30,style=ARC)
            self.draw.arc([(self.x1_line_pt,self.y1_line_pt),(self.x2_line_pt,self.y2_line_pt)],
                           fill=self.stroke_color.get(),start=30,end=150)

    def __init__(self):
        self.drawing_area=Canvas(root,width=800,height=600,bg='white')
        self.drawing_area.pack()
        self.text_font.set('Times')
        self.text_size.set(20)
        self.bold_text.set('normal')
        self.italic_text.set('roman')
        self.drawing_tool.set('line')
        self.stroke_size.set(3)
        self.fill_color.set('#000000')
        self.stroke_color.set('#000000')
        self.make_menu_bar()
        self.drawing_area.focus_set()
        self.drawing_area.bind("<Button-1>",self.left_but_down)
        self.drawing_area.bind("<B1-Motion>",self.motion)
        self.drawing_area.bind("<ButtonRelease-1>",self.left_but_up)

paint_app=PaintApp()
root.mainloop()
