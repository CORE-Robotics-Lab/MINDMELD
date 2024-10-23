from tkinter import *
import pickle
import os
import numpy as np


x=[]
y=[]
class Test:
   def __init__(self,name,train_test,goal=[20,20]):
       self.b1="up"
       self.name=name
       self.goal=goal
       self.xold=None
       self.yold=None
       self.x=[]
       self.y=[]
       self.train_test=train_test

   def create_circle(self,x, y, r, canvasName):  # center coordinates, radius
       x0 = x - r
       y0 = y - r
       x1 = x + r
       y1 = y + r
       return canvasName.create_oval(x0, y0, x1, y1)

   def test(self,obj):
       self.drawingArea=Canvas(obj)
       self.root=obj
       self.create_circle(self.goal[0], self.goal[1], 10, self.drawingArea)
       self.create_circle(20, 20, 5, self.drawingArea)

       self.drawingArea.pack()
       self.drawingArea.bind("<Motion>",self.motion)
       self.drawingArea.bind("<ButtonPress-1>",self.b1down)
       self.drawingArea.bind("<ButtonRelease-1>",self.b1up)

   def b1down(self,event):
       self.b1="down"
   def b1up(self,event):
       self.b1="up"
       self.xold=None
       self.yold=None

       if len(self.x)>0:
           with open(os.path.join('Data',self.train_test,'Dagger-rollouts',self.name), 'wb') as f:
               pickle.dump([self.x,self.y], f)
           self.root.quit()
   def motion(self,event):
      if self.b1=="down":
           if self.xold is not None and self.yold is not None:
               event.widget.create_line(self.xold,self.yold,event.x,event.y,fill="red",width=4,smooth=TRUE)
           self.xold=event.x
           self.yold=event.y
           self.x.append(self.xold)
           self.y.append(self.yold)
           #print(self.x,self.y)

def get_trial(name,goal,train_test):
    root = Tk()
    root.wm_title("Test")
    v = Test(name,train_test,goal=goal)
    v.test(root)
    root.mainloop()



if __name__=="__main__":
    get_trial("trial1")