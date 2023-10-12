import time
import tkinter as tk
from tkinter import NORMAL, DISABLED, ACTIVE

import numpy as np
import tkinter.messagebox as msg
# 调动TK创建主窗口
root_window=tk.Tk()
# 设置窗口大小
root_window.geometry('450x500')

global person,computer
person=0
computer=0
# 窗口名字
root_window.title("井字棋--人机博弈")
# 记录下棋的步数
nums = [0,1,2,3,4,5,6,7,8]
l1=tk.Label(root_window,text="player1 : X")
l2=tk.Label(root_window,text="player2 : O")
l1.grid(row=1,column=0)
l2.grid(row=2,column=0)
# 按钮被点击时执行函数
def click_button1():
# 步数
    return root_window.quit
# 棋盘的改变,记录一些重要数据
panels = ["panel"]*9
empty=''
step=0
out=0
def click_button(num):
    global step, empty, nums
    if num in nums:
        nums.remove(num)
        if step%2==0:
            empty ='X'
            panels[num]=empty
        elif step%2!=0:
            empty = 'O'
            panels[num]=empty
        ButtonList[num].config(text = empty)
        step = step+1
        sign = empty
        if(win(panels,sign) and sign=='X'):
            msg.showinfo("Result","Player1 wins")
            out=1
        elif(win(panels,sign) and sign=='O'):
            msg.showinfo("Result","Player2 wins")
            out=1
    if (step%2==0 and computer==1) or(step%2==1 and computer==0):
        AI()

    if(step>7 and win(panels,'X')==False and win(panels,'O')==False):
        msg.showinfo("Result","Match Tied")
        out=1
# 判断胜利条件
def win(panels,sign):

    return ((panels[0] == panels[1] == panels [2] == sign)
            or (panels[0] == panels[3] == panels [6] == sign)
            or (panels[0] == panels[4] == panels [8] == sign)
            or (panels[1] == panels[4] == panels [7] == sign)
            or (panels[2] == panels[5] == panels [8] == sign)
            or (panels[2] == panels[4] == panels [6] == sign)
            or (panels[3] == panels[4] == panels [5] == sign)
            or (panels[6] == panels[7] == panels [8] == sign))
ButtonList = [0,0,0,0,0,0,0,0,0]
# 按钮 退出功能
# button=tk.Button(root_window,text="退出游戏",command=root_window.quit)\

def order():
     pass
def start():
    global computer,step,nums,panels,empty
    nums = [0,1,2,3,4,5,6,7,8]
    panels = ["panel"]*9
    empty=''
    step=0
    for i in range(0,9):
            ButtonList[i]=tk.Button(root_window,width=15,height=7,command=lambda num=i: click_button(num))
    x=1
    y=1
    for i in range(0,9):
       if y<3:
           ButtonList[i].grid(row=x,column=y)
           y=y+1
       else :
          ButtonList[i].grid(row=x,column=y)
          x=x+1
          y=1
          i=i+1
    for i in ButtonList:
        i.config(text='')
    message1= msg.askyesno("提示","您是否先手？")
    if message1==1:
        msg.showinfo("提示","该您了")
    elif message1==0:
        msg.showinfo("提示","等待对方下棋")
        computer=1
        AI()
    while((step%2==0 and message1==0) or(step%2==1 and message1==1) ):
        AI()
        break;
def AI():
    p=panels.copy();
    for i in nums:
        if step%2==0:
            p[i]='X';
            if(win(p,sign='X')):
                click_button(i);
                return
            p[i]='O';
            if(win(p,sign='O')):
                click_button(i);
                return
            p[i]="";
        if step%2!=0:
            p[i]='O';
            if(win(p,sign='O')):
                click_button(i)
                return
            p[i]='X'
            if(win(p,sign='X')):
                click_button(i);
                return
            p[i]="";
    best=[4,0,2,6,8,1,3,5,7]
    for j in best:
        if j in nums:
            click_button(j)
            break

button=tk.Button(root_window,text="退出游戏",command=click_button1())
button2=tk.Button(root_window,text="开始游戏",command=lambda :start())
button2.grid(row=5,column=3)
button.grid(row=5,column=1)
# 窗口显示状态
root_window.mainloop()

