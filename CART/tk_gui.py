import numpy as np
import tkinter
from CART import cart
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def re_draw(deduct_tolerance, min_ct):
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)
    if chkBtnVar.get():
        if min_ct < 2: 
            min_ct = 2
        my_tree = cart.create_tree(re_draw.raw_dat, cart.model_leaf,
                                   cart.model_err, (deduct_tolerance, min_ct))
        y_hat = cart.create_forecast(my_tree, re_draw.testDat, 
                                     cart.model_tree_eval)
    else:
        my_tree = cart.create_tree(re_draw.raw_dat, 
                                   ops=(deduct_tolerance, min_ct))
        y_hat = cart.create_forecast(my_tree, re_draw.testDat)
    re_draw.a.scatter(re_draw.raw_dat[:, 0].flatten().A[0],
                      re_draw.raw_dat[:, 1].flatten().A[0],
                      s=5)  # use scatter for data set
    re_draw.a.plot(re_draw.testDat, y_hat, linewidth=2.0)  # use plot for y_hat
    re_draw.canvas.draw()


def get_inputs():
    try:
        min_ct = int(min_ct_entry.get())
    except:
        min_ct = 10
        print("enter Integer for min_ct")
        min_ct_entry.delete(0, tkinter.END)
        min_ct_entry.insert(0, '10')
    try:
        deduct_tolerance = float(deduct_tolerance_entry.get())
    except:
        deduct_tolerance = 1.0
        print("enter Float for deduct_tolerance")
        deduct_tolerance_entry.delete(0, tkinter.END)
        deduct_tolerance_entry.insert(0, '1.0')
    return min_ct, deduct_tolerance


def draw_new_tree():
    deduct_tolerance, min_ct = get_inputs()
    re_draw(deduct_tolerance, min_ct)


root = tkinter.Tk()

re_draw.f = Figure(figsize=(5, 4), dpi=100)  # create canvas
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.draw()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

tkinter.Label(root, text="min_ct").grid(row=1, column=0)
min_ct_entry = tkinter.Entry(root)
min_ct_entry.grid(row=1, column=1)
min_ct_entry.insert(0, '10')
tkinter.Label(root, text="deduct_tolerance").grid(row=2, column=0)
deduct_tolerance_entry = tkinter.Entry(root)
deduct_tolerance_entry.grid(row=2, column=1)
deduct_tolerance_entry.insert(0, '1.0')
tkinter.Button(root, text="re_draw", command=draw_new_tree).grid(row=1,
                                                                 column=2,
                                                                 rowspan=3)
chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

re_draw.raw_dat = np.mat(cart.load_data_set('../Data/sine.txt'))
re_draw.testDat = np.arange(min(re_draw.raw_dat[:, 0]),
                            max(re_draw.raw_dat[:, 0]),
                            0.01)
re_draw(1.0, 10)
root.mainloop()
