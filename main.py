import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


root = Tk()
root.geometry('400x100')

# UI ayalarmaları

label1 = Label(root, text='k : ')
label1.grid(row=0, column=0, padx=5, pady=10)

entry1 = Entry(root, width=5)
entry1.grid(row=0, column=1)

button1 = Button(root, text='Generate', command= lambda: generateGraph(comboBox1, comboBox2, entry1))
button1.grid(row=1, column=1)

label2 = Label(root, text='x :')
label2.grid(row=0, column=2)

label3 = Label(root, text='y :')
label3.grid(row=0, column=4, padx=10)

comboValues = ['Sports', 'Religious', 'Nature', 'Theatre', 'Shopping', 'Picnic']
comboBox1 = ttk.Combobox(root, value=comboValues, width=10)
comboBox1.grid(row=0, column=3)

comboBox2 = ttk.Combobox(root, value=comboValues, width=10)
comboBox2.grid(row=0, column=5)



df = pd.read_csv('Final-data.txt')

def generateGraph(comboBox1, comboBox2, entry1):
    xValue = comboBox1.get()
    yValue = comboBox2.get()
    n = int(entry1.get())
    df = pd.read_csv('Final-data.txt')


    # k-means algoritması burada çalışıyor
    km = KMeans(n_clusters= n)
    y_predicted = km.fit_predict(df[[xValue, yValue]])
    
    if 'cluster' in df.columns:
        df.drop('cluster', axis=1)
        df['cluster'] = y_predicted
    else:
        df['cluster'] = y_predicted


   
    # rastgele renk üretmek için döngü

    for i in range(1,n):
        globals()['df_{}'.format(i)] = df[df.cluster == i]
        
        
    import random
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))


    # girilen k degeri kadar her küme için ayrı dataframe olusturan döngü
    for i in range(1,n):
        plt.scatter(globals()['df_{}'.format(i)][xValue], globals()['df_{}'.format(i)][yValue], color = get_colors(1))



    # küme merkezlerini cizdiren alan
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color = 'purple', marker='*')
    plt.show()

    

    

    
    #dosya yazdırma islemleri

    f = open('sonuc.txt', 'w+')
    
    for inde in df.index:
        f.write('Kayit {}       kume: {}\n'.format(inde, df['cluster'][inde]))
        
    f.close()
    
    f = open('sonuc.txt', 'a')

    counts = df.groupby(['cluster']).count().to_string()
    f.write('\n'+counts)
    f.write('\n WCSS: {}\n'.format(km.inertia_))
    f.write('\n BCSS: {}\n'.format(km.n_iter_))
    f.close()



root.mainloop()

