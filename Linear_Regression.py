#MADE by zSuperDam (zsuperdam@gmail.com for reporting any problem)
import math
import numpy as np
import tkinter as tk
from scipy.stats import chi2
import matplotlib.pyplot as plt
import customtkinter as ctk


root = ctk.CTk()
root.title("Linear Regression by zSuperDam")

def drawplot(X, Y, S, m, q, Assex, Assey):
	obj1 = plt.errorbar(X, Y, yerr=S, fmt='o')
	x1 = np.linspace(min(X), max(X), num=len(X))
	y1 = m * x1 + q
	obj2 = plt.plot(x1, y1)
	plt.title('Fit')
	plt.xlabel(Assex)
	plt.ylabel(Assey)
	plt.savefig("output/fit.png")
	plt.show()

def drawresiduals(X, N, S, m, q):
	obj1 = plt.axhline(y=0, color='orange')
	Y = []
	for i in range(len(X)):
		Y.append(N[i]-m*X[i]-q)
	obj2 = plt.errorbar(range(1,len(X)+1), Y, yerr=S, fmt='o')
	plt.title('Residuals')
	plt.xlabel("Residuals")
	plt.ylabel("Data")
	plt.savefig("output/residuals.png")
	plt.show()

def media(dati):
	return sum(dati) / len(dati)

def button_clicked():
	
	fromfile = FromF.get()
	
	S=[]
	
	if fromfile == 'n':
		listX = Xinput.get(1.0, 'end')
		listY = Yinput.get(1.0, 'end')
		listS = Sinput.get(1.0, 'end')
		X = [float(numero) for numero in listX.split() if numero]
		Y = [float(numero) for numero in listY.split() if numero]
		S = [float(numero) for numero in listS.split() if numero]

	else:
		X = np.loadtxt("input/X.txt")
		Y = np.loadtxt("input/Y.txt")
		S = np.loadtxt("input/sigma.txt")
	
	if len(S)==0:
		S = [1] * len(X)
    
	sommaX = 0
	sommaX2 = 0
	sommaY = 0
	sommaS = 0
	sommaXY = 0
    
	for i in range(len(X)):
		sommaX += X[i] / (S[i] * S[i])
		sommaX2 += (X[i] * X[i]) / (S[i] * S[i])
		sommaY += Y[i] / (S[i] * S[i])
		sommaS += 1 / (S[i] * S[i])
		sommaXY += (X[i] * Y[i]) / (S[i] * S[i])

	delta = sommaS * sommaX2 - sommaX * sommaX
	m = (sommaS * sommaXY - sommaX * sommaY) / delta
	q = (sommaX2 * sommaY - sommaX * sommaXY) / delta

	scarti2 = 0
	for i in range(len(X)):
		scarti2 += (Y[i] - (m * X[i] + q)) * (Y[i] - (m * X[i] + q))
    
	sigmapost = math.sqrt(scarti2 / (len(X) - 2))
	
	sigmam = []
	sigmaq = []
	
	scelta = Spriori.get()
	confidence = float(CLinput.get())
	
	if scelta in ("no", "n"):
		sigmam = math.sqrt(sommaS / delta)
		sigmaq = math.sqrt(sommaX2 / delta)

	elif scelta in ("si", "sì", "y"):
		sommaX = 0
		sommaX2 = 0
		for i in range(len(X)):
			sommaX += X[i]
			sommaX2 += X[i] * X[i]
		deltap = len(X) * sommaX2 - sommaX * sommaX
		sigmam = sigmapost * math.sqrt(len(X) / deltap)
		sigmaq = sigmapost * math.sqrt(sommaX2 / deltap)
	
	mediaX = media(X)
	mediaY = media(Y)
	scX2 = 0
	scY2 = 0
	scXY = 0
	
	for i in range(len(X)):
		scX2 += (X[i] - mediaX) * (X[i] - mediaX)
		scY2 += (Y[i] - mediaY) * (Y[i] - mediaY)
		scXY += (X[i] - mediaX) * (Y[i] - mediaY)	
	
	rho = (scXY / math.sqrt(scX2 * scY2))
	rhosq = rho ** 2
	t = rho * math.sqrt(len(X) - 2) / math.sqrt(1 - rho * rho)	
	
	chi2s = 0
	for i in range(len(X)):
		chi2s += ((m * X[i] + q - Y[i])/S[i]) ** 2

	df = list(range(1, 101))

	# Calcolo dei valori del chi quadro
	chi2_vals = [chi2.ppf(confidence, i) for i in df]

	chi2t = chi2_vals[len(X)-2]
	
	if chi2s > chi2t:
		Xm = '<'
	else:
		Xm = '>'

	#OUTPUT on screen
	sigmapostl = tk.Entry(root)
	ml = tk.Entry(root)
	ql = tk.Entry(root)
	rhol = tk.Entry(root)
	tl = tk.Entry(root)
	chil = tk.Entry(root)
	rhosql = tk.Entry(root)
	
	sigmapostl.grid(row=7, column=0, padx=5, pady=5)
	ml.grid(row=7, column=1, padx=5, pady=5)
	ql.grid(row=7, column=2, padx=5, pady=5)
	rhol.grid(row=9, column=0, padx=5, pady=5)
	rhosql.grid(row=9, column=1, padx=5, pady=5)
	tl.grid(row=9, column=2, padx=5, pady=5)
	chil.grid(row=10, column=0, padx=5, pady=5)

	
	sigmapostl.insert(0, str(round(sigmapost,8)))
	ml.insert(0, "{} ± {}".format(round(m, 8), round(sigmam, 8)))
	ql.insert(0, "{} ± {}".format(round(q, 8), round(sigmaq, 8)))
	rhol.insert(0, str(round(rho, 10)))
	rhosql.insert(0, str(round(rho ** 2, 10)))
	tl.insert(0, str(t))
	chil.insert(0, "Xth-Xsp: {}{}{}".format(round(chi2t,3),Xm,round(chi2s,3)))
		
	
	tk.Label(root, text="Sigma Post:").grid(row=6, column=0, padx=5, pady=5)
	tk.Label(root, text="m:").grid(row=6, column=1, padx=5, pady=5)
	tk.Label(root, text="q:").grid(row=6, column=2, padx=5, pady=5)
	tk.Label(root, text="rho:").grid(row=8, column=0, padx=5, pady=5)
	tk.Label(root, text="rho^2:").grid(row=8, column=1, padx=5, pady=5)
	tk.Label(root, text="t:").grid(row=8, column=2, padx=5, pady=5)
	tk.Label(root, text="DOF: {}".format(len(X)-2)).grid(row=10, column=1, padx=5, pady=5)
	
	
	#OUTPUT on file
	with open("output/output.txt", "w") as output_file:
		output_file.write("Posteriori standard deviation: {}\n".format(sigmapost))
		output_file.write("m: {:>10} ± {}\n".format(m, sigmam))
		output_file.write("q: {:>10} ± {}\n".format(q, sigmaq))
		output_file.write("rho: {:>9}\n".format(rho))
		output_file.write("rho^2: {:>7}\n".format(rho**2))
		output_file.write("t: {:>11}\n".format(t))
		output_file.write("Xth-Xsp: {:>4} {} {} con {} gld e cl={}%.".format(chi2t, Xm, chi2s, len(X)-2, confidence*100))
	
	Xaxes = Assex.get()
	Yaxes = Assey.get()
	
	drawplot(X, Y, S, m, q, Xaxes, Yaxes)
	drawresiduals(X, Y, S, m, q)
		
	
#TKINTER

#griglia 3 colonne
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.columnconfigure(2, weight=1)

ctk.CTkLabel(root, text="X").grid(row=0, column=0, padx=5, pady=5)
ctk.CTkLabel(root, text="Y").grid(row=0, column=1, padx=5, pady=5)
ctk.CTkLabel(root, text="Sigma Y").grid(row=0, column=2, padx=5, pady=5)
ctk.CTkLabel(root, text="Use priori error (y,n)").grid(row=2, column=0, padx=5, pady=5)
ctk.CTkLabel(root, text="Confidence level").grid(row=3, column=0, padx=5, pady=5)
ctk.CTkLabel(root, text="Import from file? (y,n)").grid(row=4, column=0, padx=5, pady=5)
ctk.CTkLabel(root, text="Axes names:").grid(row=5, column=0, padx=5, pady=5)


#entry per inserire del testo
Xinput = ctk.CTkTextbox(root, height=320, width=250)
Yinput = ctk.CTkTextbox(root, height=320, width=250)
Sinput = ctk.CTkTextbox(root, height=320, width=250)
Spriori = ctk.CTkEntry(root)
CLinput = ctk.CTkEntry(root)
FromF = ctk.CTkEntry(root)
Assex = ctk.CTkEntry(root)
Assey = ctk.CTkEntry(root)


#entry nella griglia
Xinput.grid(row=1, column=0, padx=5, pady=5)
Yinput.grid(row=1, column=1, padx=5, pady=5)
Sinput.grid(row=1, column=2, padx=5, pady=5)
Spriori.grid(row=2, column=1, padx=5, pady=5)
CLinput.grid(row=3, column=1, padx=5, pady=5)
FromF.grid(row=4, column=1, padx=5, pady=5)
Assex.grid(row=5, column=1, padx=5, pady=5)
Assey.grid(row=5, column=2, padx=5, pady=5)

Spriori.insert(0, "n")
CLinput.insert(0, "0.95")
FromF.insert(0, "n")
Assex.insert(0, "X axis")
Assey.insert(0, "Y axis")

#bottone
button = ctk.CTkButton(master = root, text="Evaluate", command=button_clicked)

#bottone al centro della griglia
button.grid(row=2, column=2, padx=5, pady=5)

root.mainloop()