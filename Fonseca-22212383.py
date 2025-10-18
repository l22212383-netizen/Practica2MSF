"""
Práctica 2: Sistema cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Ivan De Jesus Fonseca Diaz
Número de control: 22212383
Correo institucional: l22212383@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiologicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""
# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx', header=None))

# Datos de la simulación
x0, t0, tF, dt, w, h = 0, 0, 10, 1E-3, 10, 5
N = round((tF-t0)/dt) + 1
t = np.linspace(t0, tF, N)
u = np.reshape(signal.resample(u, len(t)),-1)

# Componentes del circuito RLC y funcion de transferencia
def cardio(R,L,C,Z):
    num = [R*L,R*Z]
    den = [C*R*Z*L,Z*L+R*L,R*Z]
    sys = ctrl.tf(num,den)
    return sys


# Normotenso
R, L, C, Z = 0.95, 0.01, 1.5, 0.033
sysnormo = cardio(R,L,C,Z)
print(f"Individuo normotenso: {sysnormo}")

# Hipotenso
R, L, C, Z = 0.6, 0.005, 0.25, 0.02
syshipo = cardio(R,L,C,Z)
print(f"Individuo normotenso: {syshipo}")

# Hipertenso
R, L, C, Z = 1.4, 0.02, 2.5, 0.05
syshiper = cardio(R,L,C,Z)
print(f"Individuo normotenso: {syshiper}")

#Respuestas en lazo abierto
clr1 = np.array([230, 39, 39])/255
clr2 = np.array([0, 0, 0])/255
clr3 = np.array([67, 0, 255])/255
clr4 = np.array([22, 97, 14])/255
clr5 = np.array([250, 129, 47])/255
clr6 = np.array([145, 18, 188])/255

_,Pp0 = ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1 = ctrl.forced_response(syshipo,t,u,x0)
_,Pp2 = ctrl.forced_response(syshiper,t,u,x0)

fg1 = plt.figure()
plt.plot(t,Pp0,'-',linewidth=1, color = clr1, label='Pp(t): Normotenso')
plt.plot(t,Pp1,'-',linewidth=1, color = clr2, label='Pp(t): Hipotenso')
plt.plot(t,Pp2,'-',linewidth=1, color = clr3, label='Pp(t): Hipertenso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t[s]')
plt.ylabel('Pp(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema cardiovascular python.png',dpi=600,bbox_inches='tight')
fg1.savefig('sistema cardiovascular python.pdf',bbox_inches='tight')


def controlador(kP,kI):
    Cr = 1E-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI,denPI)
    return PI

PI = controlador(10,3928.62362169833)
X = ctrl.series(PI,syshipo)
hipo_PI = ctrl.feedback(X,1,sign=-1)


PI = controlador(100,3937.667589046295)
X = ctrl.series(PI,syshiper)
hiper_PI = ctrl.feedback(X,1,sign=-1)

_,Pp3 = ctrl.forced_response(hipo_PI,t,u,x0)
_,Pp4 = ctrl.forced_response(hiper_PI,t,u,x0)

fg2 = plt.figure()
plt.plot(t,Pp0,'-',linewidth=1, color = clr1, label='Pp(t): Normotenso')
plt.plot(t,Pp1,'-',linewidth=1, color = clr2, label='Pp(t): Hipotenso')
plt.plot(t,Pp2,'-',linewidth=1, color = clr3, label='Pp(t): Hipertenso')
plt.plot(t,Pp3,':',linewidth=2, color = clr4, label='Pp(t): Hipotenso-PI')
plt.plot(t,Pp4,'-.',linewidth=2, color = clr5, label='Pp(t): Hipertenso-PI')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('t[s]')
plt.ylabel('Pp(t) [V]')
plt.legend(bbox_to_anchor=(0.5,-0.2),loc='center',ncol=5)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema cardiovascular PI python.png',dpi=600,bbox_inches='tight')
fg2.savefig('sistema cardiovascular PI python.pdf',bbox_inches='tight')

