#!/usr/bin/env python
# coding: utf-8

# # REPORTE 2 

# Estudiante: Greysi Arrelucea Castañeda 

# ## Lectura

# El artículo La Historia Monetaria y Fiscal del Perú, 1960-2017 busca explicar los períodos de inflación e hiperinflación que precedieron a la estabilización económica iniciada en la década de 1990 y mantenida hasta la actualidad. De esta manera, responde a la siguiente pregunta de investigación: ¿Cómo se explican los períodos de inflación crónica, hiperinflación y estabilización (1970-1990) en el Perú desde el lente del enfoque monetarista? Así, la hipótesis principal de la investigación es que los periodos de crisis económica acontecidos antes del gobierno de Alberto Fujimori reflejan la necesidad fiscal de impuestos inflacionarios en un régimen de dominancia fiscal de la política monetaria y que precisamente la etapa de estabilización corresponde a un periodo de independencia de la política monetaria y moderación fiscal. Para demostrar estas hipótesis, realiza un análisis con dos ejercicios contables: un ejercicio de contabilidad del crecimiento y un ejercicio de contabilidad fiscal. Finalmente, concluye con una reflexión sobre el papel determinante del aprendizaje social atravesado por la población peruana en el cambio de actitud respecto a la política económica.
#  
# El enfoque monetarista de la investigación es complementado con una perspectiva institucional en tanto se hace una revisión de las políticas adoptadas antes, durante y después de la estanflación y sus efectos en la intensificación de la crisis económica. La principal fortaleza de esta propuesta es que permite una comprensión holística de las crisis inflacionarias, pues además de analizarlas cuantitativamente con los ejercicios contables antes mencionados, también las contextualiza con las decisiones políticas que marcaron cada etapa. Adicionalmente, el enfoque permite la inclusión de gráficos y tablas estadísticas, complementados con la definición de los conceptos más importantes, lo que facilita significativamente la comprensión del comportamiento de la inflación peruana a lo largo de los años. Por otro lado, el enfoque tiene una debilidad marcada por una situación externa: los cambios en el tipo de cambio real durante los periodos de crisis y estabilización hicieron que ciertos datos oficiales no estén debidamente contabilizados, de manera que este vacío debe tomarse en cuenta al momento de analizar la investigación.
#  
# A pesar de ello, indudablemente, este artículo tiene una sustancial contribución teórica y social. Por un lado, mejora la comprensión teórica de la crisis económica peruana más grave del último siglo, situando las dificultades fiscales del Perú en el contexto de un Estado pequeño y aún en desarrollo. Establece los patrones que desencadenaron e intensificaron la inflación, como los déficits fiscales cíclicos que precedieron los años de crisis. Así, permite entender las razones por las que los peruanos han asociado al intervencionismo con una imagen negativa y por las que los gobernantes políticos, desde la década de los 2000, han promovido políticas más ligadas al libre mercado. Precisamente, el estudio también tiene una ineludible contribución social en tanto contribuye con el conocimiento de los errores cometidos en el pasado de manera que no sean repetidos en el presente. Además, el estudio puede ser percibido como un pedido a la ciudadanía de mantenerse alerta a las políticas económicas propuestas por los gobernantes, pues abre el debate de que las propuestas populistas también han llevado e intensificado las crisis macroeconómicas del país.
#  
# Finalmente, además de las cuestiones ya propuestas por el artículo, propongo ciertos pasos para avanzar en la investigación. En primer lugar, estudiar las crisis inflacionarias desde un enfoque no monetarista, de manera que se puedan contrastar los hallazgos de ambos estudios. Paredes y Sanchs (1999) muestran la importancia de un enfoque estructural que explique el periodo peruano de hiperinflación, de modo que se pueda visibilizar con mayor detalle el impacto de aquellos fenómenos no económicos sobre el desencadenamiento de las crisis económicas. Los mismos autores del presente artículo reconocen que ciertas luchas políticas han llevado al establecimiento de propuestas populistas no favorables para la economía. Por ello, considero que un segundo paso a seguir es medir cuantitativamente la influencia de la opinión pública en las políticas macroeconómicas, aunque esta cuestión es abordada cualitativamente en el artículo, considero que sería muy beneficioso cuantificar esta cuestión de modo que se pueda disponer de datos más exactos. 
# 
# 
# Bibliografía
# 
# Paredes, C. y Sachs, J. (1999). Estabilización y crecimiento en el Perú. GRADE, Lima. 
# https://www.grade.org.pe/wp-content/uploads/Estabili
# 

# ## Código en Python: 
# #### Greysi Arrelucea (20200279) y Roxana Rodriguez Pilco (20200373)

# In[1]:


import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel


# #### a. El Modelo Ingreso-Gasto: la Curva IS

# ##### a.1) Utilizando las ecuaciones derive paso a paso la curva IS matemáticamente (a partir de la condición de equilibrio Y = DA).

# ###### Derivación de la curva IS
# La curva IS se deriva de la igualdad entre el ingreso ($Y$) y la demanda agregada ($DA$):
# $$Y=C+I+G+X-M$$
# 
# Entonces, considerando las siguientes ecuaciones:
# 
# $$C= C_0 + bY^d$$
# 
# $$I= I_0 + hr$$
# 
# $$G= G_0$$
# 
# $$T= tY$$
# 
# $$X= X_0$$
# 
# $$M= mY^d$$
# 
# Para llegar al equilibrio Ahorro-Inversión, se debe restar la tributación ($T$) de ambos miembros de la igualdad:
# 
# $$Y-T = C+I+G+X-M-T$$ 
# $$Y^d=C+I+G+X-M-T$$
# $$Y^d-C-G-X+M+T = I$$
# 
# Esta igualdad se puede reescribir de la siguiente forma:
# 
# $$(Y^d-C)+(T-G)+(M-X)=I$$
# 
# Las tres partes de la derecha constituyen los tres componentes del ahorro total($S$): ahorro privado ($Sp=Y^d-C$) , ahorro del gobierno ($Sg=T-G$) y ahorro externo ($Se=M-X$):
# 
# $$S=Sp+Sg+Se$$
# 
# De modo que, el ahorro total ($S$) es igual a la inversión($I$):
# 
# $$Sp+Sg+Se=I$$
# 
# $$S(Y)=I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$Sp+Sg+Se=I_0-hr$$
# 
# $$(Y^d-C_0-bY^d)+(T-G_0)+(mY^d-X_0)=I_0-hr$$
# 
# Considerando las observaciones anteriores sobre los componentes de la condición de equilibrio ($Y$):
# 
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# Entonces, la curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
# $$ hr = (C_0 + G_0 + I_0 + X_0) - [1 - (b - m)(1 - t)]Y $$
# 
# luego, la sensibilidad de inversión ante la tasa de interés $h$ pasa al otro lado de la ecuación dividiendo todos los componentes y resulta:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Siendo $ B_0 = C_0 + G_0 + I_0 + X_0  $ el intercepto y la pendiente $  B_1 = 1 - (b - m)(1 - t) $
# 

# ##### a.2) Utilizando la función de equilibrio IS donde r está en función de Y, encuentre $Δr/ΔY$:
# 
# Recordemos la ecuación del ingreso de equilibrio a corto plazo que fue obtenida a partir del equilibrio $(Y = DA)$:
# $$ Y = \frac{1}{1 - (b - m)(1 - t)} (C_0 + I_0 + G_0 + X_0 - hr) $$
# 
# Esta ecuación, después de algunas operaciones, puede expresarse en función de la tasa de interés $(r)$:
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Entonces, la curva IS puede ser simplificada de la siguiente manera:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ y $  B_1 = 1 - (b - m)(1 - t) $
# 
# Por tanto, un cambio en la tasa de interés ante una variación en el producto es:
# 
# $$ \frac{Δr}{ΔY} = - (\frac{1-(b-m)(1-t)}{h} ) < 0 $$
# 
# Donde, $(\frac{1-(b-m)(1-t)}{h}) > 0 $, pero debido al signo negativo $(-)$ que tiene delante se convierte en: $(-)(+)= (-)$ y por eso el cambio sería negativo (< 0) y esto reflejaría una relación entre la tasa de interés y el producto negativa.
# 

# ##### a.3) Lea la sección 4.4 del material de enseñanza y explique cómo se deriva la curva IS a partir del equilibrio  y grafique lo siguiente:
# En el modelo ingreso-gasto, el equilibrio entre el Ingreso y la demanda agregada representa el equilibrio en el mercado de bienes.Además, a partir de este equilibrio se puede obtener la curva IS. 
# Como se mostró en los apartados anteriores, a partir de reemplazos en la ecuación que representa la condición de equilibrio $DA=Y$ se obtiene: $$ Y = \frac{1}{1-(b-m)(1-t)}(C_0 + G_0 + I_0 + X_0) - \frac{h}{1 - (b - m)(1 - t)}r $$    ó, lo que es lo mismo,    $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# - Explicación de derivación gráfica
# 
# Esta ecuación de derivación expresa que hay un conjunto de pares ordenados $(Y,r)$, representando $Y$ al ingreso y $r$ a la tasa de interés, las cuales van a equilibrar el mercado de bienes. Este par de valores tiene una relación inversa, por ende al graficarlo la pendiente de la curva IS es negativa. Esta relación inversa se puede ilustrar gráficamente a partir del equilibro 𝐷𝐴 = 𝑌 en la recta de 45°. De modo que, a partir de un ingreso de equilibrio $Y_0$ que se obtiene en base a una tasa de interés $r_0$ y si suponemos que se hizo una reducción en la tasa de interés, la cual cambia la tasa de interés de $r_0$ a $r_1$, siendo esta menor que la primera, entonces ante esta reducción, el intercepto de la función $DA$ aumentará, lo que genera un despalzamiento hacia la derecha (hacia arriba) y a su vez produciendo un aumento del ingreso de equilibrio, el cual se llamará $Y_1$.
# 
# Por tanto, son estas desigualdades: $r_0 > r_1$ y $Y_0 < Y_1$, las cuales representan los valores de la tasa de interés y
# del ingreso que  que van a equilibrar el mercado de bienes, los cuales como ya se mencionó se pueden representar en el plano (Y, r) por medio de una recta con pendiente negativa (\) que pasará por lo puntos ($Y_0, r_0$) y ($Y_1, r_1$). Finalmente, es esta recta con pendiente negativa la que es conocida como la Curva IS y que representa el equilibrio en el mercado de bienes.

# - Para hacer la derivación gráfica, se tiene que recordar la ecuación de la Demanda Agregada ($DA$) :

# In[2]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3
r = 0.9

Y = np.arange(Y_size)

# Ecuación de la curva del ingreso de equilibrio

def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_IS_K = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)

#--------------------------------------------------
# Recta de 45°

a = 2.5 

def L_45(a, Y):
    L_45 = a*Y
    return L_45

L_45 = L_45(a, Y)

#--------------------------------------------------
# Segunda curva de ingreso de equilibrio

# Definir cualquier parámetro autónomo
Go = 95

# Generar la ecuación con el nuevo parámetro
def DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y):
    DA_K = (Co + Io + Go + Xo - h*r) + ((b - m)*(1 - t)*Y)
    return DA_K

DA_G = DA_K(Co, Io, Go, Xo, h, r, b, m, t, Y)


# In[3]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[4]:


# Gráfico de la derivación de la curva IS a partir de la igualdad (DA = Y)

    # Dos gráficos en un solo cuadro (ax1 para el primero y ax2 para el segundo)
fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 16)) 

#---------------------------------
    # Gráfico 1: ingreso de Equilibrio
ax1.yaxis.set_major_locator(plt.NullLocator())   
ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.plot(Y, DA_IS_K, label = "DA_0", color = "#400080") 
ax1.plot(Y, DA_G, label = "DA_1", color = "#9370db") 
ax1.plot(Y, L_45, color = "#404040") 

ax1.axvline(x = 70.5, ymin= 0, ymax = 0.69, linestyle = ":", color = "C4")
ax1.axvline(x = 82.5,  ymin= 0, ymax = 0.80, linestyle = ":", color = "C4")

ax1.text(6, 4, '$45°$', fontsize = 11.5, color = 'black')
ax1.text(2.5, -3, '$◝$', fontsize = 30, color = 'black')
ax1.text(72, 0, '$Y_0$', fontsize = 12, color = 'purple')
ax1.text(84, 0, '$Y_1$', fontsize = 12, color = 'purple')
ax1.text(67, 185, 'E_0', fontsize = 12, color = 'purple')
ax1.text(84, 169, 'DA(r0)', fontsize = 12, color = 'purple')
ax1.text(74, 210, 'E_1', fontsize = 12, color = 'purple')
ax1.text(84, 200, 'DA(r1);siendo r1<r0', fontsize = 12, color = 'purple')
ax1.set(title = "Derivación de la curva IS a partir del equilibrio $Y=DA$", xlabel = r'Y', ylabel = r'DA')
ax1.legend()

#---------------------------------
    # Gráfico 2: Curva IS

ax2.yaxis.set_major_locator(plt.NullLocator())   
ax2.xaxis.set_major_locator(plt.NullLocator())

ax2.plot(Y, r, label = "IS", color = "C6") 
ax2.axvline(x = 70.5, ymin= 0, ymax = 1, linestyle = ":", color = "C4")
ax2.axvline(x = 82.5,  ymin= 0, ymax = 1, linestyle = ":", color = "C4")
plt.axhline(y = 151.5, xmin= 0, xmax = 0.7, linestyle = ":", color = "C4")
plt.axhline(y = 141.5, xmin= 0, xmax = 0.8, linestyle = ":", color = "C4")

ax2.text(72, 128, '$Y_0$', fontsize = 12, color = 'purple')
ax2.text(84, 128, '$Y_1$', fontsize = 12, color = 'purple')
ax2.text(1, 153, '$r_0$', fontsize = 12, color = 'purple')
ax2.text(1, 143, '$r_1$', fontsize = 12, color = 'purple')
ax2.text(72, 152, 'E_0', fontsize = 12, color = 'purple')
ax2.text(84, 143, 'E_1', fontsize = 12, color = 'purple')


ax2.legend()

plt.show()


# ## b. La Curva IS o el equilibrio Ahorro- Inversión

# #### Derive la ecuación de IS a partir de la igualdad Ahorro-Inversión y grafique la curva IS de equilibrio en el Mercado de Bienes

# Como se mencionó en el apartado a.1): $Y= C+I+G+X-M$, y:
# 
# Para llegar al equilibrio Ahorro-Inversión, se debe restar la tributación ($T$) de ambos miembros de la igualdad:
# 
# $$Y-T = C+I+G+X-M-T$$ 
# $$Y^d=C+I+G+X-M-T$$
# $$Y^d-C-G-X+M+T = I$$
# 
# Esta igualdad se puede reescribir de la siguiente forma:
# 
# $$(Y^d-C)+(T-G)+(M-X)=I$$
# 
# Las tres partes de la derecha constituyen los tres componentes del ahorro total($S$): ahorro privado ($Sp$) , ahorro del gobierno ($Sg$) y ahorro externo ($Se$):
# 
# $$S=Sp+Sg+Se$$
# 
# De modo que, el ahorro total ($S$) es igual a la inversión($I$):
# 
# $$Sp+Sg+Se=I$$
# 
# $$S(Y)=I(r)$$
# 
# Haciendo reemplazos se obtiene que:
# 
# $$Sp+Sg+Se=I_0-hr$$
# 
# $$(Y^d-C_0-bY^d)+(T-G_0)+(mY^d-X_0)=I_0-hr$$
# 
# Considerando las observaciones anteriores sobre los componentes de la condición de equilibrio ($Y$):
# 
# $$ [1 - (b - m)(1 - t)]Y - (C_0 + G_0 + X_0) = I_0 - hr $$
# 
# La curva IS se puede expresar con una ecuación donde la tasa de interés es una función del ingreso:
# 
# $$ hr = (C_0 + G_0 + I_0 + X_0) - (1 - (b - m)(1 - t))Y $$,
# 
# $$ r = \frac{1}{h}(C_0 + G_0 + I_0 + X_0) - \frac{1 - (b - m)(1 - t)}{h}Y $$
# 
# Y puede simplificarse en:
# 
# $$ r = \frac{B_0}{h} - \frac{B_1}{h}Y $$
# 
# Donde $ B_0 = C_0 + G_0 + I_0 + X_0  $ es el intercepto y la pendiente es $  B_1 = 1 - (b - m)(1 - t) $
# 

# - Y la curva IS se grafica de la siguiente manera:

# In[5]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[6]:


# Gráfico de la curva IS

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "#7fffd4") 

# Eliminar las cantidades de los ejes
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

# Título, ejes y leyenda
ax.set(title = "La Curva IS de Equilibrio en el Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ## c. Desequilibrios en el mercado de bienes

# #### Grafique las áreas de exceso de oferta o exceso de demanda en el mercado de bienes. Explique por qué estos puntos están fuera de los puntos de equilibrio en el mercado de bienes. 

# In[7]:


# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[8]:


# Gráfico: Curva IS
# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))
ax.yaxis.set_major_locator(plt.NullLocator())   
ax.xaxis.set_major_locator(plt.NullLocator())

ax.plot(Y, r, label = "IS", color = "#40e0d0") 

#Líneas punteadas
ax.axvline(x = 70.5, ymin= 0, ymax = 0.45, linestyle = ":", color = "#add8e6")
ax.axvline(x = 54,  ymin= 0, ymax = 0.45, linestyle = ":", color = "#add8e6")
ax.axvline(x = 37,  ymin= 0, ymax = 0.45, linestyle = ":", color = "#add8e6")
plt.axhline(y = 165, xmin= 0, xmax = 0.693, linestyle = ":", color = "#add8e6")

#Títulos, ejes y leyenda
ax.text(71, 128, '$Y_B$', fontsize = 12, color = 'blue')
ax.text(55, 128, '$Y_A$', fontsize = 12, color = 'blue')
ax.text(38, 128, '$Y_C$', fontsize = 12, color = 'blue')
ax.text(1, 167, '$r_A$', fontsize = 12, color = 'blue')
ax.text(71, 166, 'B', fontsize = 12, color = 'blue')
ax.text(55, 166, 'A', fontsize = 12, color = 'blue')
ax.text(38, 166, 'C', fontsize = 12, color = 'blue')
#Coordenadas de excesos
ax.text(70, 180, '$Exceso de Oferta$', fontsize = 12, color = 'black')
ax.text(34, 150, '$Exceso de Demanda$', fontsize = 12, color = 'black')

ax.set(title = "Equilibrio y desequilibrio en el Mercado de Bienes", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# ###### Explicación:
# Cómo observamos en el gráfico, los puntos que conforman la curva $IS$ representan el conjunto de pares ordenados $(Y,r)$ y este conjunto de valores ordenados equilibran el mercado de bienes. Sin embargo, observamos que en el gráfico también hay puntos que están fuera de los puntos de equilibrio del mercado de bienes, estos son puntos de desequilibrio de mercado. Este desequilibrio se puede producir por 2 factores: un exceso de oferta (puntos ubicados a la derecha de la curva IS) o un exceso de demanda (puntos ubicados a la izquierda de la curva IS). Por ejemplo, el punto A es de equilibrio y se encuentra dentro de la curva IS, este punto denota que el par ordenado ($Y_A, r_A$) equilibra el ahorro con la inversión ($I_A = S_A$). 
# 
# Pero, el punto B, que se encuentra a la derecha de la curva IS y corresponde al par ordenado ($Y_B, r_A$) muestra que como la tasa de interés se mantiene, entonces la inversión también se mantiene igual, pero el ahorro ha aumentado en comparación con el que corresponde al punto A, esto debido a que el ingreso aumentó y ahora es mayor que el ingreso correspondiente al punto A ($Y_A$), en suma, en el punto B, el ahorro del punto B es mayor que la inversión en el punto A($I_A< S_B$), por ende, a la derecha de la curva IS se observa un desequilibrio producido por un exceso de oferta. Asimismo, el punto C corresponde a los valores ordenados ($Y_C, r_A$) y representa que si bien la tasa de interés se mantiene y consecuentemente la inversión también lo hace, el ahorro ha disminuido en comparación con el ahorro correspondiente al punto A ($Y_A$) debido a que el ingreso ha disminuido, en suma, en el punto C, el ahorro en el punto C es menor a la inversión en el punto A ($I_A > S_C$), por consiguiente, a la izquierda de la curva IS se observa un desequilibrio producido por un exceso de demanda.

# ## d. Movimientos de la curva IS

# Utilizando el modelo macroeconómico en a) responda los siguiente:
# 
# ##### d.1) Analice Política Fiscal Contractiva con caída del Gasto del Gobierno ($ΔG<0$). Análisis intuitivo y gráfico.
# 

# -Intuición:
# $$ Go↓  →  G↓  →  DA↓  →  DA < Y  →  Y↓ $$
# 
# La disminución del gasto autónomo, disminuye el gasto público, por ende, la demanda agregada de bienes también disminuye,
# generando así una reducción de demanda, lo que da lugar a un desplazamiento de la curva IS
# hacia la izquierda, donde, como se ha mostrado anteriormente, hay exceso de demanda. 

# In[9]:


# Curva IS original

# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
Go = 55

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_G = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)
    # Gráfico


# In[10]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "purple") #IS orginal
ax.plot(Y, r_G, label = "IS_G", color = "C6", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(43, 162, '∆Go', fontsize=12, color='black')
plt.text(44, 159, '←', fontsize=15, color='blue')

# Título, ejes y leyenda
ax.set(title = "Disminución en el Gasto de Gobierno $(G_0)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# #### d.2) Analice una Política Fiscal Expansiva con una caída de la Tasa de Impuestos($Δt<0$). Análisis intuitivo y gráfico.

# - Intuición:
# $$ t↓ → Co↑ → C↑ → DA↑ → DA > Y → Y↑ $$
# $$ t↓ → M↑ → DA↓ → DA < Y → Y↓ $$

# De manera intuitiva puede parecer que se llega a una contradicción, pero por teoría de la curva IS se sabe que una reducción de la tasa de tributación reduce el consumo ($C$) y produce un aumento de la demanda agregada ($DA$) y de la producción ($Y$) aunque la tasa de interés ($r$) se mantiene, generando así un desplazamiento hacia la derecha, donde hay exceso de oferta.
# 
# El cambio en el parámetro $t$ modifica la pendiente de la curva IS. Si se adopta una política fiscal expansiva y por ende se
# reduce la tasa de impuestos, la pendiente de la IS disminuirá, provocando un movimiento en el sentido de las agujas del reloj hacia la derecha.

# - Gráfico:

# In[11]:


# Curva IS original
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
t = 0.05

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_t = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[12]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "green") #IS orginal
ax.plot(Y, r_t, label = "IS_t", color = "C8", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(67, 162, '∆t', fontsize=12, color='green')
plt.text(67, 158, ' →', fontsize=15, color='orange')

# Título, ejes y leyenda
ax.set(title = "Disminución en la Tasa de Interés $(t)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()


# #### d.3) Analice una caída de la Propensión Marginal a Consumir($Δb<0$) . Análisis intuitivo y gráfico.

# - Intuición:
# $$ b↓ → C↓  → DA↓  → DA < Y → Y↓  $$    
# 
# 
# La caída de la propensión marginal a consumir ($b$) reduce el consumo ($C$) y produce una reducción de la demanda agregada de bienes y de la producción aunque la tasa de interés se mantiene, generando así un desplazamiento hacia la IZQUIERDA, donde hay exceso de demanda.
# 
# 
# El cambio en el parámetro $b$ modifica la pendiente de la curva IS. La reducción de la propensión marginal a consumir disminuye el tamaño del multiplicador, reduce la pendiente de la $DA$ y aumenta la pendiente de la curva $IS$, lo que provoca un movimiento de la curva en el sentido contrario a las agujas del reloj.

# - Gráfico:

# In[13]:


# Curva IS original
# Parámetros

Y_size = 100 

Co = 35
Io = 40
Go = 70
Xo = 2
h = 0.7
b = 0.8
m = 0.2
t = 0.3

Y = np.arange(Y_size)


# Ecuación 
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


#--------------------------------------------------
    # NUEVA curva IS
    
# Definir SOLO el parámetro cambiado
b = 0.5

# Generar la ecuación con el nuevo parámetro
def r_IS(b, m, t, Co, Io, Go, Xo, h, Y):
    r_IS = (Co + Io + Go + Xo - Y * (1-(b-m)*(1-t)))/h
    return r_IS

r_b = r_IS(b, m, t, Co, Io, Go, Xo, h, Y)


# In[14]:


# Gráfico

# Dimensiones del gráfico
y_max = np.max(r)
fig, ax = plt.subplots(figsize=(10, 8))

# Curvas a graficar
ax.plot(Y, r, label = "IS", color = "blue") #IS orginal
ax.plot(Y, r_b, label = "IS_t", color = "C9", linestyle = 'dashed') #Nueva IS

# Texto agregado
plt.text(49, 162, '∆t', fontsize=12, color='black')
plt.text(49, 158, '←', fontsize=15, color='grey')

# Título, ejes y leyenda
ax.set(title = "Disminución en la PMgC $(b)$", xlabel= 'Y', ylabel= 'r')
ax.legend()

plt.show()

