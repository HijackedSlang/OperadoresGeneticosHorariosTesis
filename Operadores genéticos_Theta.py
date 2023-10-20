#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import copy

# Load the list from the binary file
with open("databasic.pickle", "rb") as file:
    data = pickle.load(file)


# In[2]:


# Estos tres vectores los declaro para poder usarlos externamente
vector_one = []
vector_two = []
vector_three = []


# Crear la lista de números predefinidos, esta lista representa la materia que es y que profesores pueden ofertarla
predefined_numbers = {
    "1": [1,2,3,4],
    "2": [5,6,7,8],
    "3": [9,10,11,12],
    "4": [13,14,15,16],
    
    "8": [17,18,19,20],
    "9": [21,22,23,24],
    "10": [25,26,27,28],
    "11": [29,30,31,32],
    
    "14": [33,34,35,36],
    "15": [37,38,39,40],
    "16": [41,42,1,2],
    "17": [3,4,5,6],
    "18": [7,8,9,10],
    
    "20": [11,12,13,14],
    "21": [15,16,17,18],
    "22": [19,20,21,22],
    "23": [23,24,25,26],
    "24": [27,28,29,30],
    
    "28": [31,32,33,34],
    
    "31": [35,36,37,38],
    "32": [39,40,41,42],
    "33": [1,2,3,4],
    "34": [5,6,7,8],
    "35": [9,10,11,12],
    "36": [13,14,15,16],
    
    "40": [17,18,19,20],
    "41": [21,22,23,24],
    "42": [25,26,27,28],
    "45": [29,30,31,32],
    
    "47": [33,34,35,36],
    "48": [37,38,39,40],
    
    "52": [41,42,1,2],
    "53": [3,4,5,6],
    "54": [7,8,9,10],
    "55": [11,12,13,14],
    
    "57": [15,16,17,18],
    "58": [19,20,21,22],
    "59": [23,24,25,26],
    "60": [27,28,29,30]
}

valid_pairs = {
    1: [1,2,3,4],
    2: [5,6,7,8],
    3: [9,10,11,12],
    4: [13,14,15,16],
    
    8: [17,18,19,20],
    9: [21,22,23,24],
    10: [25,26,27,28],
    11: [29,30,31,32],
    
    14: [33,34,35,36],
    15: [37,38,39,40],
    16: [41,42,1,2],
    17: [3,4,5,6],
    18: [7,8,9,10],
    
    20: [11,12,13,14],
    21: [15,16,17,18],
    22: [19,20,21,22],
    23: [23,24,25,26],
    24: [27,28,29,30],
    
    28: [31,32,33,34],
    
    31: [35,36,37,38],
    32: [39,40,41,42],
    33: [1,2,3,4],
    34: [5,6,7,8],
    35: [9,10,11,12],
    36: [13,14,15,16],
    
    40: [17,18,19,20],
    41: [21,22,23,24],
    42: [25,26,27,28],
    45: [29,30,31,32],
    
    47: [33,34,35,36],
    48: [37,38,39,40],
    
    52: [41,42,1,2],
    53: [3,4,5,6],
    54: [7,8,9,10],
    55: [11,12,13,14],
    
    57: [15,16,17,18],
    58: [19,20,21,22],
    59: [23,24,25,26],
    60: [27,28,29,30]
}


import random

professors = list(range(1, 43)) * 2
random.shuffle(professors)

# Estas son las materias que no están permitidas, las pongo aquí para que se puedan poner restricciones de materias anuales
banned_numbers = [5, 6, 7, 12, 13, 19, 25, 26, 27, 29, 30, 37, 38, 39, 43, 44, 46, 49, 50, 51, 56]


# In[3]:


# Voy a hacer el esqueleto del algoritmo genético.

# Primero declararé las restricciones


import random

def intercambiar_grupos(vector):
    for i in range(100):
        #buscar parejas que violen la restriccion
        for j in range(len(vector)):
            for k in range(len(vector[j])):
                if vector[j][k][1] != 0:
                    if vector[j][k][1] not in primer_aparicion:
                        primer_aparicion[vector[j][k][1]] = j
                    else:
                        diferencia = abs(primer_aparicion[vector[j][k][1]] - j)
                        if diferencia == 0 and j%2 == primer_aparicion[vector[j][k][1]]%2:
                            #intercambiar grupos
                            if j%2 == 0:
                                grupo_random = random.choice([i for i in range(len(vector)) if i%2 !=0 ])
                            else:
                                grupo_random = random.choice([i for i in range(len(vector)) if i%2 == 0])
                            vector[j],vector[grupo_random] = vector[grupo_random],vector[j]
                            primer_aparicion = {}
                            break
        #verificar si se cumpl
    return vector

primer_aparicion = {}



def cumple_restricciones_apariciones(vector):
    #Diccionario para guardar la primera aparición de un valor Y y su posición relativa dentro del grupo
    primer_aparicion = {}
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] != 0:
                #obteniendo la primera aparición de un valor Y
                if vector[i][j][1] not in primer_aparicion:
                    primer_aparicion[vector[i][j][1]] = j
                else:
                    #checando diferencia de apariciones
                    diferencia = abs(primer_aparicion[vector[i][j][1]] - j)
                    if diferencia > 2:
                        print("La restriccion de diferencia entre apariciones se viola con la pareja: {}".format(vector[i][j]))
                        return False
    return True


def cumple_restriccion_paridad(vector):
    #Diccionario para guardar la primera aparición de un valor Y y su posición relativa dentro del grupo
    primer_aparicion = {}
    violaciones_paridad = []
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] != 0:
                #obteniendo la primera aparición de un valor Y
                if vector[i][j][1] not in primer_aparicion:
                    primer_aparicion[vector[i][j][1]] = i
                else:
                    #checando diferencia de apariciones
                    diferencia = abs(primer_aparicion[vector[i][j][1]] - i)
                    if diferencia == 0 and i%2 != primer_aparicion[vector[i][j][1]]%2:
                        violaciones_paridad.append([i,j,vector[i][j]])
    if violaciones_paridad:
        print("Las restriccion de paridad se viola con las parejas: {}".format(violaciones_paridad))
        return False
    else:
        return True




def cumple_restricciones_repeticiones(vector):
    #Diccionario para guardar la cantidad de veces que un valor Y aparece en alguna pareja
    apariciones_y = {}
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] != 0:
                #Actualizando contador de apariciones de Y
                if vector[i][j][1] in apariciones_y:
                    apariciones_y[vector[i][j][1]] += 1
                else:
                    apariciones_y[vector[i][j][1]] = 1
                #checando cantidad de apariciones
                if apariciones_y[vector[i][j][1]] > 3:
                    return False
    return True

def cumple_restricciones_indices(vector):
    #Diccionario para guardar la cantidad de apariciones de parejas no nulas en cada índice relativo
    apariciones_indice = {}
    for i in range(len(vector)):
        for j in range(len(vector[i])):
            if vector[i][j][1] != 0:
                #Actualizando contador de apariciones en índice relativo
                if j in apariciones_indice:
                    apariciones_indice[j] += 1
                else:
                    apariciones_indice[j] = 1
                #checando cantidad de apariciones en índice relativo
                if apariciones_indice[j] > 18:
                    return False
    return True




def cumplen_restricciones(vector_total):
    # Concatenando ambos vectores 
    #vector_total = vector1 + vector2
    #lista para almacenar las restricciones que no se cumplen
    restricciones_violadas = []
    #Revisando restricciones de apariciones
    if not cumple_restricciones_apariciones(vector_total):
        restricciones_violadas.append("La restriccion de apariciones no se cumple")
    # Revisando restricciones de repeticiones
    if not cumple_restricciones_repeticiones(vector_total):
        restricciones_violadas.append("La restriccion de repeticiones no se cumple")
    # Revisando restricciones de indices
    if not cumple_restricciones_indices(vector_total):
        restricciones_violadas.append("La restriccion de indices no se cumple")
    # retornando si cumplen todas las restricciones y las restricciones que no se cumplen.
    return len(restricciones_violadas) == 0, restricciones_violadas




# Debo recortar dividir el vector cuando lo regrese
def dividir_vector(vector,vec1):
    mitad = len(vec1)
    vector1 = vector[:mitad]
    vector2 = vector[mitad:]
    return vector1, vector2

# Aquí hacer el checkeo de las tres restricciones
def fast_check(vec1,vec2):
    
    vector_total = vec1 + vec2
    intentos = 0
    
    print(cumplen_restricciones(vector_total))

    #vec1, vec2 = dividir_vector(vector_total,vec1)
    
    print("Vector 1:")
    for i, grupo in enumerate(vec1):
        print(f"V1_G{i}: {grupo}")
    print("Vector 2:")
    for i, grupo in enumerate(vec2):
        print(f"V2_G{i}: {grupo}")
    return vec1, vec2

#print(vec1, vec2)
# vector_total = vec1 + vec2
# vec1, vec2 = fast_check(vec1,vec2)
#print(vec1, vec2)
#fast_check(vec1,vec2)


# Aqui hay un check para que se sepa si pasa la restricción o no

def check_parity(vector):
    wrong_pairs = []
    unique_wrong_pairs=[]
    unique_wrong_index = []
    first_appearance = {}
    for i, group in enumerate(vector):
        for j, pair in enumerate(group):
            if pair[1] != 0:
                if pair[1] not in first_appearance:
                    first_appearance[pair[1]] = [i,j]
                else:
                    if (i%2) == (first_appearance[pair[1]][0]%2) and abs(j - first_appearance[pair[1]][1]) == 0:
                        if pair not in unique_wrong_pairs:
                            unique_wrong_pairs.append(pair)
                            unique_wrong_index.append(j)
                            wrong_pairs.append(pair)
    # print("Parejas cuya restricción falló: ",unique_wrong_pairs)
    # print("Index interno cuya restricción falló: ",unique_wrong_index)
    return unique_wrong_pairs, unique_wrong_index


# check_parity(vector_total)

# Aquí hay otro check para lo mismo, pero este check devuelve un vector que servirá para la función que arregla el fallo

def check_parity_v3(vector):
    parejas_malas = []
    first_appearance = {}
    for i, group in enumerate(vector):
        for j, pair in enumerate(group):
            if pair[1] != 0:
                if pair[1] not in first_appearance:
                    first_appearance[pair[1]] = [i,j]
                else:
                    if (i%2) == (first_appearance[pair[1]][0]%2) and abs(j - first_appearance[pair[1]][1]) == 0:
                        parejas_malas.append([i,j,pair[1]])
    print(parejas_malas)
    return parejas_malas

# parejas_malas = check_parity_v3(vector_total)


# Esta parte utiliza un algoritmo rotativo para arreglar los fallos de paridad
def correct_parity_violations(vector, parejas_fallidas):

    runs = 0
    while parejas_fallidas and runs < 200:

        null_groups = []
        for i, group in enumerate(vector):
            if all(pair == [0, 0] for pair in group):
                null_groups.append(i)
        
        runs +=1
        for i, violation in enumerate(parejas_fallidas):
            if not null_groups:
                return vector
            rand_null_group = random.choice(null_groups)
            
            # Esta es la parte donde intercambia los grupos que contienen las parejas fallidas
            vector[violation[0]], vector[rand_null_group] = vector[rand_null_group], vector[violation[0]]
            print(vector[violation[0]], vector[rand_null_group])
            if all(pair == [0,0] for pair in vector[rand_null_group]):
                null_groups.append(rand_null_group)
            else:
                null_groups.remove(rand_null_group)
        parejas_fallidas = check_parity_v3(vector)
    if not parejas_fallidas:
        print("Fallos de paridad corregidos.")
    else:
        print("Fallos aún presentes, no se han podido corregir")
    return vector


def check_restrictions_and_parity(vector_total):
    cumplen, _ = cumplen_restricciones(vector_total)
    parity = check_parity(vector_total)
    if cumplen == True and len(parity[0]) == 0 and len(parity[1]) == 0:
        return True
        print("E verda")
    else:
        print("noe verda")
    return False

#vector_total


# In[4]:


# Load the list from the binary file

# Variable de conteo
num_passed = 0

# Recorrer la lista de individuos y realizar el chequeo
for individual in data:
    if check_restrictions_and_parity(individual):
        num_passed += 1

# Verificar si todos los individuos pasaron el chequeo
if num_passed == len(data):
    print("exitoso")
else:
    print("no exitoso")


# In[5]:


len(data)


# In[6]:


def restructure_data(num_numbers):
    restructured = {}
    for x, y_list in num_numbers.items():
        for y in y_list:
            if y not in restructured:
                restructured[y] = []
            restructured[y].append(x)
    return restructured
num_numbers2 = restructure_data(copy.deepcopy(predefined_numbers))


# In[7]:



best_individual = []


# In[8]:


L_bound = [4, 1, 5, 1, 0, 0, 0, 6, 6, 6, 2, 0, 0, 4, 4, 4, 4, 0, 0, 4, 2, 0, 4, 1, 0, 2, 1, 4, 2, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
U_bound = [5, 2, 6, 2, 1, 1, 1, 7, 7, 7, 3, 1, 1, 5, 5, 5, 5, 1, 1, 5, 3, 1, 5, 1, 1, 3, 2, 5, 3, 2, 1, 1, 3, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1]
bias_1= [0.2, 0.85, 0.05, 0.85, 2.0, 2.0, 2.0, 0.05, 0.05, 0.05, 0.65, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 2.0, 2.0, 0.2, 0.65, 2.0, 0.2, 0.85, 2.0, 1.0, 1.0, 0.2, 0.65, 0.3, 2.0, 2.0, 0.65, 2.0, 2.0, 0.85, 2.0, 2.0, 0.85, 0.85, 2.0, 2.0, 2.0, 2.0, 0.85, 0.85, 0.85, 0.65, 5.0, 5.0, 5.0, 0.85, 0.85, 0.85, 0.85, 2.0, 0.85, 0.85, 2.0, 2.0]
bias_2=[0.9, 0.3, 1.0, 0.3, -0.1, -0.1, -0.1, 1.0, 1.0, 1.0, 0.32, -0.1, -0.1, 0.9, 0.9, 0.9, 0.9, -0.1, -0.1, 0.9, 0.32, -0.1, 0.9, 0.3, -0.1, 0.3, 0.3, 0.9, 0.32, 0.85, -0.1, -0.1, 0.32, -0.1, -0.1, 0.3, -0.1, -0.1, 0.3, 0.3, -0.1, -0.1, -0.1, -0.1, 0.3, 0.3, 0.3, 0.32, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.1, 0.3, 0.3, -0.1, -0.1]


referencia = [6, 3, 31, 2, 0, 0, 0, 28, 32, 33, 2, 0, 0, 4, 4, 5, 14, 0, 0, 26, 30, 0, 8, 37, 0, 45, 5, 11, 45, 21, 0, 0, 28, 0, 0, 19, 0, 0, 4, 14, 0, 0, 0, 0, 32, 21, 34, 1, 2, 0, 0, 26, 26, 26, 26, 0, 26, 26, 0, 0]
multiplicadores = [1.5, 3, 6.2, 2, 0, 0, 0, 4.666666667, 5.333333333, 5.5, 1, 0, 0, 1, 1, 1.25, 3.5, 0, 0, 6.5, 15, 0, 2, 1, 0, 22.5, 5, 2.75, 22.5, 21, 0, 0, 14, 0, 0, 19, 0, 0, 4, 14, 0, 0, 0, 0, 32, 21, 34, 0.5, 0, 0, 0, 26, 26, 26, 26, 0, 26, 26, 0, 0]




def fitness(data):
    # Inicialización del diccionario de apariciones
    # Inicialización del diccionario de apariciones
    def within_boundaries(count_result, L_bound, U_bound):
        return [lower <= count <= upper for count, lower, upper in zip(count_result, L_bound, U_bound)]
    referencia = [6, 3, 31, 2, 0, 0, 0, 28, 32, 33, 2, 0, 0, 4, 4, 5, 14, 0, 0, 26, 30, 0, 8, 37, 0, 45, 5, 11, 45, 21, 0, 0, 28, 0, 0, 19, 0, 0, 4, 14, 0, 0, 0, 0, 32, 21, 34, 1, 2, 0, 0, 26, 26, 26, 26, 0, 26, 26, 0, 0]
    multiplicadores = [1.5, 3, 6.2, 2, 0, 0, 0, 4.666666667, 5.333333333, 5.5, 1, 0, 0, 1, 1, 1.25, 3.5, 0, 0, 6.5, 15, 0, 2, 1, 0, 22.5, 5, 2.75, 22.5, 21, 0, 0, 14, 0, 0, 19, 0, 0, 4, 14, 0, 0, 0, 0, 32, 21, 34, 0.5, 0, 0, 0, 26, 26, 26, 26, 0, 26, 26, 0, 0]
    banned_numbers = [5, 6, 7, 12, 13, 19, 25, 26, 27, 29, 30, 37, 38, 39, 43, 44, 46, 49, 50, 51, 56]
    appearances = [0]*60
    # Recorrido de todos los grupos de tamaños 7
    for group in data:

        # Recorrido de cada pareja dentro de cada grupo
        for pair in group:

            X = pair[0]
            # Si X es diferente de cero, se aumenta el contador de apariciones
            if X != 0:
                appearances[X-1] += 1
    # Inicialización de la lista multiplicadores
    
    # Cálculo de la nueva lista "cupo"
    #cupo = [a*m for a,m in zip(appearances, multiplicadores)]  
    bobb = [(ref / mult) * (b1 + b2) if mult != 0 else 0 for ref, mult, b1, b2 in zip(referencia, multiplicadores, bias_1, bias_2)]

    
    count_result = count(best_individual)
        # Calculate the bonus score based on within_boundaries results and bobb values
    within_bounds = within_boundaries(count_result, L_bound, U_bound)
    bonus_score = sum([2 * b if count == upper else b if is_within else -1
                       for b, is_within, count, upper in zip(bobb, within_bounds, count_result, U_bound)])

    
    
    
    
    
    
    for i in banned_numbers:
        referencia[i-1] = 0
    
    #print(cupo)
    
    # Cálculo de las diferencias
    
    #Lo que sobra
    cupo1 = [a*m for a,m in zip(appearances, multiplicadores)]
    cupo1 = [a*m for a,m in zip(cupo1, bias_1)]
    diferencias = [cupo1_value - referencia_value for cupo1_value, referencia_value in zip(cupo1, referencia) if cupo1_value > referencia_value]
    
    #Lo que falta
    cupo2 = [a*m for a,m in zip(appearances, multiplicadores)]
    cupo2 = [a*m for a,m in zip(cupo2, bias_2)]
    diferencia_pb = [cupo2_value - referencia_value for cupo2_value, referencia_value in zip(cupo2, referencia) if cupo2_value < referencia_value]
    
    
    cupo = [a*m for a,m in zip(appearances, multiplicadores)]
    score = sum(cupo) - (sum(diferencias)*20) + (sum(diferencia_pb)*20) + bonus_score*10
    #core = pow(sum(cupo), sum_cupo_exponent) * (1 if sum(cupo) >= 0 else -1) - 8*sum([pow(abs(d), diferencias_exponent) * (1 if d >= 0 else -1) for d in diferencias]) + 8*sum([pow(abs(d), diferencia_pb_exponent) * (1 if d >= 0 else -1) for d in diferencia_pb]) + 40*pow(abs(bonus_score), bonus_score_exponent) * (1 if bonus_score >= 0 else -1)
    
    return int(score)


# In[9]:


def count(data):
    # Inicialización del diccionario de apariciones
    appearances = [0]*60
    # Recorrido de todos los grupos de tamaños 7
    for group in data:
        # Recorrido de cada pareja dentro de cada grupo
        for pair in group:
            X = pair[0]
            # Si X es diferente de cero, se aumenta el contador de apariciones
            if X != 0:
                appearances[X-1] += 1
    
    return appearances

def genetic_crossover(vec1, vec2):
    m = len(vec1)
    n = len(vec2)
    original1 = copy.deepcopy(vec1)
    original2 = copy.deepcopy(vec2)
    swaps = []
    successful_swaps = 0
    necessary = random.randint(1, 3)
    for i in range(m):
        for j in range(7):
            p1 = vec1[i][j]
            if p1 == [0, 0]:
                continue
            for k in range(n):
                for l in range(7):
                    p2 = vec2[k][l]
                    if p2 == [0, 0] or ((i, j), (k, l)) in swaps:
                        continue
                    if p1[1] == p2[1] and p1[0] != p2[0] and random.random() > 0.75:
                        vec1[i][j], vec2[k][l] = vec2[k][l], vec1[i][j]
                        swaps.append(((i, j), (k, l)))
                        successful_swaps += 1
                        if successful_swaps >= necessary:
                            break
                    elif len(swaps) - successful_swaps >= 5:
                        break
                else:
                    continue
                break
            if successful_swaps >= necessary:
                break
        if successful_swaps >= necessary:
            break

    vec1_final = copy.deepcopy(vec1)
    vec2_final = copy.deepcopy(vec2)
    
    return vec1_final, vec2_final




vec1, vec2 = copy.deepcopy(genetic_crossover(copy.deepcopy(data[1]), copy.deepcopy(data[2])))
print( f" \n {vec1}")
print( f" \n {vec2}")
len (vec1)
len (vec2)


# In[10]:


# Función Mutación

def mutation(vector, num_numbers2):
    # Copiar el diccionario num_numbers2
    num_numbers_copy = copy.deepcopy(num_numbers2)
    original = copy.deepcopy(vector)
    
    # Obtener la cantidad de mutaciones exitosas
    successful_mutations = 0

    for i in range(random.randint(1,3)):
        # Seleccionar una pareja [X, Y] al azar
        random_pair = random.choice([pair for sublist in vector for pair in sublist if pair[0] != 0])
        original_X = random_pair[0]
        original_Y = random_pair[1]
        # Mutar el valor de X
        new_X = random.choice(num_numbers_copy[original_Y])
        while new_X == original_X:
            new_X = random.choice(num_numbers_copy[original_Y])
        random_pair[0] = int(new_X)

        # Revisar si la mutación cumple con las restricciones y la paridad
        if check_restrictions_and_parity(vector):
            successful_mutations += 1
            #print(f"Mutation successful: [{original_X}, {original_Y}] -> [{new_X}, {original_Y}]")
        else:
            # Revertir la mutación
            random_pair[0] = original_X
    
    fit_prev = copy.deepcopy(fitness(original))
    fit_new = copy.deepcopy(fitness(vector))
    
    #if fit_new > fit_prev and check_restrictions_and_parity(vector):
    #    return copy.deepcopy(vector)
    #else:
    #    return copy.deepcopy(original)
    return copy.deepcopy(vector)
    

#vec_t = mutation(copy.deepcopy(data[1]),num_numbers2)

#print(f" \n {data[1]}")
#print(f" \n {vec_t}")


# In[ ]:


import copy
import random
import keyboard
import time
import multiprocessing as mp

# New function definitions
def within_boundaries(count_result, L_bound, U_bound):
    scores = []
    for i, (count, lower, upper) in enumerate(zip(count_result, L_bound, U_bound)):
        if lower <= count <= upper:
            scores.append(1)
        elif count == lower - 1:
            scores.append(count/lower)
        else:
            scores.append(0)
    return sum(scores)/len(scores)

def contar_tfd(lista):
    return lista

def calculate_similarity(population1, population2):
    # We convert each individual to a string so they can be hashed
    population1_set = set(str(individual) for individual in population1)
    population2_set = set(str(individual) for individual in population2)
    common_individuals = population1_set.intersection(population2_set)
    similarity = len(common_individuals) / len(population1_set)
    return similarity


def tournament_selection(population, tournament_size):
    return max(random.sample(population, tournament_size), key=fitness)

def maxmax(population):
    scores = []
    for individual in population:
        count_result = count(individual)
        scores_list = within_boundaries(count_result, L_bound, U_bound)
        individual_score = contar_tfd(scores_list)
        scores.append(individual_score)

    max_score = max(scores)
    max_index = scores.index(max_score)
    best_individual = population[max_index]

    return best_individual, max_score

def genetic_algorithm(data, num_numbers2, bias_1, bias_2, min_population_size=600, elitism_rate=0.35, mutation_rate_change_factor=0.1, mutation_rate_min=0.01, mutation_rate_max=1.0, itera=10, tournament_size=300,cyc_n=10):
    start_time_t = time.time()
    
    for cycle in range(cyc_n):  # Re-run the whole algorithm 3 times
        print(f"Start of cycle {cycle + 1}...")
        start_time = time.time()

        # Reset for each cycle
        population_size = 1100
        original_population = copy.deepcopy(data) if cycle == 0 else updated_original_population
        population = copy.deepcopy(original_population)
        best_individuals = []  # List to store the best individual from each generation

        # Initialize
        _, max_score = maxmax(data)
        print(max_score)
        mutation_probability = 0.99
        cross_probability = 0.99
        generations = 0
        best_individual = None
        best_individuals_history = []
        previous_best_score = 0
        last_score_values = []
        convergence_threshold = 0.01  # adjust this as needed
        best_score_generation = None  # Track when the best score first appeared
        # Initialize overall score
        best_individual_overall = None
        best_score_overall = float('-inf')

        # Evaluate initial population scores and sort
        initial_population_scores = [contar_tfd(within_boundaries(count(individuo), L_bound, U_bound)) for individuo in original_population]
        original_population = [x for _,x in sorted(zip(initial_population_scores, original_population), key=lambda pair: pair[0], reverse=True)]

        # Copy the sorted population
        population = copy.deepcopy(original_population)
        best_score = float('-inf')
        best_score_history = []
        
        print("Iterating...")
        for generations in range(itera + 1):
                    # Reset population every n generations
            if generations % 5 == 0 and generations != 0:
                print("Resetting original population at generation", generations)

                # Calculate and print the similarity before resetting the population
                similarity_before_reset = calculate_similarity(population, original_population)
                print(f"Similarity before reset: {similarity_before_reset * 100:.2f}%")

                population = copy.deepcopy(original_population)
                
                # Replace the worst individual with the current best individual
                worst_individual_index = population.index(min(population, key=lambda x: contar_tfd(within_boundaries(count(x), L_bound, U_bound))))
                population[worst_individual_index] = copy.deepcopy(best_individual)  # Make sure to use a copy of the best individual

                similarity_after_reset = calculate_similarity(population, original_population)
                print(f"Similarity after reset: {similarity_after_reset * 100:.2f}%")
            
            # Calculate scores
            population_scores = [contar_tfd(within_boundaries(count(individuo), L_bound, U_bound)) for individuo in population]

            # Score-based Selection
            population = genetic_selection(population, population_scores, min_population_size)

            # New population
            new_population = []
            while len(new_population) < population_size:
                ind1 = tournament_selection(population, tournament_size)
                ind2 = tournament_selection(population, tournament_size)
                r_num = random.uniform(0, 1)
                if r_num <= cross_probability:
                    offspring1, offspring2 = genetic_crossover(ind1, ind2)
                else:
                    offspring1, offspring2 = ind1, ind2
                new_population.append(offspring1)
                new_population.append(offspring2)

            # Elitism
            sorted_population = sorted(population, key=lambda x: contar_tfd(within_boundaries(count(x), L_bound, U_bound)), reverse=True)
            elite_count = int(elitism_rate * population_size)
            new_population[:elite_count] = sorted_population[:elite_count]

            # Mutation
            for i in range(len(new_population)):
                random_num = random.uniform(0, 1)
                if random_num <= mutation_probability:
                    new_population[i] = mutation(new_population[i], copy.deepcopy(num_numbers2))

            # Update mutation and crossover probabilities
            if generations % 9 == 0:
                if max(population_scores) - previous_best_score < 0.01:
                    mutation_probability = min(mutation_probability * mutation_rate_change_factor, mutation_rate_max)
                    cross_probability = min(cross_probability * mutation_rate_change_factor, mutation_rate_max)
                else:
                    mutation_probability = max(mutation_probability / mutation_rate_change_factor, mutation_rate_min)
                    cross_probability = max(cross_probability / mutation_rate_change_factor, mutation_rate_min)

            # Update population
            population = new_population

            # Score evaluation
            population_scores = [contar_tfd(within_boundaries(count(individuo), L_bound, U_bound)) for individuo in population]

            best_new_score = max(population_scores)
            best_new_index = population_scores.index(best_new_score)
            best_new_individual = population[best_new_index]

            # Update best score and individual
            if best_new_score > best_score:
                best_score = best_new_score
                best_individual = copy.deepcopy(best_new_individual)
                best_score_history.append(best_score)
                best_score_generation = generations  # Update when the best score first appeared

                # Update overall best score and individual
                if best_new_score > best_score_overall:
                    best_score_overall = best_new_score
                    best_individual_overall = copy.deepcopy(best_new_individual)
            
            # Print the score every generation for the best individual
            print(f"Gen.{generations}. Best score: {best_score}")
            
            # Save the best individual from the current generation
            best_individuals.append(copy.deepcopy(best_individual))

            # After every 10th generation, update the original population with the best individuals and reset the population
            if generations % 10 == 0 and generations != 0 and best_individuals:
                print(f"Injecting best individuals into original population at generation {generations}...")
                original_population[:len(best_individuals)] = copy.deepcopy(best_individuals)  # Replace the first n individuals with the best individuals
                updated_original_population = original_population  # Update the original population for the next cycle
                best_individuals = []  # Reset the list of best individuals
                population = copy.deepcopy(original_population)  # Reset the population
                # Replace the worst individual with the current best individual
                worst_individual_index = population.index(min(population, key=lambda x: contar_tfd(within_boundaries(count(x), L_bound, U_bound))))
                population[worst_individual_index] = copy.deepcopy(best_individual)

            best_individuals_history.append(copy.deepcopy(best_individual))

            # Check for convergence
            if len(last_score_values) >= 5:
                # Calculate the changes in scores
                score_changes = [abs(last_score_values[i] - last_score_values[i - 1]) for i in range(1, len(last_score_values))]
                if all(change < convergence_threshold for change in score_changes):
                    print(f"Convergence reached at generation {generations}. Stopping the algorithm...")
                    break

            # Update the list of last score values
            last_score_values.append(best_score)
            if len(last_score_values) > 4:
                last_score_values.pop(0)

            generations += 1

            if keyboard.is_pressed('q') and keyboard.is_pressed('w'):
                break
        
        print(f"End of cycle {cycle + 1}. The best individual first appeared at generation {best_score_generation}.")
        print(f"The best individual is: {best_individual} with a score of: {best_score}")
        print("---Time, %s seconds ---" % (time.time() - start_time))
        
    print(f"The best individual overall is: {best_individual_overall} with a score of: {best_score_overall}")
    print("---Full time, %s seconds ---" % (time.time() - start_time_t))
    return best_individual_overall, best_individuals_history, best_score_history




def genetic_selection(population, population_scores, min_population_size):
    population_and_scores = list(zip(population, population_scores))
    population_and_scores.sort(key=lambda x: x[1], reverse=True)
    selected_population = [individuo for individuo, score in population_and_scores[:int(len(population)/2)]]

    # If the population size is less than the minimum, add some of the least apt individuals
    if len(selected_population) < min_population_size:
        population_and_scores.sort(key=lambda x: x[1])
        for individuo, score in population_and_scores[:min_population_size - len(selected_population)]:
            selected_population.append(individuo)

    return selected_population

best_individual, best_individuals_history, best_score_history=genetic_algorithm(data, num_numbers2, bias_1, bias_2, min_population_size=600, elitism_rate=0.35, mutation_rate_change_factor=0.1, mutation_rate_min=0.01, mutation_rate_max=1.0, itera=10) #, elitism_rate=0.1, island_hopping_rate=0.2
#, best_indiv_fitness, best_curr_individual 


# In[ ]:


a = [[[14, 35], [31, 38], [17, 6], [0, 0], [0, 0], [0, 0], [0, 0]], [[11, 30], [28, 33], [1, 4], [2, 6], [0, 0], [0, 0], [0, 0]], [[8, 19], [52, 42], [54, 8], [0, 0], [0, 0], [0, 0], [0, 0]], [[8, 19], [10, 26], [17, 3], [0, 0], [0, 0], [0, 0], [0, 0]], [[14, 33], [14, 36], [11, 31], [0, 0], [0, 0], [0, 0], [0, 0]], [[18, 7], [8, 20], [28, 32], [28, 32], [0, 0], [0, 0], [0, 0]], [[55, 14], [4, 13], [9, 24], [23, 25], [0, 0], [0, 0], [0, 0]], [[53, 3], [20, 13], [8, 20], [9, 21], [0, 0], [0, 0], [0, 0]], [[48, 40], [17, 5], [9, 21], [0, 0], [0, 0], [0, 0], [0, 0]], [[57, 16], [10, 27], [16, 41], [0, 0], [0, 0], [0, 0], [0, 0]], [[3, 9], [40, 18], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], [[16, 41], [16, 1], [15, 40], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [9, 22], [23, 23], [33, 2]], [[0, 0], [0, 0], [0, 0], [0, 0], [3, 10], [0, 0], [8, 17]], [[0, 0], [0, 0], [0, 0], [0, 0], [10, 28], [45, 29], [20, 12]], [[0, 0], [0, 0], [0, 0], [0, 0], [24, 29], [0, 0], [3, 11]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [10, 28]], [[0, 0], [0, 0], [0, 0], [0, 0], [10, 25], [20, 12], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [3, 11], [9, 22]], [[0, 0], [0, 0], [0, 0], [0, 0], [48, 37], [36, 15], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [23, 23], [21, 15]], [[0, 0], [0, 0], [0, 0], [0, 0], [33, 2], [28, 34], [3, 10]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [1, 4], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [8, 17], [0, 0], [15, 39]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [47, 34]], [[0, 0], [0, 0], [0, 0], [0, 0], [15, 39], [48, 37], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
b = [[[14, 35], [31, 38], [17, 6], [0, 0], [0, 0], [0, 0], [0, 0]], [[11, 30], [28, 33], [1, 4], [2, 6], [0, 0], [0, 0], [0, 0]], [[58, 19], [52, 42], [54, 8], [0, 0], [0, 0], [0, 0], [0, 0]], [[8, 19], [10, 26], [53, 3], [0, 0], [0, 0], [0, 0], [0, 0]], [[14, 33], [14, 36], [11, 31], [0, 0], [0, 0], [0, 0], [0, 0]], [[18, 7], [8, 20], [28, 32], [28, 32], [0, 0], [0, 0], [0, 0]], [[20, 14], [20, 13], [9, 24], [10, 25], [0, 0], [0, 0], [0, 0]], [[17, 3], [4, 13], [8, 20], [9, 21], [0, 0], [0, 0], [0, 0]], [[48, 40], [17, 5], [9, 21], [0, 0], [0, 0], [0, 0], [0, 0]], [[57, 16], [10, 27], [16, 41], [0, 0], [0, 0], [0, 0], [0, 0]], [[3, 9], [40, 18], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], [[16, 41], [33, 1], [15, 40], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [9, 22], [23, 23], [33, 2]], [[0, 0], [0, 0], [0, 0], [0, 0], [3, 10], [0, 0], [8, 17]], [[0, 0], [0, 0], [0, 0], [0, 0], [10, 28], [24, 29], [20, 12]], [[0, 0], [0, 0], [0, 0], [0, 0], [45, 29], [0, 0], [3, 11]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [10, 28]], [[0, 0], [0, 0], [0, 0], [0, 0], [23, 25], [55, 12], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [3, 11], [9, 22]], [[0, 0], [0, 0], [0, 0], [0, 0], [48, 37], [36, 15], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [23, 23], [21, 15]], [[0, 0], [0, 0], [0, 0], [0, 0], [33, 2], [47, 34], [3, 10]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [1, 4], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [8, 17], [0, 0], [15, 39]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [28, 34]], [[0, 0], [0, 0], [0, 0], [0, 0], [15, 39], [15, 37], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]] 

def variables_iguales(a, b):
    return a == b

variables_iguales(a, b)


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


def plot_score(best_individuals, best_score_history):
    scores = [score for score in best_score_history]
    abs_min_score = abs(min(scores))
    scores = [s + abs_min_score + 1 for s in scores]
    max_score = max(scores)
    min_score = min(scores)

    plt.style.use('dark_background')  # set the background to black
    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)  # add grid

    sns.lineplot(data=scores, ax=ax, color='#668cff', label='Individuals')

    score_diff = np.diff(scores)
    rate_of_change = np.divide(score_diff, scores[:-1])
    rate_of_change = np.insert(rate_of_change, 0, 0)

    ax2 = ax.twinx()
    sns.lineplot(x=range(len(rate_of_change)), y=rate_of_change, ax=ax2, color='orange')

    ax2_ylim = ax2.get_ylim()
    ax2.set_ylim([ax2_ylim[0]/max_score, 0.75])

    plt.title('Best Score vs. Generation', color='white')
    ax.set_xlabel('Generation', color='white')
    ax.set_ylabel('Score', color='white')
    ax2.set_ylabel('Rate of Change', color='white')
    ax.set_ylim([min_score - 10, max_score + 10])
    ax.set_xlim([0, len(best_individuals) - 1])
    ax.patch.set_facecolor('black')  # set the background of the plot
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.spines['left'].set_color('white')
    
    plt.grid(True, alpha=0.3)

    line = ax2.get_lines()[-1]
    line.set_label('Rate of Change')
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles+handles2, labels+labels2, loc='upper right', bbox_to_anchor=(1, 0.85))
    
    legend_text = f'Maximum (normalized): {max_score:.2f}\n'
    legend_text += f'Minimum (normalized): {min_score:.2f}\n'
    plt.annotate(legend_text, xy=(1, 0), xytext=(-12, 0), fontsize=10,
                 xycoords='axes fraction', textcoords='offset points',
                 ha='right', va='bottom')
    
    plt.savefig('score.png', dpi=300, bbox_inches='tight')

    plt.show()

plot_score(best_individuals_history, best_score_history)


# In[ ]:


#test =  [3, 1, 3, 0, 0, 0, 0, 5, 6, 7, 2, 0, 0, 4, 3, 2, 4, 1, 0, 4, 3, 0, 2, 1, 0, 0, 0, 3, 0, 0, 0, 1, 3, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1]
#individual = [[[4, 14], [40, 22], [10, 31], [54, 7], [0, 0], [0, 0], [0, 0]], [[2, 6], [16, 5], [4, 14], [0, 0], [0, 0], [0, 0], [0, 0]], [[17, 3], [23, 25], [33, 7], [0, 0], [0, 0], [0, 0], [0, 0]], [[24, 32], [8, 17], [58, 19], [0, 0], [0, 0], [0, 0], [0, 0]], [[3, 10], [9, 24], [21, 15], [0, 0], [0, 0], [0, 0], [0, 0]], [[8, 20], [15, 39], [3, 9], [55, 13], [0, 0], [0, 0], [0, 0]], [[8, 17], [3, 12], [23, 28], [0, 0], [0, 0], [0, 0], [0, 0]], [[3, 9], [23, 26], [17, 4], [0, 0], [0, 0], [0, 0], [0, 0]], [[52, 1], [20, 11], [57, 18], [10, 26], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [16, 41], [0, 0], [48, 38]], [[28, 35], [11, 34], [15, 42], [15, 41], [0, 0], [0, 0], [0, 0]], [[14, 34], [31, 37], [15, 40], [0, 0], [0, 0], [0, 0], [0, 0]], [[10, 30], [14, 39], [20, 16], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [14, 33], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [10, 29], [47, 33], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [21, 15], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [17, 4], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [20, 13], [0, 0], [9, 21]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [9, 23], [53, 8], [17, 8]], [[0, 0], [0, 0], [0, 0], [0, 0], [59, 23], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [45, 29], [0, 0], [33, 2]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [10, 27], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [8, 21], [28, 36], [9, 27]], [[0, 0], [0, 0], [0, 0], [0, 0], [16, 42], [28, 36], [48, 38]], [[0, 0], [0, 0], [0, 0], [0, 0], [16, 2], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [8, 18], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
test = count(best_individual)
#test = count(individual)
def within_boundaries(count_result, L_bound, U_bound):
    scores = []
    one_shy_indices = []
    one_shy_values_boundaries = []
    for i, (count, lower, upper) in enumerate(zip(count_result, L_bound, U_bound)):
        if lower <= count <= upper:
            scores.append(1)
        elif count == lower - 1:
            scores.append(count/lower)
            one_shy_indices.append(i + 1)  # Add 1 since the index is zero-based
            one_shy_values_boundaries.append((count, lower, upper))
        else:
            scores.append(0)
    print("Índices de conteos 1 menos que el límite inferior:", one_shy_indices)
    
    print("Valores y sus límites inferior y superior correspondientes:")
    for valor, límite_inferior, límite_superior in one_shy_values_boundaries:
        print(f"Valor: {valor}, Límite inferior: {límite_inferior}, Límite superior: {límite_superior}")
    print(f"Valor de score {sum(scores)}")
    return sum(scores)/len(scores)



def contar_tfd(scores):
    print(f"La lista tiene {scores}.")
    print("L_bound\n")
    print(L_bound)
    print("\nResultado de count(individual): ")
    print(test)
    
#mi_lista = [True, True, True, True, True, True, False, False, False, False, True, False, False, False, False, False, True, True, False, True, False, False, False, False, False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, False, True, True, True, True, True, False, True, True, True, False, False, False, False, True, True, False, False, True]
#test = count(best_individual)
testd = within_boundaries(test,L_bound, U_bound)
contar_tfd(testd)

def obtener_valores_por_indice(n, L_bound, individual, bias_1, bias_2):
    try:
        lbound_valor = L_bound[n]
        individual_valor = test[n]
        bias_1_valor = bias_1[n]
        bias_2_valor = bias_2[n]
        return lbound_valor, individual_valor, bias_1_valor, bias_2_valor
    except IndexError:
        print(f"El índice {n} está fuera del rango de la lista.")

nem = 19
nem = nem-1
lbound_valor, individual_valor, bias_1_valor, bias_2_valor = obtener_valores_por_indice(nem, L_bound, individual, bias_1, bias_2)

print(f"Valor de L_bound en el índice {nem+1}: {lbound_valor}")
print(f"Valor de count(individual) en el índice {nem+1}: {individual_valor}")
print(f"Valor de bias_1 en el índice {nem+1}: {bias_1_valor}")
print(f"Valor de bias_2 en el índice {nem+1}: {bias_2_valor}")


# In[ ]:




