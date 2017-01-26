from __future__ import print_function
from copy import deepcopy
import sys
import numpy
import scipy

# Author_Ramit
asset_return=[]
asset_vol=[]
asset_weights=[]
const_vector=[]
det_array=[]
zero=0
one=1

def minor(matrix,i):
    """Returns the Minor M_0i of matrix"""
    minor = deepcopy(matrix)
    del minor[0] #Delete first row
    for b in list(range(len(matrix)-1)): #Delete column i
        del minor[b][i]
    return minor

def det(A):
    """Recursive function to find determinant"""
    if len(A) == 1: #Base case on which recursion ends
        return A[0][0]
    else:
        determinant = 0
        for x in list(range(len(A))): #Iterates along first row finding cofactors
            # print("A:", A)
            determinant += A[0][x] * (-1)**(2+x) * det(minor(A,x)) #Adds successive elements times their cofactors
            # print("determinant:", determinant)
        return determinant

def printMatrix(mat,size):
	for i in range(size):
		for j in range(size):
			print (str(mat[i][j])+'\t',end='')
		print()	

def calMatrix(mat,size,index):
	for i in range(size):
		mat[i][index]=const_vector[i]
	printMatrix(mat,size)	
	return det(mat)

# input console based
print("Enter the Number of Asset :\n")
n_asset=raw_input()
print("Enter the Desired Return :\n")
total_return=raw_input()
n_asset=int(n_asset)
total_return=float(total_return)

for i in range(n_asset):
	asset_return.append(float(raw_input("Enter asset return "+str(i+1)+":")))
for i in range(n_asset):
	asset_vol.append(float(raw_input("Enter asset volatility "+str(i+1)+":")))	
add_vector=[1 for i in range(n_asset)]
print (add_vector)
# print add_vector[1]
# print add_vector[2]
asset_corr=[[0 for i in range(n_asset)] for j in range(n_asset)]
for i in range(n_asset):
	for j in range(i,n_asset):
		asset_corr[i][j]=float(raw_input("Enter the correlation between asset "+ str(i+1)+" Asset"+str(j+1)+":"))
		asset_corr[j][i]=asset_corr[i][j]

# create constant vector 
for i in range(n_asset):
	const_vector.append(zero)
const_vector.append(total_return)
const_vector.append(one)
print (const_vector)
# compute the coeficient matrix 

coeficient_matrix=[[0 for i in range(n_asset)] for j in range(n_asset)]

for i in range(n_asset):
	for j in range(n_asset):
		if i==j:
			coeficient_matrix[i][j]=2*asset_vol[i]*asset_vol[i]
		else :
			coeficient_matrix[i][j]=2*asset_vol[i]*asset_vol[j]*asset_corr[i][j]

coeficient_matrix.append(asset_return)
coeficient_matrix.append(add_vector)
asset_return_temp=asset_return+[0,0]
add_vector_temp=add_vector+[0,0]
for i,row in enumerate(coeficient_matrix):
	row.append(asset_return_temp[i])
	row.append(add_vector_temp[i])

calculation_matrix=deepcopy(coeficient_matrix)	

printMatrix(coeficient_matrix,n_asset+2)
for i in range(n_asset):
	det_array.append(calMatrix(calculation_matrix,n_asset+2,i))
	calculation_matrix=deepcopy(coeficient_matrix)	
for i in range(n_asset):
	asset_weights.append(det_array[i]/det(coeficient_matrix)) 


