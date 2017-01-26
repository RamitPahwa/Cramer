from __future__ import print_function
from copy import deepcopy
import sys
import numpy
import scipy
import math
import itertools
from operator import itemgetter

# Decleration of the variable
# Author_ramit
asset_return=[]
asset_vol=[]
asset_weights=[]
const_vector=[]
det_array=[]
asset_return_sample=[]
asset_vol_sample=[]
sample_variance=[]
asset_weights_total=[]
zero=0
one=1


#add input from excell sheet 
#pyxcel can be used

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
def generateCoefMatrix(n_asset,asset_return_sub,asset_vol_sub,asset_corr_sub):
    coeficient_matrix=[[0 for i in range(n_asset)] for j in range(n_asset)]
    add_vector=[1 for i in range(n_asset)]
    for i in range(n_asset):
        for j in range(n_asset):
            if i==j:
                coeficient_matrix[i][j]=2*asset_vol_sub[i]*asset_vol_sub[i]
            else :
                coeficient_matrix[i][j]=2*asset_vol_sub[i]*asset_vol_sub[j]*asset_corr_sub[i][j]

    coeficient_matrix.append(asset_return_sub)
    coeficient_matrix.append(add_vector)
    asset_return_temp=asset_return_sub+[0,0]
    add_vector_temp=add_vector+[0,0]
    for i,row in enumerate(coeficient_matrix):
        row.append(asset_return_temp[i])
        row.append(add_vector_temp[i])
    return coeficient_matrix

def calMatrix(mat,size,index):
    for i in range(size):
        mat[i][index]=const_vector[i]
    # printMatrix(mat,size)   
    return det(mat)

def calVariance(asset_corr_sub,asset_vol_sub,asset_weights):
    total_variance=0
    for i in range(len(asset_weights)):
        total_variance+=math.pow(asset_weights[i],2)*math.pow(asset_vol_sub[i],2)
    iter_array_temp=[i for i in range(len(asset_weights))]
    iter_array=list(itertools.permutations(iter_array_temp,2))

    for i in range(len(iter_array)):
        total_variance+=asset_weights[iter_array[i][0]]*asset_weights[iter_array[i][1]]*asset_vol_sub[iter_array[i][0]]*asset_vol_sub[iter_array[i][1]]*asset_corr_sub[iter_array[i][0]][iter_array[i][1]]
    return total_variance   


def main():
    # input console based
    print("Enter the Total Number of Asset:\n")
    m_asset=int(raw_input())
    print("Enter the Number of Asset to be selected :\n")
    n_asset=raw_input()
    print("Enter the Desired Return :\n")
    total_return=raw_input()
    n_asset=int(n_asset)
    total_return=(float(total_return)/100.0)

    for i in range(m_asset):
        asset_return.append(float(raw_input("Enter asset return "+str(i+1)+":"))/100.0)
    for i in range(m_asset):
        asset_vol.append(float(raw_input("Enter asset volatility "+str(i+1)+":"))/100.0)  

    # print add_vector[1]
    # print add_vector[2]
    asset_corr=[[0 for i in range(m_asset)] for j in range(m_asset)]
    for i in range(m_asset):
        for j in range(i,m_asset):
            asset_corr[i][j]=float(raw_input("Enter the correlation between asset "+ str(i+1)+" Asset"+str(j+1)+":"))
            asset_corr[j][i]=asset_corr[i][j]

    # create constant vector 
    for i in range(n_asset):
        const_vector.append(zero)
    const_vector.append(total_return)
    const_vector.append(one)
    asset_comb_temp=[i for i in range(m_asset)]
    asset_comb=list(itertools.combinations(asset_comb_temp,n_asset))
    

    for sample in asset_comb:
        corr_sample=list(itertools.product(sample, repeat=2))
        k=0
        asset_corr_sample=[[0 for i in range(m_asset)] for j in range(m_asset)]
        for i in range(len(sample)):
            for j in range(i,len(sample)):
                asset_corr_sample[i][j]=asset_corr[corr_sample[k][0]][corr_sample[k][1]]
                k=k+1
        for index in sample:
            asset_return_sample.append(asset_return[index])
            asset_vol_sample.append(asset_vol[index])


        coeficient_matrix=generateCoefMatrix(n_asset,asset_return_sample,asset_vol_sample,asset_corr_sample)
        calculation_matrix=deepcopy(coeficient_matrix)  
        # printMatrix(coeficient_matrix,n_asset+2)
        for i in range(n_asset):
            det_array.append(calMatrix(calculation_matrix,n_asset+2,i))
            calculation_matrix=deepcopy(coeficient_matrix)  
        for i in range(n_asset):
            if det(coeficient_matrix)==0:
                print("coeficient matrix is zero")
            else:
                asset_weights.append(det_array[i]/det(coeficient_matrix))
                asset_weights_total.append(det_array[i]/det(coeficient_matrix)) 
        sample_variance.append(calVariance(asset_corr_sample,asset_vol_sample,asset_weights)) 
        asset_vol_sample[:]=[]
        asset_return_sample[:]=[]
        det_array[:]=[]
        asset_weights[:]=[]
    min_variance_index=min(enumerate(sample_variance), key=itemgetter(1))[0]
    for i in range(n_asset):
        print("ASSET "+str(asset_comb[min_variance_index][i]+1)+": Weight is :"+str(asset_weights_total[(min_variance_index*n_asset)+i]))
        
main()


