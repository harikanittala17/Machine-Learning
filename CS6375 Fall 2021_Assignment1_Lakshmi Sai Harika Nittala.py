#Code Submitted by- Lakshmi Sai Harika Nittala
#Course Code - CS6375
#Semester- Fall 2021

import numpy as np 
# The true function
def f_true(x):
    y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
    return y

                      # For all our math needs
n = 750                                     # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                       # True labels with noise


import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

# Plot the data

plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r');

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue');

# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
  # Creation of n X d matrix of phi
    
    phi=[]
    for i in X:
        temp=[]
        for dimension in range(0,d+1):
            temp.append(np.power(i,dimension))
        phi.append(temp)
    phi=np.asarray(phi)
    return phi   

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(phi, y):
    phi=np.asarray(phi)
    y=np.asarray(y)
    w=(np.linalg.inv(phi.T@phi))@(phi.T)@y
    return w
  
    
  # Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    sum=0
    w_transpose=w.T
    for i in range(0,np.size(y)):
        sum=sum+np.power((y[i]-(w_transpose@Phi[i])),2)
        
    E_MSE=sum/np.size(y)
    return E_MSE


w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60]);


plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
    X_d = polynomial_transform(x_true, d)
    y_d = X_d @ w[d]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15]);


# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    
    X=np.array(X,dtype=float)
    B=np.array(B,dtype=float)
    
    radial_basis_kernel=[]
    
    for i in X:
        temp=[]
        for j in B:
            subs=np.power((i-j),2)
            temp.append(np.exp((gamma*-1)*subs))
        radial_basis_kernel.append(temp)
    radial_basis_kernel=np.asarray(radial_basis_kernel)
    
    return radial_basis_kernel

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
import math
def train_ridge_model(Phi, y, lam):
    #print(Phi.shape)
    A=(Phi.T@Phi)
    B=lam*np.eye(int(math.sqrt(Phi.size)),dtype=int)
    C=np.add(A,B)
    w=(np.linalg.inv(C))@(Phi.T)@y.T
    w=np.asarray(w)
    return w


radial_basis_w={}
radial_basis_val_error={}

radial_basis_test_error={}

log_lamda={}

initial=10**(-3)
lamda_list=[]

while initial<=10**3:
    lamda_list.append(initial)
    initial=initial*10
    
for lamda_values in lamda_list:
    
    #training on the training set
    log_lamda[lamda_values]=math.log(lamda_values,10)
    radial_basis_trn_phi=radial_basis_transform(X_trn,X_trn)
    radial_basis_w[lamda_values]=train_ridge_model(radial_basis_trn_phi,y_trn,lamda_values)
    
    #validating on the validation set
    radial_basis_val_phi=radial_basis_transform(X_val,X_trn)
    radial_basis_val_error[lamda_values]=evaluate_model(radial_basis_val_phi,y_val,radial_basis_w[lamda_values])
    
    #testing on the test set
    radial_basis_test_phi=radial_basis_transform(X_tst.T,X_trn)
    radial_basis_test_error[lamda_values]=evaluate_model(radial_basis_test_phi,y_tst,radial_basis_w[lamda_values])
       
#plotting lamda vs validation error and lamda vs test error
plt.figure()
plt.plot(log_lamda.values(), radial_basis_val_error.values(),marker='o',linewidth=3,markersize=12)
plt.plot(log_lamda.values(), radial_basis_test_error.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('log10(Lamda Values)', fontsize=16,labelpad=40)
plt.ylabel('Validation/Test error', fontsize=16)
plt.rcParams["figure.autolayout"] = True
plt.xticks(list(log_lamda.values()))
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([log_lamda.get(0.001),log_lamda.get(1000),15,65]);

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for lamda in lamda_list:
    X_n = radial_basis_transform(x_true,X_trn)
    y_n = X_n @ radial_basis_w[lamda]
    plt.plot(x_true, y_n, marker='None', linewidth=2)

plt.legend(['true'] + lamda_list,loc='best')
plt.axis([-10, 10, -18, 18]);

