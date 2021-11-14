#!/usr/bin/env python
# coding: utf-8

# In[109]:


#Create a one Dimensional array
import numpy as np
a=np.array([1,2,3])
a


# In[110]:


#Type and shape of the array
print(type(a))
print(a.shape)


# In[111]:


#To get the elements of an array using the index
print(a[0],a[2])


# In[112]:


#Modifying the existing array
a[0]=5
a


# In[113]:


#Create two dimensional array
b=np.array([[1,2,3],[4,5,6]])
b


# In[114]:


#To get the elements of 2D array using the index
b[1,2]


# In[115]:


#Type and shape of the array
print(type(b))
print(b.shape)


# In[116]:


#Create an array with zeros
a=np.zeros((3,3))
print(a)
print(a.dtype)


# In[117]:


#create an array with ones
b=np.ones((2,2))
b


# In[118]:


#Create an array with a constant
c=np.full((2,2),5)
print(c)
print(c.dtype)         #Type is int for full function


# In[119]:


#Create an identity matrix
d=np.eye(3)        #One one value is given because identity matrix is 1*1, 2*2, 3*3....
d


# In[120]:


#Create an array with random numbers
e=np.random.random((3,3))
print(e)
f= np.random.randint(1,100, size=(3,2))
print(f)


# In[121]:


#Create an array and perform matrix multiplication
x=np.array([[1,2],[3,4],[4,5]])
y=np.array([[2,1],[3,5]])
z=np.matmul(x,y)
print("matmul:",z)
r=np.dot(x,y)
print("dot:",r)


# In[122]:


""""
Write a NumPy program to create an element-wise comparison (greater,
greater_equal, less and less_equal) of two given arrays.
x = np.array([72, 79, 85, 90, 150, -135, 120, -10, 60, 100])
y = np.array([72, 79, 85, 90, 150, -135, 120, -10, 60, 100.000001])
"""
x = np.array([72, 79, 85, 90, 150, -135, 120, -10, 60, 100])
y = np.array([72, 79, 85, 90, 150, -135, 120, -10, 60, 100.000001])
x_g_y=np.greater(x,y)
print(x_g_y)
print(np.greater_equal(x,y))
print(np.less(x,y))
print(np.less_equal(x,y))


# In[123]:


#Write a NumPy program to create an array of 10 zeros,10 ones, 10 fives.
a=np.zeros(10)
print(a)
b=np.ones(10)
print(b)
c=np.full(10,5)
print(c)


# In[124]:


#Write a NumPy program to generate a random number between 0 and 1
e=np.random.random((3,3))
print(e)


# In[125]:


#Write a NumPy program to create a 3x4 matrix filled with values from 10 to 21.
x =  np.arange(10, 22).reshape(3,4)
x


# In[126]:


"""
Write a NumPy program to convert an array to a float type.
Sample output:
Original array
[1, 2, 3, 4]
"""
x=np.array([1, 2, 3, 4])
print(x)
print(x.dtype)
#y=np.asfarray(x)
y=x.astype(float)
print(y)
print(y.dtype)


# In[127]:


#Write a NumPy program to convert a list and tuple into arrays
l1=(1,2,3)
t1=[4,5,6]
x1=np.array([l1])
print(x1,"\n",type(x1))
x2=np.array([t1])
print(x2,"\n",type(x2))


# In[128]:


"""
Write a NumPy program to find common values between two arrays. 
Array1: [ 0 10 20 40 60]
Array2: [10, 30, 40]
"""
a1=np.array([0,10,20,40,60])
a2=np.array([10,30,40])
d=np.intersect1d(a1,a2)
d


# In[139]:


"""
Write a NumPy program to find the indices of the maximum and minimum
values along the given axis of an array. 
Original array: [1 2 3 4 5 6]
"""
import numpy as np
a1=np.array([1,2,8,4,7,6])
max_a1=np.argmax(a1)
print("Index of Max:",max_a1)
min_a1=np.argmin(a1)
print("Index of Min:",min_a1)

r=np.where(a1 == np.amax(a1))
print(r[0])
r=np.where(a1 == np.amin(a1))
print(r[0])


# In[142]:


#Write a NumPy program to compute the determinant of a given square array.
a1=np.array([[3,8],[4,6]])
print(np.linalg.det(a1))


# In[146]:


#Write a NumPy program to compute the multiplication of two given matrixes.
a1=np.array([[3,8],[4,6]])
a2=np.array([[1,2],[2,3]])
r=np.dot(a1,a2)
print("Matrix Multiplication:",r)

r1=a1*a2
print("Element wise Multiplication:",r1)


# In[149]:


"""
Write a NumPy program to create a structured array from given
student name, height, class and their data types. Now sort the array on
height. 
Original array:
[(b'James', 5, 48.5 ) (b'Nail', 6, 52.5 ) (b'Paul', 5, 42.1 )
(b'Pit', 5, 40.11)]
"""
import numpy as np
data_type = [('name', 'S15'), ('class', int), ('height', float)]
students_details = [('James', 5, 48.5), ('Nail', 6, 52.5),('Paul', 5, 42.10), ('Pit', 5, 40.11)]
# create a structured array
students = np.array(students_details, dtype=data_type)
print("Original array:")
print(students)
print("Sort by class, then height if class are equal:")
print(np.sort(students, order=[ 'height']))


# In[ ]:


"""
Write a NumPy program to get the indices of the sorted elements of
a given array. 
Original array:
[1023 5202 6230 1671 1682 5241 4532]
"""
a1=np.array([1023,5202,6230,1671,1682,5241,4532])
for i in a1
    


# In[151]:


"""
Write a NumPy program to concatenate element-wise two arrays of
string. 
"""
a1=np.array(['Python ','PHP'])
a2=np.array(['Java ','C++'])
new_a=np.char.add(a1,a2)
print(new_a)


# In[159]:


"""
Write a NumPy program to repeat all the elements three times of a
given array of string 
Expected Output:
Original Array:
[&#39;Python&#39; &#39;PHP&#39; &#39;Java&#39; &#39;C++&#39;]"""
a1=np.array([['Python ',' PHP',' Java ',' C++']])
#for i in range(3):
#    print(a1)

new_a=np.char.multiply(a1, 3)
print(new_a)


# In[157]:


"""
Write a NumPy program to split the element of a given array with
spaces. 
Original Array:
[&#39;Python PHP Java C++&#39;]
"""
a1=np.array([['Python PHP Java C++']])
r=np.char.split(a1)
print(r)


# In[ ]:




