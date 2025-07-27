import numpy as np

# Create a 1D numpy array
one_d_array = np.array([1,2,3,4,5])
print(one_d_array)

#create a 2D numpy array
two_d_array = np.array([[1,2,3,4,5],[10,11,12,13,14]])
print(two_d_array)


#shape of array
print("Shape of 1D array:",one_d_array.shape)
print("Shape of 2D array:",two_d_array.shape)

#basic maths
arr1 = np.array([[1,2,3],[7,8,9]])
arr2 = np.array([[4,5,6],[10,11,12]])
print("\n\narr1\n",arr1)
print("\narr2\n",arr2)
print("\narray addition:\n",arr1+arr2)
print("\narray subraction:\n",arr1-arr2)
print("\narray multiplication:\n",arr1*arr2)
print("\narray division:\n",arr1/arr2)

#statistics
arr = np.array([1,2,3,4,5])
print("mean",np.mean(arr))
print("median",np.median(arr))
print("standard deviation",np.std(arr))

