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
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print("arr1",arr1)
print("arr2",arr2)
print("array addition:",arr1+arr2)
print("array subraction:",arr1-arr2)
print("array multiplication:",arr1*arr2)
print("array division:",arr1/arr2)

#statistics
arr = np.array([1,2,3,4,5])
print("mean",np.mean(arr))
print("median",np.median(arr))
print("standard deviation",np.std(arr))

