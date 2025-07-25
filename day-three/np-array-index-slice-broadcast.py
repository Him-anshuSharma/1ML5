import numpy as np  

arr = np.array([[1, 2, 3,4], [5, 6,7,8], [9,10,11,12], [13,14,15,16]])
arr_oned = np.array([1, 2, 3, 4, 5])

#slicing
print("\nOriginal Array:\n",arr)
print("\n2nd row:",arr[1])
print("3rd row:",arr[2])
print("\nLast column:",arr[:, -1])
print("Second last column:",arr[:,-2])

#indexing
rows = [0,2]
cols = [2,0]
print("\nIndexed Elements of 2d array:\n", arr[[0,2],[2,0]])
print("\n1d array:\n",arr_oned)
indices = [0, 2, 4]
print("\nIndexed Elements of 1d array:\n", arr_oned[indices])

#broadcasting
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[10,20]])
print("\nArray 1:\n", arr1)
print("Array 2:\n", arr2)
print("\nBroadcasted Addition:\n", arr1 + arr2)