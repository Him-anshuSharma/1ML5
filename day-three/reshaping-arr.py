import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# Reshaping the array
new_arr = arr.reshape(2, 8)  # Reshape to 2 rows and 8 columns
print("Original Array:\n", arr)
print("\nReshaped Array:\n", new_arr)
print("\nFlattened Array:\n", arr.flatten())  # Flatten the array to 1D
print("\nFlattened Array with reshape:\n", arr.reshape(-1))
print("\nReshaped Array with one dimension mentioned:\n", arr.reshape(2, -1))  

#conditional selection
filtered_arr = arr[0:3,-2:]
print("\Sliced Array:\n", filtered_arr)
indices = np.where(arr > 10)
print("\nIndices where elements are greater than 10:\n", indices)
print("conditional selection:\n", arr[arr>10])

#syntax to keep shape result = np.where(some_array > condition, some_array, fill_value)
print(np.where(arr > 10, arr, -1))  