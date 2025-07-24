#sytnax = {key, value}
#Keys must be unique and immutable (like strings, numbers, or tuples).
#Values can be any data type

students = {"Diksha":30,"Sheena":107,"Himanshu":"21BCI0253"}

nums = [1,2,3,4,5,6,7,8,9,10,10,4,2,1,3,1,2,4,5,2,1,2,5,2,5,1,2,124,214421,3,24,2]

count = {}

for num in sorted(nums):
    count[num] = count.get(num,0) + 1
print(count[10])

del count[10]

print(count)