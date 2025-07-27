nums = [1,2,3,4,5]
people = [('Himanshu', 25), ('John', 30), ('Alice', 22), ('Bob', 28), ('Eve', 35)]

lambda_double = lambda x: x*2
lambda_even = lambda x: x % 2 == 0

#map -> apply to all
d_map = list(map(lambda_double, nums))

#filter -> apple to some
d_filter = list(filter(lambda_even,nums))

#
sorted_by_age = sorted(people,key = lambda x:x[1])

print(d_map,"\n",d_filter,"\n",sorted_by_age)