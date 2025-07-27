#storing squares of a range using list comprehension instead of for loop
squares = [i*i for i in range(5)]
print("squares comprehension",squares,"\n")

even = [i for i in range(10) if i%2==0]
print("even comprehension",even,"\n")

sentence = "Data Science is cool"
words = [word.upper() for word in sentence.split() if len(word) > 2]
print("words comprehension", words,"\n")