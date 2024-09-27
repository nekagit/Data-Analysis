# Sample list of numbers
numbers = [1, 2, 3, 4, 5]

# Implementations using loops
# 1. Using a for loop
doubled_with_loop_1 = []
for num in numbers:
    doubled_with_loop_1.append(num * 2)

# 2. Using a while loop
doubled_with_loop_2 = []
i = 0
while i < len(numbers):
    doubled_with_loop_2.append(numbers[i] * 2)
    i += 1

# 3. Using a nested loop (for demonstration)
doubled_with_loop_3 = []
for num in numbers:
    for _ in range(1):  # Simulating a nested loop
        doubled_with_loop_3.append(num * 2)

# 4. Using a for loop with index
doubled_with_loop_4 = []
for i in range(len(numbers)):
    doubled_with_loop_4.append(numbers[i] * 2)

# 5. Using a for loop with condition
doubled_with_loop_5 = []
for num in numbers:
    if num % 2 == 0:  # Only double even numbers
        doubled_with_loop_5.append(num * 2)
    else:
        doubled_with_loop_5.append(num)

# 6. Using a for loop to create a dictionary
doubled_with_loop_6 = {num: num * 2 for num in numbers}

# 7. Using a for loop with list concatenation
doubled_with_loop_7 = []
for num in numbers:
    doubled_with_loop_7 += [num * 2]

# 8. Using a for loop with a list comprehension for nested structure
doubled_with_loop_8 = [num * 2 for num in numbers]

# 9. Using a for loop with a function
def double(num):
    return num * 2

doubled_with_loop_9 = [double(num) for num in numbers]

# 10. Using a for loop with a combined operation
doubled_with_loop_10 = []
for num in numbers:
    doubled_with_loop_10.append(num + num)

# Implementations using list comprehensions (similar to map)
# 1. Basic list comprehension
doubled_with_map_1 = [num * 2 for num in numbers]

# 2. Using a conditional in list comprehension
doubled_with_map_2 = [num * 2 if num % 2 == 0 else num for num in numbers]

# 3. Using a function in list comprehension
def double_func(num):
    return num * 2

doubled_with_map_3 = [double_func(num) for num in numbers]

# 4. Using index in list comprehension
doubled_with_map_4 = [numbers[i] * 2 for i in range(len(numbers))]

# 5. Creating a list of tuples
doubled_with_map_5 = [(num, num * 2) for num in numbers]

# 6. Doubling and adding 1 in one comprehension
doubled_with_map_6 = [(num * 2) + 1 for num in numbers]

# 7. Nested list comprehension (demonstration)
doubled_with_map_7 = [num * 2 for num in [x for x in numbers]]

# 8. Using map with a lambda function
doubled_with_map_8 = list(map(lambda num: num * 2, numbers))

# 9. List comprehension with string formatting
doubled_with_map_9 = [f"Double of {num} is {num * 2}" for num in numbers]

# 10. Using list comprehension with filtering
doubled_with_map_10 = [num * 2 for num in numbers if num > 2]

# Implementations using lambda functions
# 1. Basic lambda usage
doubled_with_lambda_1 = list(map(lambda x: x * 2, numbers))

# 2. Lambda with condition
doubled_with_lambda_2 = list(map(lambda x: x * 2 if x % 2 == 0 else x, numbers))

# 3. Lambda with function definition
double_lambda_func = lambda x: x * 2
doubled_with_lambda_3 = list(map(double_lambda_func, numbers))

# 4. Lambda with index
doubled_with_lambda_4 = list(map(lambda x, i: x * 2, numbers, range(len(numbers))))

# 5. Lambda with tuples
doubled_with_lambda_5 = list(map(lambda x: (x, x * 2), numbers))

# 6. Doubling and adding 1 with lambda
doubled_with_lambda_6 = list(map(lambda x: x * 2 + 1, numbers))

# 7. Nested lambda
doubled_with_lambda_7 = list(map(lambda x: (lambda y: y * 2)(x), numbers))

# 8. Lambda with filter
doubled_with_lambda_8 = list(map(lambda x: x * 2, filter(lambda x: x > 2, numbers)))

# 9. Lambda with string formatting
doubled_with_lambda_9 = list(map(lambda x: f"Double of {x} is {x * 2}", numbers))

# 10. Combining lambdas
doubled_with_lambda_10 = list(map(lambda x: x * 2 if x % 2 == 0 else x, filter(lambda x: x > 0, numbers)))

# Printing results
print("Doubled with loops:")
print(doubled_with_loop_1)
print(doubled_with_loop_2)
print(doubled_with_loop_3)
print(doubled_with_loop_4)
print(doubled_with_loop_5)
print(doubled_with_loop_6)
print(doubled_with_loop_7)
print(doubled_with_loop_8)
print(doubled_with_loop_9)
print(doubled_with_loop_10)

print("\nDoubled with maps (list comprehensions):")
print(doubled_with_map_1)
print(doubled_with_map_2)
print(doubled_with_map_3)
print(doubled_with_map_4)
print(doubled_with_map_5)
print(doubled_with_map_6)
print(doubled_with_map_7)
print(doubled_with_map_8)
print(doubled_with_map_9)
print(doubled_with_map_10)

print("\nDoubled with lambda functions:")
print(doubled_with_lambda_1)
print(doubled_with_lambda_2)
print(doubled_with_lambda_3)
print(doubled_with_lambda_4)
print(doubled_with_lambda_5)
print(doubled_with_lambda_6)
print(doubled_with_lambda_7)
print(doubled_with_lambda_8)
print(doubled_with_lambda_9)
print(doubled_with_lambda_10)
