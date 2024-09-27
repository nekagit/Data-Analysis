import os

# 1. Basic function without arguments
def greet():
    print("Hello, World!")

greet()  # Calling the function, Output: "Hello, World!"


# 2. Function with positional arguments
def add(a, b):
    return a + b

result = add(5, 3)  # Passing arguments 5 and 3
print(result)  # Output: 8


# 3. Function with positional and keyword arguments
def introduce(name, age):
    print(f"Name: {name}, Age: {age}")

introduce("Alice", 30)            # Positional arguments
introduce(age=25, name="Bob")      # Keyword arguments


# 4. Function with default argument value
def greet(name="Guest"):
    print(f"Hello, {name}!")

greet()        # Output: "Hello, Guest!" (default argument used)
greet("John")  # Output: "Hello, John!" (argument overrides the default)


# 5. Function with return value
def multiply(a, b):
    return a * b

result = multiply(4, 5)
print(result)  # Output: 20


# 6. Function with arbitrary arguments (*args)
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # Output: 6


# 7. Function with arbitrary keyword arguments (**kwargs)
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])

cheeseshop("Cheddar", "It's very runny, sir.", "It's really very, very runny, sir.", client="John", shopkeeper="Michael")


# 8. Recursive function
def tri_recursion(k):
    if k > 0:
        result = k + tri_recursion(k - 1)  # Recursive call
        print(result)
    else:
        result = 0
    return result

print("\n\nRecursion Example Results")
tri_recursion(6)  # Output: Recursive sum of numbers 6 to 0


# 9. Function using `sorted()` with lambda function (Sorting)
def sort_list(lst):
    return sorted(lst, key=lambda x: x)

numbers = [3, 1, 4, 1, 5, 9]
sorted_numbers = sort_list(numbers)
print(sorted_numbers)  # Output: [1, 1, 3, 4, 5, 9]


# 10. Using `map()` function (Transformation of a list)
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square, numbers))
print(squared_numbers)  # Output: [1, 4, 9, 16, 25]


# 11. Using `filter()` function (Filtering a list)
def is_even(n):
    return n % 2 == 0

even_numbers = list(filter(is_even, numbers))
print(even_numbers)  # Output: [2, 4]


# 12. Using `os` library to interact with the file system
def list_files_in_directory(path='.'):
    return os.listdir(path)

print(list_files_in_directory())  # Lists all files in the current directory


# 13. Function to get the current working directory
def get_current_directory():
    return os.getcwd()

print(get_current_directory())  # Output: Current working directory


# 14. Function to check if a path exists
def check_path_exists(path):
    return os.path.exists(path)

print(check_path_exists("/tmp"))  # Output: True or False, depending on the existence of the path


# 15. Lambda function for quick mathematical operation
multiply_by_two = lambda x: x * 2
print(multiply_by_two(5))  # Output: 10


# 16. List comprehension for generating a list of squares
squares = [x**2 for x in range(6)]
print(squares)  # Output: [0, 1, 4, 9, 16, 25]

# 1. Basic function without arguments
def greet():
    print("Hello, World!")

greet()  # Calling the function, Output: "Hello, World!"


# 2. Function with positional arguments
def add(a, b):
    return a + b

result = add(5, 3)  # Passing arguments 5 and 3
print(result)  # Output: 8


# 3. Function with positional and keyword arguments
def introduce(name, age):
    print(f"Name: {name}, Age: {age}")

introduce("Alice", 30)            # Positional arguments
introduce(age=25, name="Bob")      # Keyword arguments


# 4. Function with default argument value
def greet(name="Guest"):
    print(f"Hello, {name}!")

greet()        # Output: "Hello, Guest!" (default argument used)
greet("John")  # Output: "Hello, John!" (argument overrides the default)


# 5. Function with return value
def multiply(a, b):
    return a * b

result = multiply(4, 5)
print(result)  # Output: 20


# 6. Function with arbitrary arguments (*args)
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # Output: 6


# 7. Function with arbitrary keyword arguments (**kwargs)
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])

cheeseshop("Cheddar", "It's very runny, sir.", "It's really very, very runny, sir.", client="John", shopkeeper="Michael")


# 8. Recursive function
def tri_recursion(k):
    if k > 0:
        result = k + tri_recursion(k - 1)  # Recursive call
        print(result)
    else:
        result = 0
    return result

print("\n\nRecursion Example Results")
tri_recursion(6)  # Output: Recursive sum of numbers 6 to 0


# 9. Function using `sorted()` with lambda function (Sorting)
def sort_list(lst):
    return sorted(lst, key=lambda x: x)

numbers = [3, 1, 4, 1, 5, 9]
sorted_numbers = sort_list(numbers)
print(sorted_numbers)  # Output: [1, 1, 3, 4, 5, 9]


# 10. Using `map()` function (Transformation of a list)
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square, numbers))
print(squared_numbers)  # Output: [1, 4, 9, 16, 25]


# 11. Using `filter()` function (Filtering a list)
def is_even(n):
    return n % 2 == 0

even_numbers = list(filter(is_even, numbers))
print(even_numbers)  # Output: [2, 4]


# 12. Using `os` library to interact with the file system
import os

def list_files_in_directory(path='.'):
    return os.listdir(path)

print(list_files_in_directory())  # Lists all files in the current directory


# 13. Function to get the current working directory
def get_current_directory():
    return os.getcwd()

print(get_current_directory())  # Output: Current working directory


# 14. Function to check if a path exists
def check_path_exists(path):
    return os.path.exists(path)

print(check_path_exists("/tmp"))  # Output: True or False, depending on the existence of the path


# 15. Lambda function for quick mathematical operation
multiply_by_two = lambda x: x * 2
print(multiply_by_two(5))  # Output: 10


# 16. List comprehension for generating a list of squares
squares = [x**2 for x in range(6)]
print(squares)  # Output: [0, 1, 4, 9, 16, 25]


# 17. Function Calling Inside Another Function (Function Composition)
def multiply(a, b):
    return a * b

def square_of_sum(a, b):
    return multiply(a + b, a + b)

result = square_of_sum(3, 4)  # Calls multiply(7, 7)
print(result)  # Output: 49


# 18. Function Returning a Function (Higher-Order Functions)
def make_multiplier(x):
    def multiplier(y):
        return x * y
    return multiplier

times_two = make_multiplier(2)
times_three = make_multiplier(3)

print(times_two(5))   # Output: 10
print(times_three(5)) # Output: 15


# 19. Anonymous Functions (Lambda Functions)
double = lambda x: x * 2
print(double(10))  # Output: 20

# Using lambda in a function call
result = list(map(lambda x: x**2, [1, 2, 3, 4]))
print(result)  # Output: [1, 4, 9, 16]


# 20. Calling Functions with Unpacking Operator `*`
def greet_people(greeting, *names):
    for name in names:
        print(f"{greeting}, {name}!")

people = ['Alice', 'Bob', 'Charlie']
greet_people("Hello", *people)  # Output: "Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"


# 21. Partial Function Application Using `functools.partial`
from functools import partial

def power(base, exponent):
    return base ** exponent

# Create a new function that always squares numbers
square = partial(power, exponent=2)

print(square(4))  # Output: 16
print(square(5))  # Output: 25


# 22. Recursive Function with Memoization
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

print(fibonacci(10))  # Output: 55


# 23. Using `map()` with Multiple Iterables
a = [1, 2, 3]
b = [4, 5, 6]
result = list(map(lambda x, y: x + y, a, b))
print(result)  # Output: [5, 7, 9]


# 24. Using `zip()` to Combine Multiple Iterables
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
combined = list(zip(names, ages))
print(combined)  # Output: [('Alice', 25), ('Bob', 30), ('Charlie', 35)]


# 25. Using `enumerate()` for Indexed Iteration
names = ["Alice", "Bob", "Charlie"]
for index, name in enumerate(names, start=1):
    print(f"{index}: {name}")
# Output:
# 1: Alice
# 2: Bob
# 3: Charlie


# 26. Decorators (Functions Modifying Functions)
def my_decorator(func):
    def wrapper():
        print("Something before the function.")
        func()
        print("Something after the function.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Something before the function.
# Hello!
# Something after the function.


# 27. Generator Functions Using `yield`
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for number in countdown(5):
    print(number)
# Output: 5, 4, 3, 2, 1


# 28. Calling a Function in an Exception Block
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Cannot divide by zero!"
    return result

print(divide(10, 2))  # Output: 5.0
print(divide(10, 0))  # Output: "Cannot divide by zero!"
