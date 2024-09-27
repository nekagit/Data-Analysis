# 1. Grundlegende Funktionen
# ------------------------

# 1.1 Funktion ohne Argumente
def begruessung():
    print("Hallo, Welt!")

begruessung()

# 1.2 Funktion mit Positionsargumenten
def addiere(a, b):
    return a + b

print(addiere(4,3))

# 1.3 Funktion mit Positions- und Schlüsselwortargumenten
def vorstellen(name, alter):
    print(f"Name: {name}, Alter: {alter}")

vorstellen("Tobias", 28)

# 1.4 Funktion mit Standardargumentwert
def begruessung_mit_name(name="Gast"):
    print(f"Hallo, {name}!")

begruessung_mit_name()


# 1.5 Funktion mit Rückgabewert
def multipliziere(a, b):
    return a * b

ausgabeMultiplikation = multipliziere(3,4)
print(ausgabeMultiplikation)

# 2. Fortgeschrittene Funktionskonzepte
# ------------------------------------

# 2.1 Funktion mit beliebigen Argumenten (*args)
def summe_aller(*args):
    return sum(args)



# 2.2 Funktion mit beliebigen Schlüsselwortargumenten (**kwargs)
def kaeseladen(sorte, *argumente, **schluesselworte):
    print("-- Haben Sie", sorte, "?")
    print("-- Tut mir leid, wir haben keinen", sorte, "mehr")
    for arg in argumente:
        print(arg)
    print("-" * 40)
    for schluessel in schluesselworte:
        print(schluessel, ":", schluesselworte[schluessel])

# 2.3 Rekursive Funktion
def dreieck_rekursion(k):
    if k > 0:
        ergebnis = k + dreieck_rekursion(k - 1)
        print(ergebnis)
    else:
        ergebnis = 0
    return ergebnis

# 2.4 Funktion, die eine Funktion zurückgibt (Höhere Ordnung)
def erstelle_multiplizierer(x):
    def multiplizierer(y):
        return x * y
    return multiplizierer

# 2.5 Dekorator (Funktion, die Funktionen modifiziert)
def mein_dekorator(func):
    def wrapper():
        print("Etwas vor der Funktion.")
        func()
        print("Etwas nach der Funktion.")
    return wrapper

@mein_dekorator
def sage_hallo():
    print("Hallo!")

# 2.6 Generatorfunktion mit `yield`
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# 2.7 Partielle Funktionsanwendung
from functools import partial

def potenz(basis, exponent):
    return basis ** exponent

quadrat = partial(potenz, exponent=2)


# 3. Funktionen mit eingebauten Funktionen
# ---------------------------------------

# 3.1 Verwendung von `sorted()` mit Lambda-Funktion
def sortiere_liste(lst):
    return sorted(lst, key=lambda x: x)

# 3.2 Verwendung von `map()`
def quadriere(x):
    return x ** 2

zahlen = [1, 2, 3, 4, 5]
quadrierte_zahlen = list(map(quadriere, zahlen))

# 3.3 Verwendung von `filter()`
def ist_gerade(n):
    return n % 2 == 0

gerade_zahlen = list(filter(ist_gerade, zahlen))

# 3.4 Verwendung von `zip()`
# namen = ["Alice", "Bob", "Charlie"]
# alter = [25, 30, 35]
# kombiniert = list(zip(namen, alter))

# 3.5 Verwendung von `enumerate()`
# for index, name in enumerate(namen, start=1):
#     print(f"{index}: {name}")


# 4. Funktionen mit Bibliotheken
# -----------------------------

import os

# 4.1 Dateisysteminteraktion
def liste_dateien_im_verzeichnis(pfad='.'):
    return os.listdir(pfad)

# 4.2 Aktuelles Arbeitsverzeichnis
def hole_aktuelles_verzeichnis():
    return os.getcwd()

# 4.3 Pfadexistenz prüfen
def pruefe_pfad_existenz(pfad):
    return os.path.exists(pfad)


# 6. Fehlerbehandlung in Funktionen
# --------------------------------

# 6.1 Funktion mit Ausnahmebehandlung
def dividiere(a, b):
    try:
        ergebnis = a / b
    except ZeroDivisionError:
        return "Kann nicht durch Null teilen!"
    return ergebnis


# 7. Zusätzliche fortgeschrittene Konzepte
# ---------------------------------------

# 7.1 Closure (Funktionsabschluss)
def aeussere_funktion(x):
    def innere_funktion(y):
        return x + y
    return innere_funktion

addiere_5 = aeussere_funktion(5)
ergebnis = addiere_5(3)  # Ergebnis: 8

# 7.2 Memoization für rekursive Funktionen
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# 7.3 Funktion mit Typannotationen
def gruss_mit_alter(name: str, alter: int) -> str:
    return f"Hallo {name}, du bist {alter} Jahre alt."


# 7.5 Asynchrone Funktion
import asyncio

async def async_gruss(name):
    await asyncio.sleep(1)
    print(f"Hallo, {name}!")

# Verwendung:
# asyncio.run(async_gruss("Alice"))

# 7.6 Funktion mit optionalen Argumenten und Typprüfung
def komplexe_berechnung(a: int, b: int, operation: str = 'add', runden: bool = False) -> float:
    if operation not in ['add', 'subtract', 'multiply', 'divide']:
        raise ValueError("Ungültige Operation")
    
    if operation == 'add':
        ergebnis = a + b
    elif operation == 'subtract':
        ergebnis = a - b
    elif operation == 'multiply':
        ergebnis = a * b
    else:
        if b == 0:
            raise ZeroDivisionError("Kann nicht durch Null teilen")
        ergebnis = a / b
    
    return round(ergebnis) if runden else ergebnis

# 5. Lambda-Funktionen und List Comprehensions
# -------------------------------------------

# 5.1 Lambda-Funktion
multipliziere_mit_zwei = lambda x: x * 2

# 5.2 List Comprehension
quadrate = [x**2 for x in range(6)]

