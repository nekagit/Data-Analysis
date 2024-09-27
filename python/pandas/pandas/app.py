import pandas as pd
import matplotlib.pyplot as plt

def print_section(title):
    print(f"\n{'=' * 50}")
    print(f"{title.center(50)}")
    print(f"{'=' * 50}\n")

# Slide 2: Was ist Pandas?
print_section("Was ist Pandas?")
print("Pandas ist eine leistungsfähige Open-Source-Python-Bibliothek für Datenanalyse und -manipulation.")

# Slide 5: Grundstruktur von Pandas
print_section("Grundstruktur von Pandas")
data = {
    'Employee ID': [101, 102, 103, 104],
    'First Name': ['John', 'Jane', 'Michael', 'Emily'],
    'Last Name': ['Doe', 'Smith', 'Johnson', 'Williams'],
    'Department': ['Marketing', 'Sales', 'Finance', 'HR'],
    'Position': ['Manager', 'Associate', 'Analyst', 'Coordinator'],
    'Salary': [50000, 35000, 45000, 40000]
}
df = pd.DataFrame(data)
print("DataFrame erstellt:")
print(df)

# Slide 9: Daten anzeigen
print_section("Daten anzeigen")
print("Erste Zeilen des DataFrames:")
print(df.head(3))

# Slide 10: Daten filtern
print_section("Daten filtern")
print("Mitarbeiter mit einem Gehalt über 40.000:")
high_salary = df[df['Salary'] > 40000]
print(high_salary)

# Slide 11: Daten sortieren
print_section("Daten sortieren")
print("Sortierte Daten nach Nachnamen:")
sorted_df = df.sort_values(by='Last Name')
print(sorted_df)

# Slide 12: Fehlende Daten behandeln
print_section("Fehlende Daten behandeln")
df.loc[2, 'Salary'] = None
print("DataFrame mit fehlendem Wert:")
print(df)
print("\nDataFrame nach Ersetzen fehlender Werte:")
print(df.fillna(0))

# Slide 13: Grundlegende Statistiken
print_section("Grundlegende Statistiken")
print("Beschreibende Statistik:")
print(df.describe())

# Slide 14: Gruppieren von Daten
print_section("Gruppieren von Daten")
print("Durchschnittsgehalt pro Abteilung:")
grouped_salary = df.groupby('Department')['Salary'].mean()
print(grouped_salary)

# Slide 15: Bedingte Auswahl von Daten
print_section("Bedingte Auswahl von Daten")
print("Filtern nach Position 'Analyst' und Abteilung 'Finance':")
filtered_data = df[(df['Position'] == 'Analyst') & (df['Department'] == 'Finance')]
print(filtered_data)

# Slide 16: Datenvisualisierung
print_section("Datenvisualisierung")
plt.figure(figsize=(10, 5))
plt.bar(df['First Name'], df['Salary'])
plt.title('Gehälter der Mitarbeiter')
plt.xlabel('Mitarbeiter')
plt.ylabel('Gehalt')
plt.savefig('salary_bar_chart.png')
print("Balkendiagramm wurde als 'salary_bar_chart.png' gespeichert.")

# Slide 23: Daten kombinieren - Zusammenfügen von DataFrames (Merge)
print_section("Daten kombinieren")
bonus_data = {
    'Employee ID': [101, 102, 103, 104],
    'Bonus': [5000, 3000, 4000, 3500]
}
df_bonus = pd.DataFrame(bonus_data)
merged_df = pd.merge(df, df_bonus, on='Employee ID')
print("Zusammengeführte Daten (Gehälter + Boni):")
print(merged_df)

# Slide 24: Daten aneinander anhängen
print_section("Daten aneinander anhängen")
new_data = {
    'Employee ID': [105],
    'First Name': ['Sarah'],
    'Last Name': ['Brown'],
    'Department': ['Marketing'],
    'Position': ['Associate'],
    'Salary': [37000]
}
df_new = pd.DataFrame(new_data)
df_appended = df._append(df_new, ignore_index=True)
print("DataFrame nach dem Anhängen neuer Daten:")
print(df_appended)

# Neue Themen

# Slide 25: Daten transformieren
print_section("Daten transformieren")
df['Salary_EUR'] = df['Salary'].apply(lambda x: x * 0.85 if pd.notnull(x) else None)
print("DataFrame mit neuer Spalte (Gehalt in EUR):")
print(df)

# Slide 26: Daten aggregieren
print_section("Daten aggregieren")
agg_data = df.groupby('Department').agg({
    'Salary': ['mean', 'min', 'max'],
    'Employee ID': 'count'
}).reset_index()
agg_data.columns = ['Department', 'Avg_Salary', 'Min_Salary', 'Max_Salary', 'Employee_Count']
print("Aggregierte Daten nach Abteilung:")
print(agg_data)

# Slide 27: Zeitreihendaten
print_section("Zeitreihendaten")
date_range = pd.date_range(start='2023-01-01', periods=5, freq='D')
time_series_data = pd.Series(range(5), index=date_range)
print("Zeitreihendaten:")
print(time_series_data)

# Slide 28: Daten exportieren
print_section("Daten exportieren")
df.to_csv('employee_data.csv', index=False)
print("Daten wurden als 'employee_data.csv' exportiert.")

# Slide 29: Daten importieren
print_section("Daten importieren")
imported_df = pd.read_csv('employee_data.csv')
print("Importierte Daten aus 'employee_data.csv':")
print(imported_df.head())

print("\nDas Tutorial ist abgeschlossen.")