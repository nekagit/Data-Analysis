import pandas as pd
import matplotlib.pyplot as plt

# Daten erstellen
data = {
    'Employee ID': [101, 102, 103, 104, 105],
    'First Name': ['John', 'Jane', 'Michael', 'Emily', 'David'],
    'Last Name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown'],
    'Department': ['Marketing', 'Sales', 'Finance', 'HR', 'IT'],
    'Position': ['Manager', 'Associate', 'Analyst', 'Coordinator', 'Developer'],
    'Salary': [50000, 35000, 45000, 40000, 55000],
    'Start Date': ['2020-01-15', '2019-05-01', '2021-03-10', '2018-11-01', '2022-02-15']
}

df = pd.DataFrame(data)

# **Einfach (Easy) Aufgaben:**
# 1. Wähle die Spalten „Vorname“ und „Abteilung“ aus.
print("1. Spalten „Vorname“ und „Abteilung“:")
print(df[['First Name', 'Department']])

# 2. Filtere die Mitarbeiter, die im „Marketing“ arbeiten.
print("\n2. Mitarbeiter im „Marketing“:")
print(df[df['Department'] == 'Marketing'])

# 3. Benenne die Spalte „Salary“ in „Gehalt“ um.
df.rename(columns={'Salary': 'Gehalt'}, inplace=True)
print("\n3. Spalte „Salary“ in „Gehalt“ umbenannt:")
print(df)

# 4. Berechne den Durchschnittswert des Gehalts.
print("\n4. Durchschnittswert des Gehalts:")
print(df['Gehalt'].mean())

# **Mittel (Middle) Aufgaben:**
# 1. Filtere alle Mitarbeiter mit einem Gehalt von mehr als 40.000 und sortiere sie nach Gehalt in absteigender Reihenfolge.
print("\n1. Mitarbeiter mit Gehalt > 40.000, sortiert nach Gehalt:")
print(df[df['Gehalt'] > 40000].sort_values(by='Gehalt', ascending=False))

# 2. Gruppiere die Mitarbeiter nach Abteilung und berechne das durchschnittliche Gehalt pro Abteilung.
print("\n2. Durchschnittliches Gehalt pro Abteilung:")
print(df.groupby('Department')['Gehalt'].mean())

# **Mittel (Middle) Aufgaben:**
# 3. Erstelle ein Balkendiagramm, das das Gehalt für jeden Mitarbeiter zeigt.
df.plot(kind='bar', x='First Name', y='Gehalt', title='Gehalt der Mitarbeiter', legend=False)
plt.xlabel('Vorname')
plt.ylabel('Gehalt')
plt.xticks(rotation=45)  # Optional: Rotiert die X-Achsen-Beschriftungen für bessere Lesbarkeit
plt.tight_layout()  # Optional: Stellt sicher, dass Layout eng und sauber ist

# Speichern des Diagramms als Bild
plt.savefig('gehalt_der_mitarbeiter.png')


# 4. Setze für fehlende Gehälter den Durchschnittswert der „Finance“-Abteilung ein (falls vorhanden).
departmentColum = df['Department']
df.loc[departmentColum == 'Finance', 'Gehalt'] = df.loc[departmentColum == 'Finance', 'Gehalt'].fillna(df[departmentColum == 'Finance']['Gehalt'].mean())
print("\n4. Fehlende Gehälter in der „Finance“-Abteilung gefüllt:")
print(df)

# **Schwierig (Hard) Aufgaben:**
# 1. Füge eine neue Tabelle mit Boni pro Mitarbeiter basierend auf der Mitarbeiter-ID hinzu. Berechne dann das Gesamtgehalt (Gehalt + Bonus).
bonus_df = pd.DataFrame({
    'Employee ID': [101, 102, 103, 104],
    'Bonus': [5000, 3000, 4000, 2500]
})
merged_df = pd.merge(df, bonus_df, on='Employee ID')
merged_df['Total Salary'] = merged_df['Gehalt'] + merged_df['Bonus']
print("\n1. Gesamtgehalt (Gehalt + Bonus):")
print(merged_df) 

# 2. Berechne eine neue Spalte „Gehalt + Erhöhung“, bei der das Gehalt um 10% erhöht wird, falls der Mitarbeiter im „Marketing“ arbeitet, und um 5% in allen anderen Abteilungen.
df['Gehalt + Erhöhung'] = df['Gehalt'] * df['Department'].apply(lambda x: 1.10 if x == 'Marketing' else 1.05)
print("\n2. Gehalt + Erhöhung:")
print(df)

# 3. Erstelle eine Pivot-Tabelle, die das durchschnittliche Gehalt pro Abteilung und Position zeigt.
pivot_table = df.pivot_table(values='Gehalt', index='Department', columns='Position', aggfunc='mean')
print("\n3. Pivot-Tabelle (durchschnittliches Gehalt pro Abteilung und Position):")
print(pivot_table)

# 4. Filtere Mitarbeiter, die entweder mehr als 45.000 verdienen oder im „HR“ arbeiten und sortiere nach Gehalt.
print("\n4. Mitarbeiter mit Gehalt > 45.000 oder im „HR“, sortiert nach Gehalt:")
print(df[(df['Gehalt'] > 45000) | (df['Department'] == 'HR')].sort_values(by='Gehalt', ascending=False))
