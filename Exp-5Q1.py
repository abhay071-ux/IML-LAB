# Dataset
weather = ['Sunny', 'Sunny', 'Rainy', 'Rainy']
play = ['Yes', 'Yes', 'No', 'No']

total = len(weather)

# Count classes
yes_count = play.count('Yes')
no_count = play.count('No')

# Prior probabilities
p_yes = yes_count / total
p_no = no_count / total

# Likelihood probabilities
sunny_yes = 0
sunny_no = 0

for i in range(total):
    if weather[i] == 'Sunny' and play[i] == 'Yes':
        sunny_yes += 1
    if weather[i] == 'Sunny' and play[i] == 'No':
        sunny_no += 1

p_sunny_given_yes = sunny_yes / yes_count
p_sunny_given_no = sunny_no / no_count

# Evidence
p_sunny = weather.count('Sunny') / total

# Posterior probabilities using Naive Bayes
p_yes_given_sunny = (p_sunny_given_yes * p_yes) / p_sunny
p_no_given_sunny = (p_sunny_given_no * p_no) / p_sunny

print("P(Yes|Sunny) =", p_yes_given_sunny)
print("P(No|Sunny) =", p_no_given_sunny)

# Prediction
if p_yes_given_sunny > p_no_given_sunny:
    print("Prediction: PlayGame = Yes")
else:
    print("Prediction: PlayGame = No")