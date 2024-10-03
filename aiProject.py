import random

# Define the search space based on the provided table
inventory = {
    "top": [
        {"name": "T-shirt", "price": 0.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 5},
        {"name": "Formal Shirt", "price": 120.0, "dress_code": "Business", "color": "Dark", "comfort_level": 3},
        {"name": "Polo Shirt", "price": 80.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 4},
        {"name": "Hoodie", "price": 60.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 4},
        {"name": "Evening Blouse", "price": 150.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
        {"name": "Sweater", "price": 0.0, "dress_code": "Casual", "color": "Dark", "comfort_level": 5},
        {"name": "Tank Top", "price": 0.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 4},
        {"name": "Silk Blouse", "price": 200.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
    ],
    "bottom": [
        {"name": "Jeans", "price": 0.0, "dress_code": "Casual", "color": "Dark", "comfort_level": 4},
        {"name": "Formal Trousers", "price": 150.0, "dress_code": "Business", "color": "Dark", "comfort_level": 3},
        {"name": "Skirt", "price": 100.0, "dress_code": "Evening", "color": "Bright", "comfort_level": 3},
        {"name": "Sports Shorts", "price": 0.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Chinos", "price": 90.0, "dress_code": "Business", "color": "Dark", "comfort_level": 4},
        {"name": "Leggings", "price": 60.0, "dress_code": "Casual", "color": "Dark", "comfort_level": 5},
        {"name": "Athletic Pants", "price": 80.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Evening Gown", "price": 250.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 1}
    ],
    "shoes": [
        {"name": "Sneakers", "price": 0.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Leather Shoes", "price": 180.0, "dress_code": "Business", "color": "Dark", "comfort_level": 2},
        {"name": "Running Shoes", "price": 120.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Ballet Flats", "price": 90.0, "dress_code": "Casual", "color": "Dark", "comfort_level": 4},
        {"name": "High Heels", "price": 250.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 2},
        {"name": "Sandals", "price": 0.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 5},
        {"name": "Loafers", "price": 150.0, "dress_code": "Business", "color": "Dark", "comfort_level": 3},
        {"name": "Evening Pumps", "price": 220.0, "dress_code": "Evening", "color": "Bright", "comfort_level": 2}
    ],
    "neck": [
        {"name": "Silk Scarf", "price": 70.0, "dress_code": "Business", "color": "Dark", "comfort_level": 3},
        {"name": "Sports Scarf", "price": 0.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 4},
        {"name": "Necklace", "price": 220.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
        {"name": "Casual Scarf", "price": 0.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 5},
        {"name": "Bow Tie", "price": 80.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
        {"name": "Athletic Headband", "price": 50.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Diamond Necklace", "price": 750.0, "dress_code": "Evening", "color": "Bright", "comfort_level": 3},
        {"name": "Choker", "price": 0.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 4}
    ],
    "purse": [
        {"name": "Clutch Bag", "price": 100.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
        {"name": "Canvas Bag", "price": 0.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 5},
        {"name": "Leather Briefcase", "price": 180.0, "dress_code": "Business", "color": "Dark", "comfort_level": 1},
        {"name": "Sports Backpack", "price": 80.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 5},
        {"name": "Tote Bag", "price": 0.0, "dress_code": "Casual", "color": "Bright", "comfort_level": 4},
        {"name": "Wristlet", "price": 150.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3},
        {"name": "Fanny Pack", "price": 50.0, "dress_code": "Sportswear", "color": "Bright", "comfort_level": 4},
        {"name": "Elegant Handbag", "price": 250.0, "dress_code": "Evening", "color": "Dark", "comfort_level": 3}
    ]
}

# Function to generate the initial population
def create_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            "top": random.choice(inventory["top"]),
            "bottom": random.choice(inventory["bottom"]),
            "shoes": random.choice(inventory["shoes"]),
            "neck": random.choice(inventory["neck"]),
            "purse": random.choice(inventory["purse"])
        }
        population.append(individual)
    return population

# Fitness function
def fitness(individual, user_preferences):
    dress_code_weight = 0.35
    budget_weight = 0.35
    color_palette_weight = 0.15
    comfort_weight = 0.15

    # Matching dress code
    dress_code_score = sum(1 for item in individual.values() if item['dress_code'] == user_preferences['dress_code']) / 5

    # Color palette match
    color_score = sum(1 for item in individual.values() if item['color'] == user_preferences['color_palette']) / 5

    # Comfort level match
    comfort_score = sum(1 for item in individual.values() if item['comfort_level'] >= user_preferences['comfort_level']) / 5

    # Budget evaluation
    total_price = sum(item['price'] for item in individual.values())
    if total_price <= user_preferences['budget']:
        budget_score = 1 
    elif user_preferences['budget'] < total_price <= user_preferences['budget'] * 2:
        budget_score = 1 - (user_preferences['budget'] / total_price)
    else:
        budget_score = 0  # Penalize if the outfit exceeds twice the budget
    

    # Weighted sum of all the scores
    fitness_value = (dress_code_score * dress_code_weight + 
                     budget_score * budget_weight + 
                     color_score * color_palette_weight + 
                     comfort_score * comfort_weight)
    return fitness_value

class Parent:
    def __init__(self, solution, fitness):
        self.solution = solution  # The outfit solution (e.g., the list of selected items)
        self.fitness = fitness    # The fitness score of the solution

    def __repr__(self):
        return f"Parent(Solution: {self.solution}, Fitness: {self.fitness})"

# Binary tournament selection
def select_parents(population, fitnesses):
    #Select two parents using binary tournament selection and return Parent objects.
    
    def tournament_selection():
        # Select two individuals randomly
        a, b = random.sample(range(len(population)), 2)
        
        # Choose the fittest between the two
        if fitnesses[a] > fitnesses[b]:
            return Parent(population[a], fitnesses[a])
        elif fitnesses[b] > fitnesses[a]:
            return Parent(population[b], fitnesses[b])
        else:
            # If fitness is equal, select randomly
            return random.choice([Parent(population[a], fitnesses[a]), Parent(population[b], fitnesses[b])])
    
    # Select two parents
    parent1 = tournament_selection()
    parent2 = tournament_selection()
    
    
    return parent1, parent2

def print_parent(parent, parent_num):
    """
    Print the details of a parent solution (outfit) in a formatted way.
    
    Args:
    parent (object): The parent object containing the outfit (solution) and fitness.
    parent_num (int): The parent number to be printed (e.g., Parent1, Parent2).
    """
    print(f"Parent{parent_num}:")
    
    # Iterate through each category (top, bottom, etc.) in the parent solution
    for category, item in parent.solution.items():
        print(f"  {category}: {item['name']} - price: {item['price']} - dress_code: {item['dress_code']} - "
              f"color: {item['color']} - comfort_level: {item['comfort_level']}")
    
    # Print the fitness level
    print(f"  Fitness level: {parent.fitness:.4f}")  # Printing fitness up to 4 decimal places for clarity
    print()  # Newline for formatting

def get_user_preferences():
    #Prompt the user to enter their preferences and return them as a dictionary.
    user_preferences = {}

    # Prompt for user name
    name = input("What is your name? ")

    # Prompt for dress code preference with validation
    valid_dress_codes = ['Casual', 'Sportswear', 'Business', 'Evening']
    while True:
        dress_code = input(f"Hi {name}! Please enter your dress code preference (Casual, Sportswear, Business, Evening): ")
        if dress_code in valid_dress_codes:
            user_preferences['dress_code'] = dress_code
            break
        else:
            print("Invalid input. Please choose from Casual, Sportswear, Business, Evening.")

    # Prompt for color palette preference with validation
    valid_color_palettes = ['Dark', 'Bright']
    while True:
        color_palette = input("Please enter your color palette preference (Dark, Bright): ")
        if color_palette in valid_color_palettes:
            user_preferences['color_palette'] = color_palette
            break
        else:
            print("Invalid input. Please choose from Dark or Bright.")

    # Prompt for comfort level with validation
    while True:
        try:
            comfort_level = int(input("Please enter your comfort level (1 - least comfortable, to 5 - most comfortable): "))
            if 1 <= comfort_level <= 5:
                user_preferences['comfort_level'] = comfort_level
                break
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Prompt for budget with validation
    while True:
        try:
            budget = float(input("Please enter your budget (in SAR): "))
            if budget >= 0:
                user_preferences['budget'] = budget
                break
            else:
                print("Invalid input. Please enter a non-negative number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print("\nThank you! We have saved your preferences.")
    return user_preferences

# Call the function and store the preferences
user_preferences = get_user_preferences()

# Main Genetic Algorithm  (for Phase 1)
population_size = 10 # Will be changed if any update to Phase1 happen 
#generations = 20 (for Phase 2)


# Create initial population
population = create_initial_population(population_size)

#Calculate fitness for each individual
fitnesses = [fitness(ind, user_preferences) for ind in population]

# Select parents using binary tournament
parent1, parent2 = select_parents(population, fitnesses)
# We will print the parents in Phase1 just for validation 
print_parent(parent1, 1)
print_parent(parent2, 2)


# Main Genetic Algorithm loop (for Phase 2)
# Evolutionary process
#for generation in range(generations):
    # Calculate fitness for each individual
    #fitnesses = [fitness(ind, user_preferences) for ind in population]

    # Select parents using binary tournament
    #parent1, parent2 = select_parents(population, fitnesses)
    #print_parent(parent1, 1)
    #print_parent(parent2, 2)

    # Cross-over, mutation, and replacement steps would go here (Phase 2)
