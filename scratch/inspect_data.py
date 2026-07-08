with open("data/raw/capa1.txt", "r") as f:
    content = f.read()

tokens = content.split()
print("Header:", tokens[:2])
print("First facility:", tokens[2:4])
print("Last facility:", tokens[198:202])

print("\nFirst customer block tokens:")
# The first customer block should start at token 202
# Let's print the first 110 tokens of the customer block
print("Demand of customer 0:", tokens[202])
print("Costs to customer 0 (first 10):", tokens[203:213])
print("Costs to customer 0 (last 10):", tokens[303:313])
print("Next token (should be customer 1 demand):", tokens[303])
