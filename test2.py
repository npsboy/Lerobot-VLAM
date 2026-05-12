import json

with open("preprocessed_data.json", "r") as f:
    data = json.load(f)

X = data["X"]
Y = data["Y"]

print ("Len X:", len(X))
print ("Len X[0]:", len(X[0]) if X else "N/A")
print ("Len X[0][0]:", len(X[0][0]) if X and X[0] else "N/A")
print ("Len Y:", len(Y))
print ("Len Y[0]:", len(Y[0]) if Y else "N/A")