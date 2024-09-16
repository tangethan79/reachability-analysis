import numpy as np

def find_int_distance(r, a) -> float:
    coefficients = [(1 - a) / 12, 0, a * r**2, a*-4 / 3 * r**3]
    roots = np.roots(coefficients)
    roots = roots.tolist()
    distance = next((x for x in roots if 0 <= x <= 2*r), None)
    return distance

print(find_int_distance(1.5, 2))