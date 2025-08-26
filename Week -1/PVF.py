def MyInput():
    UserInput = input("Enter a binary sequence in the format of (1,0,0,1): ")
    Cleaned = UserInput.strip().replace("(", "").replace(")", "").replace(" ", "")
    bits = tuple(int(x) for x in Cleaned.split(",") if x != "")

    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"Invalid bit at position {i}: {b}. Only 0 and 1 are allowed.")

    return bits


def BinaryToDecimal(bits): 
    n = len(bits)
    total = 0
    for i, b in enumerate(bits):
        total += b * (2 ** (n - i - 1))
    return total

def in_bounds(value, n):
    return 0 <= value <= (2 ** n - 1)

bits = MyInput()
n = len(bits)
decimal_value = BinaryToDecimal(bits)
print(f"Binary {bits} = Decimal {decimal_value}")

if in_bounds(decimal_value, n):
    print(f"{decimal_value} is within [0, {2**n - 1}]")
else:
    print(f"{decimal_value} is OUTSIDE [0, {2**n - 1}]")
