```
    Parse a line of text containing three floating-point numbers in scientific notation
    using a regular expression.

    This function uses the following regular expression pattern:
    `r'\s*([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)\s+([-+]?\d*\.\d+E[-+]?\d+)'`

    The pattern matches and extracts three numbers written in scientific notation
    (e.g., `-3.217780E-01`), typically representing X-Coordinate, Y-Coordinate, and
    Phase Periods, that are separated by spaces.

    Regular expression breakdown:
    
    - `r''`: Raw string notation to treat backslashes (`\`) as literal characters for regex.
    
    - `\s*`: Matches zero or more whitespace characters (spaces, tabs, newlines) before the numbers.
    
    - `([-+]?\d*\.\d+E[-+]?\d+)`: Matches a floating-point number in scientific notation:
        - `[-+]?`: Optional plus (`+`) or minus (`-`) sign.
        - `\d*`: Zero or more digits before the decimal point.
        - `\.`: Literal decimal point.
        - `\d+`: One or more digits after the decimal point.
        - `E[-+]\d+`: Scientific notation exponent (`E`), optional sign (`+` or `-`), followed by one or more digits.
    
    - `\s+`: One or more whitespace characters separating the three values.
    
    Purpose:
    This function looks for three numbers written in scientific notation and separated
    by spaces within the `line`. These numbers represent X-Coordinate, Y-Coordinate, 
    and Phase Periods.

    Example:
        Input: `-1.000000E-02  0.000000E+00   -3.217780E-01`
        Output: X-Coordinate = `-1.000000E-02`, Y-Coordinate = `0.000000E+00`, 
                Phase Periods = `-3.217780E-01`

    Args:
        line (str): A line of text containing three floating-point numbers in scientific notation.
    
    Returns:
        match (re.Match or None): A match object if the pattern is found, otherwise `None`.
        If matched, `match.group(1)`, `match.group(2)`, and `match.group(3)` will contain
        the X-Coordinate, Y-Coordinate, and Phase Periods values, respectively.
```

1. For convinience do not change the name of the directories inside the apexPHASE folder.

2. Now, the unit cell design outout is required to be brought to the full lens folder manually. This can be automated later.