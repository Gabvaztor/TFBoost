def number_relationship_phi(x):
    """
    Return the number of relationship between sensors (φ) from the number of sensors (x).

    To calculate φ --> f(φ) = x + (f(φ(x-1)))
    Examples:
        - 1 sensor: φ1 = 1 + 0
        - 2 sensors: φ2 = 2 + φ(2-1) = 3
        - 3 sensors: φ3 = 3 + φ(3-1) = 6
    Args:
        x: Number of sensors
    Return: φ: Number of relationship between sensors
    """

    if x == 1:
        return 1
    else:
        return x + number_relationship_phi(x - 1)


def number_of_records_by_relationship_phi(number_of_sensors):
    return number_relationship_phi(number_of_sensors) * 365

