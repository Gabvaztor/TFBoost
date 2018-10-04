def data_integrity(data, *args, **kwargs):
    """
    Check data integrity
    This must check if it is possible to get conclusions from a type of data.

    Args:
        data: Data to be analyzed
        *args: Other flags as type of data or type of analysis
        **kwargs: Other flags as type of data or type of analysis

    Returns: This must return a Boolean:
        - False: it is not a good idea to analyse data because:
            * There are missing values.
            * There are noise values.
            * Other cases
        - True: If the data is healthy
    """

    pass