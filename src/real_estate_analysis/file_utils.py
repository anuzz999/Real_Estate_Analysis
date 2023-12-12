def save_to_file(data, filename):
    """
    Saves the given data to a file.

    This function takes any data, converts it to a string, and writes it to a file.
    It is primarily used to save structured data like a list of URLs to a text file.

    Parameters:
    data: The data to be saved. Could be of any type that can be converted to a string.
    filename (str): The name of the file where the data will be saved.

    Returns:
    None
    """
    with open(filename, "w") as output:
        output.write(str(data))


def load_from_file(filename):
    """
    Loads and converts data from a file back to its original format.

    This function reads data from a file, assuming the data was stored as a string
    representation of its original format (for example, a string representation of a list).
    It then converts the data back to its original format.

    Parameters:
    filename (str): The name of the file from which to load the data.

    Returns:
    The data in its original format.
    """
    with open(filename, "r") as input_file:
        data = input_file.read()
        # Convert the data back to the original format if necessary
        # For example, if it's a list of URLs stored as a string
        data_list = eval(data)
        return data_list
