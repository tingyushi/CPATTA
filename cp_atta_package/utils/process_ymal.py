import yaml
from munch import Munch

def convert_to_munch(obj):
    """
    Recursively convert a dictionary into a Munch object.
    
    Args:
        obj (dict): The input dictionary.

    Returns:
        Munch: A Munch object with the same structure as the input dictionary.
    """
    if isinstance(obj, dict):
        return Munch({key: convert_to_munch(value) for key, value in obj.items()})
    elif isinstance(obj, list):
        return tuple(convert_to_munch(item) for item in obj)
    else:
        return obj


def load_hyperparameters_as_munch(yaml_file_path: str) -> Munch:
    """
    Load a YAML file and convert the hyperparameters into a Munch object,
    including nested dictionaries.

    Args:
        yaml_file_path (str): Path to the YAML file containing hyperparameters.

    Returns:
        Munch: A Munch object containing all the hyperparameters, including nested ones.
    """
    # Load the YAML file
    with open(yaml_file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    # Convert the dictionary (including nested ones) to a Munch object
    return convert_to_munch(hyperparameters)


def pretty_print_munch(obj, indent=0):
    """
    Recursively print a nested Munch object or dictionary in a nicely formatted way.
    
    Args:
        obj (Munch or dict): The Munch object or dictionary to print.
        indent (int): Current indentation level (used for recursive calls).
    """
    # Iterate through each key-value pair
    for key, value in obj.items():
        # Print the key with the correct indentation
        print(' ' * indent + str(key) + ':', end=' ')
        
        # If the value is a Munch object or a dictionary, recurse into it
        if isinstance(value, (Munch, dict)):
            print()  # Move to the next line for better readability
            pretty_print_munch(value, indent + 4)
        # If the value is a list, handle it
        elif isinstance(value, list):
            print('[')
            for item in value:
                if isinstance(item, (Munch, dict)):
                    pretty_print_munch(item, indent + 4)
                else:
                    print(' ' * (indent + 4) + str(item))
            print(' ' * indent + ']')
        # Otherwise, just print the value
        else:
            print(str(value))
