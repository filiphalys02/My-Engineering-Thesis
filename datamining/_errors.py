from functools import wraps


def _validate_argument_types1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        annotations = func.__annotations__

        for arg, (name, expected_type) in zip(args, annotations.items()):
            if expected_type and not isinstance(arg, expected_type):
                raise TypeError(f"Argument '{name}' must be {expected_type}, got {type(arg)} instead.")

        return func(*args, **kwargs)
    return wrapper


@_validate_argument_types1
def _validate_inputs(dic: dict):
    """
    :param dictionary: dict
        dictionary constructed in the following way:
            key: "argument-name"
            value: ["expected type", argument] or value: ["expected type or expected type", argument]
    :return:
        raise TypeError if any argument has unexpected type
    """
    for key, value in dic.items():
        expected_type = value[0]
        actual_type = str(type(value[1]).__name__)
        if actual_type not in expected_type:
            raise TypeError(f"Argument '{key}' must be of type '{expected_type}', not '{actual_type}'")