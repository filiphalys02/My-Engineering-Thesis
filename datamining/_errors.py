from functools import wraps


def _validate_argument_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        annotations = func.__annotations__

        # for arg, (name, expected_type) in zip(args, annotations.items()):
        #     if expected_type and not isinstance(arg, expected_type):
        #         raise TypeError(f"Argument '{name}' must be {expected_type}, got {type(arg)} instead.")

        for name, value in kwargs.items():
            expected_type = annotations.get(name)
            if expected_type and not isinstance(value, expected_type):
                raise TypeError(f"Argument '{name}' must be {expected_type}, got {type(value)} instead.")

        return func(*args, **kwargs)
    return wrapper
