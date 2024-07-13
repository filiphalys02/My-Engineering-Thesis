from functools import wraps


def _validate_argument_types(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        annotations = func.__annotations__
        for arg, (name, expected_type) in zip(args, annotations.items()):
            if not isinstance(arg, expected_type):
                raise TypeError(f"Argument {name} must be {expected_type}")
        return func(*args, **kwargs)
    return wrapper
