from datamining._errors import _validate_argument_types


@_validate_argument_types
def add(a: int, b: int):
    return a + b
