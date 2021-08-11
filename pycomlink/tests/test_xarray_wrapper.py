from unittest import TestCase
from collections import OrderedDict
import pycomlink.processing.xarray_wrapper


def test_get_new_args_dict():
    def foo_func(a, b, c=None, d=1):
        return None

    expected = OrderedDict(
        [
            ("a", 123),
            ("b", "foo"),
            ("c", "bar"),
            ("d", 1),
        ]
    )
    result = pycomlink.processing.xarray_wrapper._get_new_args_dict(
        func=foo_func, args=[123, "foo", "bar", 1], kwargs={}
    )
    TestCase().assertDictEqual(expected, result)

    result = pycomlink.processing.xarray_wrapper._get_new_args_dict(
        func=foo_func,
        args=[
            123,
        ],
        kwargs={"b": "foo", "c": "bar"},
    )
    TestCase().assertDictEqual(expected, result)
