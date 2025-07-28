"""
Viewer

A class to help show the items in a object's container,
to avoid changing the original object.
"""

# pylint: disable=import-outside-toplevel


class Viewer:
    """
    A class to help show the items in a object's container,
    to avoid changing the original object.
    """

    def __init__(self, obj):
        object.__setattr__(self, "obj", obj)

    def __getitem__(self, key):
        return Viewer(self.obj[key])

    def __setitem__(self, key, value):
        raise TypeError("Viewer object is read-only")

    def __delitem__(self, key):
        raise TypeError("Viewer object is read-only")

    def __getattr__(self, attr):
        return Viewer(getattr(self.obj, attr))

    def __setattr__(self, attr, value):
        raise TypeError("Viewer object is read-only")

    def __delattr__(self, attr):
        raise TypeError("Viewer object is read-only")

    def __iter__(self):
        return (Viewer(item) for item in self.obj)

    def __len__(self):
        return len(self.obj)

    def __repr__(self):
        from .basic_class import BasicClass

        if BasicClass._constants("REPR_USE_VIEWER", "bool"):
            return f"Viewer({repr(self.obj)})"
        return repr(self.obj)

    def __str__(self):
        return str(self.obj)
