from sklearn.preprocessing import LabelEncoder


class Mapper(LabelEncoder):
    """Class Mapper to give convinient interface for LabelEncoder"""

    def __init__(self, labels):
        """Constructor for the class"""
        super().__setattr__('__dict__', {})

        # Initialises the encoder
        LabelEncoder.__init__(self)

        self.fit(labels)
        self.fitted = self.transform(labels)

    def __getattr__(self, key):
        """
        Gets class attribute
        Raises AttributeError if key is invalid
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        """
        Sets class attribute according to value
        If key was not found, new attribute is added
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)
