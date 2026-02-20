"""This module contains all custom ops for spyre"""

from . import rms_norm


def register_all():
    print("registering custom ops for spyre")
    rms_norm.register()
