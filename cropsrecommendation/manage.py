#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cropsrecommendation.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    # from public.algorithm.rf import RandomForest
    # from public.algorithm.rf import DecisionTree
    # from public.algorithm.rf import Node
    # from public.algorithm.svm import SVM
    # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    main()
