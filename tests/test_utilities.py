"""
Test the utilities of the project, such as the performance boost and the loading of
configuration files.
"""

import unittest

from YACS.yacs import Config
from soft_computing.utilities.reproducibility import load_configuration
from soft_computing.utilities.performance import is_debugger_active, performance_boost


class TestUtilities(unittest.TestCase):
    """
    Test the utilities of the project, such as the performance boost and the loading of
    configuration files.
    """

    def test_performance_boost(self) -> None:
        """
        Test that a performance boost is enabled when the debugger is not active, or vice-versa.

        Returns:
            None
        """
        self.assertEqual(not performance_boost(), is_debugger_active())

    def test_load_configuration(self) -> None:
        """
        Test that the configuration is loaded correctly (e.g., the learning rate is a float).

        Returns:
            None
        """
        config: Config = load_configuration()
        self.assertTrue(isinstance(config.training.learning_rate, float))
