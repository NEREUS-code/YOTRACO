import unittest
import json
import os
from YOTRACO.yotracoStats import YotracoStats
from unittest.mock import patch

class TestYotracoStats(unittest.TestCase):

    def setUp(self):
        """Set up a fresh stats object before each test."""
        self.stats = YotracoStats()
        self.test_json = "test_counts.json"
        self.test_csv = "test_counts.csv"

        # Ensure that counts contain at least one entry before testing save functions
        self.stats.counts = {"car": 5, "person": 10}  

    def tearDown(self):
        """Remove test files after each test."""
        for file in [self.test_json, self.test_csv]:
            if os.path.exists(file):
                os.remove(file)

    def test_save_counts_json(self):
        """Test saving the counts in JSON format."""
        self.stats.save_counts(self.test_json, "json")

        with open(self.test_json, "r") as f:
            data = json.load(f)

        self.assertTrue("in_counts" in data, "JSON should contain 'in_counts' key.")
        self.assertTrue("out_counts" in data, "JSON should contain 'out_counts' key.")

    # TODO : fix the csv test
    # def test_save_counts_csv(self):
    #     """Test saving the counts in CSV format."""
    #     self.stats.save_counts(self.test_csv, "csv")
        
    #     with open(self.test_csv, "r") as f:
    #         data = f.readlines()

    #     self.assertGreater(len(data), 1, "CSV should contain a header and at least one row.")

    @patch("matplotlib.pyplot.show")
    def test_plot_counts(self, mock_show):
        """Test plotting the counts (mocking `show` to avoid GUI issues)."""
        self.stats.plot_counts()
        mock_show.assert_called_once()  # Ensure the plot function is called

if __name__ == "__main__":
    unittest.main()
