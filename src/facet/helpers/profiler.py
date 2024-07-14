import time

import matplotlib.pyplot as plt


class Profiler:
    def __init__(self):
        self.results = []
        self.titles = []

    def start(self, title):
        self.start_time = time.time()
        self.titles.append(title)

    def stop(self):
        elapsed_time = time.time() - self.start_time
        self.results.append(elapsed_time)

    def get_results(self):
        return self.results

    def plot_results(self, title="Profiler Results"):
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
        ]  # Add more colors if needed
        plt.bar(self.titles, self.results, color=colors)
        plt.xlabel("Title")
        plt.ylabel("Elapsed Time (s)")
        plt.title(title)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
        plt.show()
