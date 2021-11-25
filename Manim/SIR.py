from manim import *
import os
import math

Colors = [
    "#58C4DD",
    "#61BFD6",
    "#69BACF",
    "#72B5C8",
    "#7BAFC0",
    "#83AAB9",
    "#8CA5B2",
    "#94A0AB",
    "#9D9BA4",
    "#A6969D",
    "#AE9095",
    "#B78B8E",
    "#C08687",
    "#C88180",
    "#D17C79",
    "#D97772",
    "#E2716A",
    "#EB6C63",
    "#F3675C",
    "#FC6255"
]

Colors_ten = [
    "#58C4DD",
    "#6AB9CE",
    "#7CAEBF",
    "#8FA3B0",
    "#A198A1",
    "#B38E91",
    "#C58382",
    "#D87873",
    "#EA6D64",
    "#FC6255"
]

class ExponentialGrowth(Scene):
    def construct(self):
        def add_exponential_graph(self, axes, color, c, k):
            f = lambda t : c*math.e**(k*t)
            graph = axes.plot(
                f, x_range=[0, 20], color = color
            )
            return graph
        
        # Create our exponential graph
        ax = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 20, 2],
            tips=False,
            axis_config={"include_numbers": True}
        )
        labels = ax.get_axis_labels(x_label="t", y_label="P")
        self.play(Create(ax), Create(labels), run_time=2)
        self.wait(1)


        # Our exponential parameters
        c = 1 # Initial Condition
        k = 0.15 # Rate of Growth

        # Create one graph and add it to Axes:
        self.play(Write(add_exponential_graph(self, ax, "#69BACF", c, k)))

        # Add a point and track the value.

        # Create other solutions
        for i in range(0, 20, 1):
            if i != 2:
                graphs = add_exponential_graph(self, ax, Colors[i], i/2, k)
                self.play(Write(graphs), run_time=1)

class ExponentialGrowth(Scene):
    def construct(self):
        def add_exponential_graph(self, axes, color, c, k):
            f = lambda t : c*math.e**(k*t)
            graph = axes.plot(
                f, x_range=[0, 20], color = color
            )
            return graph
        
        # Create our exponential graph
        ax = Axes(
            x_range=[0, 20, 2],
            y_range=[0, 20, 2],
            tips=False,
            axis_config={"include_numbers": True}
        )
        labels = ax.get_axis_labels(x_label="t", y_label="P")
        self.play(Create(ax), Create(labels), run_time=2)
        self.wait(1)


        # Our exponential parameters
        c = 1 # Initial Condition
        k = 0.15 # Rate of Growth

        # Create one graph and add it to Axes:
        self.play(Write(add_exponential_graph(self, ax, "#69BACF", c, k)))

        # Add a point and track the value.

        # Create other solutions
        # Colors for the lines

        for i in range(0, 20, 1):
            if i != 2:
                graphs = add_exponential_graph(self, ax, Colors[i], i/2, k)
                self.play(Write(graphs), run_time=1)


if __name__ == "__main__":
    os.system('cls')
    os.system('manim ".\SIR.py" ExponentialGrowth -sp')
    #os.system('manim ".\SIR.py" ExponentialGrowth -q k -p')