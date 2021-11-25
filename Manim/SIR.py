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
    "#FC6255",
    "#F8645B",
    "#F56660",
    "#F16866",
    "#ED6A6C",
    "#E96C71",
    "#E66E77",
    "#E2707D",
    "#DE7282",
    "#DA7488",
    "#D7768E",
    "#D37793",
    "#CF7999",
    "#CB7B9E",
    "#C87DA4",
    "#C47FAA",
    "#C081AF",
    "#BC83B5",
    "#B985BB",
    "#B587C0",
    "#B189C6"
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

class ExponentialDecay(Scene):
    def construct(self):
        def add_exponential_graph(self, axes, color, c, k, N):
            b = (10/c) - 1
            f = lambda t : N/(1 + b*math.e**(-k*t))
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
        N = 10 # Limiting Capacity
        c = 1 # Desired initial population
        k = 0.25 # Rate of Growth

        # Create one graph and add it to Axes:
        self.play(Write(add_exponential_graph(self, ax, Colors[c], c, k, N)))

        # Add a point and track the value.

        # Create other solutions
        # Colors for the lines
        for i in range(1, 40, 1):
            if i != c*2:
                graphs = add_exponential_graph(self, ax, Colors[i], i/2, k, N)
                self.play(Write(graphs), run_time=1)

class PredatorPreyModelWrong(MovingCameraScene):
    def construct(self):

        # Standard axes
        ax = Axes(
            tips=False
        )
        self.play(Write(ax))

        # Vector Field
        colors = [RED, YELLOW, BLUE, DARK_GRAY]
        f = lambda pos : (2 * pos[0] - 1.6 * pos[0] * pos[1])* RIGHT + (-1 * pos[1] + 0.8 * pos[1] * pos[0]) * UP
        vector_field = ArrowVectorField(
            f,
            colors=Colors
            )
        self.play(Write(vector_field))
        self.wait(1)
        self.play(Uncreate(ax), Uncreate(vector_field))


class PredatorPreyModelGraph(Scene):
    def construct(self):
        f = lambda pos : (2 * pos[0] - 1.6 * pos[0] * pos[1])* RIGHT + (-1 * pos[1] + 0.8 * pos[1] * pos[0]) * UP
        
        vector_field = ArrowVectorField(
            f,
            colors=Colors
            )
        
        ax = Axes(
            x_range = [0, 10, 1],
            y_range = [0, 10, 1],
            tips=False
        )

        vector_field = ArrowVectorField(
            f,
            x_range = [0, 11.5, 0.5],
            y_range = [0, 6, 0.5]
        )
        dot = Dot(point = ax.coords_to_point(0.0195, 0.036))

        vector_field.align_to(dot, DL)
        self.play(Write(ax), Write(vector_field))

        stream_lines = StreamLines(
            f,
            x_range = [0.1, 15, 0.5],
            y_range = [0.15, 10, 0.5],
            virtual_time = 1.5,
            opacity = 1,
            stroke_width=0.7,
            color = PURPLE
        )
        stream_lines.align_to(dot, DL)
        stream_lines.shift(UP*0.05)
        self.add(stream_lines)
        stream_lines.start_animation(warm_up=True, flow_speed=1)
        self.wait(10)
        self.play(Uncreate(stream_lines))

class PredatorPreyModelRF(Scene):
    def construct(self):
        ax = Axes(
            x_range = [0, 10, 1],
            y_range = [0, 10, 1],
            tips = False
        )
        print('Nothing yet')


if __name__ == "__main__":
    os.system('cls')
    os.system('manim ".\SIR.py" PredatorPreyModel -p')
    #os.system('manim ".\SIR.py" PredatorPreyModel -sp')
    #os.system('manim ".\SIR.py" PredatorPreyModel -q k -p')