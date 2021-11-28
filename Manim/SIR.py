from manim import *
import os
import math
#import numpy as np
from scipy.integrate import odeint

COLOR_MAP = {
    "S": BLUE,
    "I": RED,
    "R": GREY_D,
}

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

# Infects 1-1 #
class Relationship(VGroup):
    def __init__(self, 
        amount = 1,
        preinfect = 0,
        **kwargs):
        # Initializes amount of pairs (n pairs of infected:healthy)
        self.amount = amount
        self.preinfect = preinfect
        self.healthy = VGroup()
        self.infected = VGroup()
        self.infected_susceptible = self.infected.copy()

        # Populate with dots
        self.healthy.add(*[Dot(color = BLUE) for n in range(amount - preinfect)])
        self.healthy.add(*[Dot(color = RED) for n in range(preinfect)])
        self.infected.add(*[Dot(color = RED) for n in range(amount)])

        # Arrange into a grid
        self.healthy.arrange_in_grid(rows=amount)
        self.infected.arrange_in_grid(rows=amount)

        # Arrange both pairs:
        self.pair = self.arrange_pair()
        
        # Finish
        super().__init__(**kwargs)

    def arrange_pair(self):
        return VGroup(self.infected, self.healthy).arrange_in_grid(cols=2)

class DotsExponentialGrowth(Scene):
    def infect(self, A, B): # Healthy, Infected
        C = B.copy() # Copy Infected
        self.play(C.animate(lag_ratio=0.1).move_to(A))
        self.play(Uncreate(A), run_time = 0)
        return C

    def construct(self):
         self.A = Relationship(amount = 1)
         self.B = Relationship(amount = 2)
         self.C = Relationship(amount = 4)
         self.D = Relationship(amount = 8)
         self.E = Relationship(amount = 16, preinfect = 5)

         examples = VGroup(self.A.pair, self.B.pair, self.C.pair, self.D.pair, self.E.pair).align_to(self.E.pair, DOWN)
         examples_full = VGroup(self.A, self.B, self.C, self.D, self.E)
         examples.arrange_in_grid(cols=5, cell_alignment=DOWN, buff=1)
         self.add(examples) # Animate this

         for item in examples_full:
             item.susceptible_infected = self.infect(item.healthy, item.infected)
             self.wait(0.5)

class DotsLogisticGrowth(Scene):
    def infect(self, A, B): # Healthy, Infected
        C = B.copy() # Copy Infected
        self.play(C.animate(lag_ratio=0.1).move_to(A))
        self.play(Uncreate(A), run_time = 0)
        return C

    def construct(self):
         self.A = Relationship(amount = 1)
         self.B = Relationship(amount = 2, preinfect = 1)
         self.C = Relationship(amount = 4, preinfect = 2)
         self.D = Relationship(amount = 8, preinfect = 4)
         self.E = Relationship(amount = 16, preinfect = 8)

         examples = VGroup(self.A.pair, self.B.pair, self.C.pair, self.D.pair, self.E.pair).align_to(self.E.pair, DOWN)
         examples_full = VGroup(self.A, self.B, self.C, self.D, self.E)
         examples.arrange_in_grid(cols=5, cell_alignment=DOWN, buff=1)
         self.add(examples) # Animate this
         self.wait(2)

         for item in examples_full:
             item.susceptible_infected = self.infect(item.healthy, item.infected)

# Calculates 100% infections from 0.001% #
## SCRAPPED, REDO AFTERWARDS ##
class SIRGraph(VGroup):
    def __init__(
        self,
        color_map = COLOR_MAP,
        height = 7,
        width = 5, 
        **kwargs
        ):
        color_map = color_map
        height = height,
        width = width
        super().__init__(**kwargs)
        self.add_axes()
        self.add_x_labels()
    
    def add_axes(self):
        axes = Axes(
            y_range=[0, 1],
            y_axis_config={
                "include_numbers": True,
                "numbers_to_include": np.arange(0, 1.001, 0.1),
                "decimal_number_config": {"num_decimal_places": 1}
            },
            x_range=[0, 1],
            axis_config={
                "include_tip": False,
            }
        )
        self.axes = axes
    def add_x_labels():
        self.x_labels = VGroup()
        self.x_ticks = VGroup()

### SIR GRAPHS ###
# Always 2x infections
class SIRGraphExponentialInfections_zeroth(Scene):
    def construct(self):
        # Total Population, N:
        N = 100000
        # Initial Number of infected and recovered individuals, I0 and R0
        P0 = 1
        
        # Contact Rate, beta, and mean recovery rate, gamma
        k = 0.6931471805599454
        # A grid of time points
        t = np.linspace(0, 365, 366)
        #t = np.linspace(0, 160, 161)

        # SIR Model
        def deriv(y, t, N, k):
            P = y
            dPdt = k*P
            return dPdt
        
        # Initial conditions vector
        y0 = P0

        # Integrate the SIR equations over the time grid, t
        ret = odeint(deriv, y0, t, args=(N, k))
        P = ret

        # Manim Stuff
        # Add base axes
        ax = Axes(
            y_range=[0, 1],
            y_axis_config={
                "include_numbers": True,
                "numbers_to_include": np.arange(0, 1.001, 0.1),
                "decimal_number_config": {"num_decimal_places": 1}
            },
            x_range=[0, 20, 2],
            x_axis_config={
                "include_numbers":True,
                "decimal_number_config": {"num_decimal_places": 0}
            },
            axis_config={
                "include_tip": False,
            }
        )
        self.add(ax)

        # Add graph
        graph_P = ax.plot_line_graph(t, P/N, line_color=RED, add_vertex_dots = False)
        self.play(
            Create(graph_P),
            run_time = 10    
        )

# Pure infections, no recoveries
class SIRGraphExponentialInfections_first(Scene):
    def construct(self):
        # Total Population, N:
        N = 100000
        # Initial Number of infected and recovered individuals, I0 and R0
        I0, R0 = 1, 0
        # Everyone else
        S0 = N - I0 - R0
        # Contact Rate, beta, and mean recovery rate, gamma
        beta, gamma = 0.6931471805599454, 0
        # A grid of time points
        t = np.linspace(0, 365, 366)
        #t = np.linspace(0, 160, 161)

        # SIR Model
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        
        # Initial conditions vector
        y0 = S0, I0, R0

        # Integrate the SIR equations over the time grid, t
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T

        # Manim Stuff
        # Add base axes
        ax = Axes(
            y_range=[0, 1],
            y_axis_config={
                "include_numbers": True,
                "numbers_to_include": np.arange(0, 1.001, 0.1),
                "decimal_number_config": {"num_decimal_places": 1}
            },
            x_range=[0, t[-1], t[-1]/8],
            x_axis_config={
                "include_numbers":True,
                "decimal_number_config": {"num_decimal_places": 0}
            },
            axis_config={
                "include_tip": False,
            }
        )
        self.add(ax)

        # Add graph
        graph_S = ax.plot_line_graph(t, S/N, line_color=BLUE, add_vertex_dots = False)
        graph_I = ax.plot_line_graph(t, I/N, line_color=RED, add_vertex_dots = False)
        graph_R = ax.plot_line_graph(t, R/N, line_color=GRAY_D, add_vertex_dots = False)
        self.play(
            Create(graph_S),
            Create(graph_I),
            Create(graph_R),
            run_time = 10         
        )

class SIRGraphExponentialInfections_second(Scene):
    def construct(self):
        # Total Population, N:
        N = 100000
        # Initial Number of infected and recovered individuals, I0 and R0
        I0, R0 = 1, 0
        # Everyone else
        S0 = N - I0 - R0
        # Contact Rate, beta, and mean recovery rate, gamma
        beta, gamma = 0.6931471805599454, 0
        # A grid of time points
        t = np.linspace(0, 32, 33)
        #t = np.linspace(0, 160, 161)

        # SIR Model
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        
        # Initial conditions vector
        y0 = S0, I0, R0

        # Integrate the SIR equations over the time grid, t
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T

        # Manim Stuff
        # Add base axes
        ax = Axes(
            y_range=[0, 1],
            y_axis_config={
                "include_numbers": True,
                "numbers_to_include": np.arange(0, 1.001, 0.1),
                "decimal_number_config": {"num_decimal_places": 1}
            },
            x_range=[0, t[-1], t[-1]/8],
            x_axis_config={
                "include_numbers":True,
                "decimal_number_config": {"num_decimal_places": 0}
            },
            axis_config={
                "include_tip": False,
            }
        )
        self.add(ax)

        # Add graph
        graph_S = ax.plot_line_graph(t, S/N, line_color=BLUE, add_vertex_dots = False)
        graph_I = ax.plot_line_graph(t, I/N, line_color=RED, add_vertex_dots = False)
        graph_R = ax.plot_line_graph(t, R/N, line_color=GRAY_D, add_vertex_dots = False)
        self.play(
            Create(graph_S),
            Create(graph_I),
            Create(graph_R),
            run_time = 10         
        )

# Model infections #
class SIRGraphTest(Scene):
    def construct(self):
        # Total Population, N:
        N = 100000
        # Initial Number of infected and recovered individuals, I0 and R0
        I0, R0 = 1, 0
        # Everyone else
        S0 = N - I0 - R0
        # Contact Rate, beta, and mean recovery rate, gamma
        beta, gamma = 0.2, 1./10
        # A grid of time points
        t = np.linspace(0, 365, 366)
        #t = np.linspace(0, 160, 161)

        # SIR Model
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
        
        # Initial conditions vector
        y0 = S0, I0, R0

        # Integrate the SIR equations over the time grid, t
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T

        # Manim Stuff
        # Add base axes
        ax = Axes(
            y_range=[0, 1],
            y_axis_config={
                "include_numbers": True,
                "numbers_to_include": np.arange(0, 1.001, 0.1),
                "decimal_number_config": {"num_decimal_places": 1}
            },
            x_range=[0, t[-1], t[-1]/8],
            x_axis_config={
                "include_numbers":True,
                "decimal_number_config": {"num_decimal_places": 0}
            },
            axis_config={
                "include_tip": False,
            }
        )
        self.add(ax)

        # Add graph
        graph_S = ax.plot_line_graph(t, S/N, line_color=BLUE, add_vertex_dots = False)
        graph_I = ax.plot_line_graph(t, I/N, line_color=RED, add_vertex_dots = False)
        graph_R = ax.plot_line_graph(t, R/N, line_color=GRAY_D, add_vertex_dots = False)
        self.play(
            Create(graph_S),
            Create(graph_I),
            Create(graph_R),
            run_time = 10         
        )
        #self.add(graph_S)
        #self.add(graph_I)
        #self.add(graph_R)


class Test(Scene):
    def construct(self):
        self.firstGraph = SIRGraph()
        self.add(self.firstGraph.axes, self.firstGraph.x_labels, self.firstGraph.x_ticks)







if __name__ == "__main__":
    os.system('cls')
    os.system('manim ".\SIR.py" SIRGraphTest -p')
    #os.system('manim ".\SIR.py" Test -sp')
    #os.system('manim ".\SIR.py" Test -q k -p')