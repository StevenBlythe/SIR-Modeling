from manim import *
import math
class Coloring(Scene):
    def construct(self):
        k = 0.2
        c = 1
        f = lambda t : c * math.e**(k*t)
        colors = [RED, YELLOW, BLUE, DARK_GRAY]
        ax = Axes(
            tips=False
        )
        self.add(ax)
        graph = ax.plot(f)
        self.play(Write(graph))
        vf = ArrowVectorField(
            f, min_color_scheme_value=2, max_color_scheme_value=10, colors=colors
        )
        self.play(Write(vf,rate_func=rate_functions.ease_in_out_sine), run_time=4)