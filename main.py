from abc import ABC, abstractmethod
import numpy as np
import os
from manim import (
    Scene,
    MathTex,
    NumberPlane,
    Dot,
    VGroup,
    DashedLine,
    Create,
    Write,
    FadeIn,
    FadeOut,
    Transform,
    interpolate_color,
    BLUE,
    RED,
    TEAL,
    GREY,
    WHITE,
    YELLOW,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    UL,
)

# Add LaTeX path
os.environ["PATH"] += ":/usr/local/texlive/2025/bin/universal-darwin"


class BaseEvolutionScene(Scene, ABC):
    """
    Base class for evolutionary algorithm animations.
    Handles the common flow: Reproduction -> Concatenation -> Evaluation -> Competition -> Selection.
    """

    def construct(self):
        # Seed randomness
        np.random.seed(0)

        # Configuration
        self.N = 16  # Population size
        self.B = 8  # Reproduction batch size
        self.num_generations = 3
        self.descriptor_range = [-3, 3, 1]

        # Layout
        self.setup_layout()

        # Run Evolution
        self.run_evolution()

    def setup_layout(self):
        """Sets up the static visual elements of the scene."""
        # Title
        title = MathTex(self.get_title_text(), font_size=36).to_edge(UP)
        self.add(title)

        # Descriptor Space (NumberPlane with grid)
        self.axes = NumberPlane(
            x_range=self.descriptor_range,
            y_range=self.descriptor_range,
            x_length=6,
            y_length=6,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.6,
            },
            faded_line_style={
                "stroke_color": TEAL,
                "stroke_opacity": 0.2,
                "stroke_width": 1,
            },
            faded_line_ratio=5,
        ).shift(LEFT * 2)

        # Labels
        descriptor_label = MathTex(
            r"\text{Descriptor Space } \mathcal{D}", font_size=30, color=TEAL
        ).to_corner(UL)

        self.play(Create(self.axes), Write(descriptor_label))

        # Algorithm Steps Display
        self.steps_text = (
            VGroup(*self.get_steps_content())
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            .to_edge(RIGHT)
            .shift(LEFT)
        )

        self.play(Write(self.steps_text))

        # Generation Counter
        self.gen_counter = MathTex(r"\text{Generation } 1", font_size=30).next_to(
            self.axes, DOWN
        )
        self.play(Write(self.gen_counter))

    @abstractmethod
    def get_title_text(self) -> str:
        """Returns the title of the algorithm."""
        pass

    @abstractmethod
    def get_steps_content(self) -> list[MathTex]:
        """Returns the list of MathTex objects for the steps."""
        pass

    @abstractmethod
    def perform_competition(self, dots: list[Dot]) -> None:
        """
        Calculates and sets the competition fitness for each dot.
        Must set `dot.competition_fitness`.
        """
        pass

    def highlight_step(self, index: int):
        """Highlights the current step in the steps list."""
        self.play(
            self.steps_text.animate.set_color(WHITE),
            self.steps_text[index].animate.set_color(YELLOW),
            run_time=0.5,
        )

    def get_objective_fitness(self, point: np.ndarray) -> float:
        """Calculates the objective fitness (Gaussian at center)."""
        x, y = point[0], point[1]
        dist = np.sqrt(x**2 + y**2)
        return np.exp(-(dist**2) / 2)

    def get_color_from_fitness(self, fitness: float) -> str:
        """Maps objective fitness to color (Blue to Red)."""
        return interpolate_color(BLUE, RED, fitness)

    def create_dot(self, pos: np.ndarray, color: str = GREY) -> Dot:
        """Creates a dot with attached descriptor and fitness attributes."""
        dot = Dot(self.axes.c2p(*pos[:2]), color=color, radius=0.1)
        dot.descriptor = pos
        # Initial fitness values
        dot.objective_fitness = 0.0
        dot.competition_fitness = 0.0
        return dot

    def run_evolution(self):
        # Initialize Population
        population = []
        dots_group = VGroup()

        for _ in range(self.N):
            pos = np.array(
                [
                    np.random.uniform(
                        self.descriptor_range[0], self.descriptor_range[1]
                    ),
                    np.random.uniform(
                        self.descriptor_range[0], self.descriptor_range[1]
                    ),
                    0,
                ]
            )
            dot = self.create_dot(pos)

            # Initial Evaluation
            fit = self.get_objective_fitness(pos)
            dot.objective_fitness = fit
            dot.competition_fitness = fit  # Default for initial display
            dot.set_color(self.get_color_from_fitness(fit))

            population.append(dot)
            dots_group.add(dot)

        self.play(FadeIn(dots_group))
        self.wait()

        # Generation Loop
        for generation in range(self.num_generations):
            # Update Generation Counter
            if generation > 0:
                new_gen_text = MathTex(
                    r"\text{Generation } " + str(generation + 1), font_size=30
                ).next_to(self.axes, DOWN)
                self.play(Transform(self.gen_counter, new_gen_text))

            # 1. Reproduction
            self.highlight_step(0)

            # Select parents uniformly (random choice from current population)
            parent_indices = np.random.choice(
                len(population), size=self.B, replace=True
            )
            parents = [population[i] for i in parent_indices]
            child_dots = []

            anims = []
            for parent in parents:
                # Mutation: small random offset
                offset = np.array(
                    [np.random.normal(0, 0.5), np.random.normal(0, 0.5), 0]
                )
                new_pos = parent.descriptor + offset
                # Clamp to axes range roughly
                new_pos = np.clip(new_pos, -3, 3)
                new_pos[2] = 0  # Ensure z is 0

                child = self.create_dot(new_pos, color=GREY)
                child_dots.append(child)

                # Animate creation
                line = DashedLine(
                    parent.get_center(),
                    child.get_center(),
                    color=WHITE,
                    stroke_opacity=0.5,
                    stroke_width=1,  # Reduced thickness
                    dashed_ratio=0.75,  # Longer dashes
                    dash_length=0.1,  # Increased dash length
                )
                anims.append(Create(line))
                anims.append(FadeIn(child))

            self.play(*anims, run_time=0.5)

            # 2. Concatenation
            self.highlight_step(1)
            all_dots = population + child_dots
            self.wait(0.5)

            # Remove lines
            lines = [m for m in self.mobjects if isinstance(m, DashedLine)]
            if lines:
                self.play(*[FadeOut(line) for line in lines], run_time=0.5)

            # 3. Evaluation (Objective)
            self.highlight_step(2)
            eval_anims = []
            for dot in child_dots:
                fit = self.get_objective_fitness(dot.descriptor)
                dot.objective_fitness = fit
                # Color always reflects objective fitness
                eval_anims.append(
                    dot.animate.set_color(self.get_color_from_fitness(fit))
                )

            if eval_anims:
                self.play(*eval_anims)
            self.wait(0.5)

            # 4. Competition
            self.highlight_step(3)
            self.perform_competition(all_dots)
            self.wait(0.5)

            # 5. Selection
            self.highlight_step(4)
            # Sort by competition_fitness (descending)
            all_dots.sort(key=lambda d: d.competition_fitness, reverse=True)

            survivors = all_dots[: self.N]
            dead = all_dots[self.N :]

            # Animate death
            death_anims = []
            for d in dead:
                death_anims.append(d.animate.set_color(GREY).scale(0.5))

            if death_anims:
                self.play(*death_anims, run_time=1.0)
                self.play(*[FadeOut(d) for d in dead], run_time=1.0)

            population = survivors
            # Note: dots_group is not strictly tracked as a VGroup for updating,
            # but manim handles mobjects added to scene.

            self.wait()

        # End
        self.play(self.steps_text.animate.set_color(WHITE))
        self.wait(2)


class GeneticAlgorithm(BaseEvolutionScene):
    """
    Standard Genetic Algorithm implementation.
    Competition is based on identity (objective fitness).
    """

    def get_title_text(self) -> str:
        return r"\textbf{Genetic Algorithm}"

    def get_steps_content(self) -> list[MathTex]:
        return [
            MathTex(r"\text{1. Reproduction}", font_size=24),
            MathTex(r"\text{2. Concatenation}", font_size=24),
            MathTex(r"\text{3. Evaluation}", font_size=24),
            MathTex(r"\text{4. Competition (Identity)}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
        # Identity: Competition fitness is the same as objective fitness
        for dot in dots:
            dot.competition_fitness = dot.objective_fitness


class NoveltySearch(BaseEvolutionScene):
    """
    Novelty Search implementation.
    Competition is based on novelty score (average distance to k-nearest neighbors).
    """

    def get_title_text(self) -> str:
        return r"\textbf{Novelty Search}"

    def get_steps_content(self) -> list[MathTex]:
        return [
            MathTex(r"\text{1. Reproduction}", font_size=24),
            MathTex(r"\text{2. Concatenation}", font_size=24),
            MathTex(r"\text{3. Evaluation}", font_size=24),
            MathTex(r"\text{4. Competition (Novelty)}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
        k = 3  # Number of nearest neighbors

        # Calculate distances and novelty scores
        for i, dot_i in enumerate(dots):
            distances = []
            for j, dot_j in enumerate(dots):
                if i != j:
                    # Euclidean distance in descriptor space (first 2 coords)
                    dist = np.linalg.norm(dot_i.descriptor[:2] - dot_j.descriptor[:2])
                    distances.append(dist)

            # Sort distances and take top k
            distances.sort()
            k_nearest = distances[:k]

            # Novelty score is average distance to k-nearest neighbors
            novelty_score = np.mean(k_nearest) if k_nearest else 0.0

            dot_i.competition_fitness = novelty_score


class DominatedNoveltySearch(BaseEvolutionScene):
    """
    Dominated Novelty Search implementation.
    Competition is based on dominated novelty score (average distance to k-nearest FITTER neighbors).
    """

    def get_title_text(self) -> str:
        return r"\textbf{Dominated Novelty Search}"

    def get_steps_content(self) -> list[MathTex]:
        return [
            MathTex(r"\text{1. Reproduction}", font_size=24),
            MathTex(r"\text{2. Concatenation}", font_size=24),
            MathTex(r"\text{3. Evaluation}", font_size=24),
            MathTex(r"\text{4. Competition (Dom. Novelty)}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
        k = 3  # Number of nearest neighbors

        # Calculate distances to FITTER neighbors
        for i, dot_i in enumerate(dots):
            fitter_distances = []
            for j, dot_j in enumerate(dots):
                if i != j and dot_j.objective_fitness > dot_i.objective_fitness:
                    # Euclidean distance in descriptor space
                    dist = np.linalg.norm(dot_i.descriptor[:2] - dot_j.descriptor[:2])
                    fitter_distances.append(dist)

            # Sort distances and take top k
            fitter_distances.sort()
            k_nearest = fitter_distances[:k]

            # Dominated novelty score is average distance to k-nearest FITTER neighbors
            # If no fitter neighbors exist (local optimum), assign infinite score to ensure survival
            if k_nearest:
                dominated_novelty_score = np.mean(k_nearest)
            elif len(fitter_distances) > 0:
                # Fewer than k fitter neighbors, take average of what we have
                dominated_novelty_score = np.mean(fitter_distances)
            else:
                # No fitter neighbors -> Max score
                dominated_novelty_score = float("inf")

            dot_i.competition_fitness = dominated_novelty_score
