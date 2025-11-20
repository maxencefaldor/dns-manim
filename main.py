from abc import ABC, abstractmethod
import numpy as np
import os
from manim import (
    MovingCameraScene,
    MathTex,
    NumberPlane,
    Dot,
    VGroup,
    Line,
    Create,
    Write,
    FadeIn,
    FadeOut,
    Transform,
    interpolate_color,
    ManimColor,
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

# Define colors
MAGENTA = ManimColor.from_hex("#FF00FF")


class BaseEvolutionScene(MovingCameraScene, ABC):
    """
    Base class for evolutionary algorithm animations.
    Handles the common flow: Reproduction -> Concatenation -> Evaluation -> Competition -> Selection.
    """

    def construct(self):
        # Seed randomness
        np.random.seed(2)

        # Configuration
        self.N = 16  # Population size
        self.B = 8  # Reproduction batch size
        self.num_generations = 3
        self.descriptor_range = [-3, 3, 1]
        self.k = 3

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
        self.axes.z_index = -100  # Ensure axes are always in background

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

    @abstractmethod
    def visualize_competition(self, target_dot: Dot, all_dots: list[Dot]) -> None:
        """
        Visualizes the competition calculation for the target dot.
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
        """
        Maps objective fitness to color.
        Uses Blue -> Magenta -> Red to avoid looking like Grey in the middle.
        """
        if fitness < 0.5:
            # Map 0.0-0.5 to 0.0-1.0
            return interpolate_color(BLUE, MAGENTA, fitness * 2)
        else:
            # Map 0.5-1.0 to 0.0-1.0
            return interpolate_color(MAGENTA, RED, (fitness - 0.5) * 2)

    def create_dot(self, pos: np.ndarray, color: str = GREY) -> Dot:
        """Creates a dot with attached descriptor and fitness attributes."""
        dot = Dot(self.axes.c2p(*pos[:2]), color=color, radius=0.1)
        dot.descriptor = pos
        # Initial fitness values
        dot.objective_fitness = 0.0
        dot.competition_fitness = 0.0
        return dot

    def explain_competition(self, all_dots: list[Dot]):
        """
        Zooms in on a high objective fitness individual and explains the competition metric.
        We choose the 5th highest to ensure there are enough fitter neighbors
        to visualize for Dominated Novelty Search (when k=3).
        """
        # Sort by objective fitness (descending)
        sorted_dots = sorted(all_dots, key=lambda d: d.objective_fitness, reverse=True)
        if len(sorted_dots) < 5:
            return

        subject = sorted_dots[4]  # 5th highest

        # Save camera state
        self.camera.frame.save_state()

        # Zoom in (0.6 scale for slightly wider view)
        self.play(
            self.camera.frame.animate.scale(0.6).move_to(subject.get_center()),
            run_time=1.5,
        )
        self.wait(0.5)

        # Visualize
        self.visualize_competition(subject, all_dots)

        # Restore camera
        self.play(self.camera.frame.animate.restore(), run_time=1.5)

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
                
                # Start child at parent position (appearing from behind)
                target_pos = child.get_center()
                child.move_to(parent.get_center())
                child.z_index = parent.z_index - 1

                # Animate creation
                # FadeIn (appear) + move to new position
                anims.append(FadeIn(child))
                anims.append(child.animate.move_to(target_pos))

            self.play(*anims, run_time=1.0)

            # 2. Concatenation
            self.highlight_step(1)
            all_dots = population + child_dots
            self.wait(0.5)

            # 3. Evaluation (Objective)
            self.highlight_step(2)

            # Calculate fitness first
            for dot in child_dots:
                fit = self.get_objective_fitness(dot.descriptor)
                dot.objective_fitness = fit

            if child_dots:
                # 1. Pulse Up
                self.play(
                    *[d.animate.scale(1.5) for d in child_dots],
                    run_time=0.5,
                )
                # 2. Pulse Down + Color Change
                self.play(
                    *[d.animate.scale(1/1.5).set_color(self.get_color_from_fitness(d.objective_fitness)) for d in child_dots],
                    run_time=0.5,
                )

            self.wait(0.5)

            # 4. Competition
            self.highlight_step(3)
            self.perform_competition(all_dots)

            # Explain Competition logic (Zoom and show lines)
            self.explain_competition(all_dots)

            # Pulse all nodes to show global competition
            self.play(*[d.animate.set_color(YELLOW) for d in all_dots], run_time=0.5)
            self.play(
                *[
                    d.animate.set_color(
                        self.get_color_from_fitness(d.objective_fitness)
                    )
                    for d in all_dots
                ],
                run_time=0.5,
            )

            self.wait(0.5)

            # 5. Selection
            self.highlight_step(4)
            # Sort by competition_fitness (descending)
            all_dots.sort(key=lambda d: d.competition_fitness, reverse=True)

            survivors = all_dots[: self.N]
            dead = all_dots[self.N :]

            # Animate death
            if dead:
                # 1. Turn Gray and Pulse Up
                self.play(
                    *[d.animate.set_color(GREY).scale(1.5) for d in dead],
                    run_time=0.5,
                )

                # 2. Shrink to point
                self.play(*[d.animate.scale(0.0) for d in dead], run_time=0.5)

                self.remove(*dead)

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
            MathTex(r"\text{4. Competition}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
        # Identity: Competition fitness is the same as objective fitness
        for dot in dots:
            dot.competition_fitness = dot.objective_fitness

    def visualize_competition(self, target_dot: Dot, all_dots: list[Dot]) -> None:
        # Identity: Competition fitness is the same as objective fitness
        fit = target_dot.objective_fitness

        # Formula (Simplified)
        formula = MathTex(rf"\tilde{{f}} = {fit:.2f}", font_size=24).move_to(
            target_dot.get_center()
        )
        formula.z_index = 2

        # Animate similar to Novelty Search: Turn yellow while "calculating"
        original_color = target_dot.get_color()
        self.play(target_dot.animate.set_color(YELLOW), run_time=1.0)

        # Show formula popping out
        self.play(formula.animate.next_to(target_dot, UP, buff=0.2), run_time=0.5)

        # Wait with yellow color
        self.wait(1)

        self.play(
            FadeOut(formula),
            target_dot.animate.set_color(original_color),
        )


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
            MathTex(r"\text{4. Competition}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
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
            k_nearest = distances[: self.k]

            # Novelty score is average distance to k-nearest neighbors
            novelty_score = np.mean(k_nearest) if k_nearest else 0.0

            dot_i.competition_fitness = novelty_score

    def visualize_competition(self, target_dot: Dot, all_dots: list[Dot]) -> None:
        # Find k nearest neighbors
        distances = []
        for other in all_dots:
            if other is target_dot:
                continue
            dist = np.linalg.norm(target_dot.descriptor[:2] - other.descriptor[:2])
            distances.append((dist, other))

        distances.sort(key=lambda x: x[0])
        k_nearest = distances[: self.k]

        lines = []
        labels = []
        vals = []

        for dist, neighbor in k_nearest:
            line = Line(
                target_dot.get_center(),
                neighbor.get_center(),
                color=WHITE,
                stroke_width=1,
            )
            line.z_index = -1  # Put line in background
            lines.append(line)

            # Label at midpoint
            label = (
                MathTex(f"{dist:.2f}", font_size=16)
                .move_to(line.get_center())
                .shift(UP * 0.15)
            )
            label.z_index = 1  # Ensure label is readable
            labels.append(label)
            vals.append(dist)

        self.play(*[Create(line) for line in lines])
        self.play(*[Write(lbl) for lbl in labels])
        self.wait(0.5)

        # --- New Visualization: Move labels to merge ---

        avg = np.mean(vals) if vals else 0.0

        # Formula (Simplified)
        formula = MathTex(rf"\tilde{{f}} = {avg:.2f}", font_size=24).move_to(
            target_dot.get_center()
        )
        formula.z_index = 2

        # Animate distances moving towards target AND turn target yellow
        original_color = target_dot.get_color()
        self.play(
            *[
                label.animate.move_to(target_dot.get_center()).scale(0.1).set_opacity(0)
                for label in labels
            ],
            *[FadeOut(line) for line in lines],
            target_dot.animate.set_color(YELLOW),
            run_time=1.0,
        )

        self.play(formula.animate.next_to(target_dot, UP, buff=0.2), run_time=0.5)

        # Wait with yellow color
        self.wait(1)

        self.play(
            FadeOut(formula),
            target_dot.animate.set_color(original_color),
        )


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
            MathTex(r"\text{4. Competition}", font_size=24),
            MathTex(r"\text{5. Selection (Top-}N\text{)}", font_size=24),
        ]

    def perform_competition(self, dots: list[Dot]) -> None:
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
            k_nearest = fitter_distances[: self.k]

            # Dominated novelty score
            if k_nearest:
                dominated_novelty_score = np.mean(k_nearest)
            elif len(fitter_distances) > 0:
                dominated_novelty_score = np.mean(fitter_distances)
            else:
                dominated_novelty_score = float("inf")

            dot_i.competition_fitness = dominated_novelty_score

    def visualize_competition(self, target_dot: Dot, all_dots: list[Dot]) -> None:
        # Find k nearest FITTER neighbors
        fitter_neighbors = []
        for other in all_dots:
            if other is target_dot:
                continue
            if other.objective_fitness > target_dot.objective_fitness:
                dist = np.linalg.norm(target_dot.descriptor[:2] - other.descriptor[:2])
                fitter_neighbors.append((dist, other))

        fitter_neighbors.sort(key=lambda x: x[0])
        k_nearest = fitter_neighbors[: self.k]

        lines = []
        labels = []
        vals = []

        if not k_nearest and not fitter_neighbors:
            # Local optimum case
            text = MathTex(r"\tilde{f} = \infty", font_size=24).next_to(
                target_dot, UP, buff=0.2
            )
            self.play(Write(text))
            original_color = target_dot.get_color()
            self.play(target_dot.animate.set_color(YELLOW))
            self.wait(1)
            self.play(FadeOut(text), target_dot.animate.set_color(original_color))
            return

        # If we have fewer than k, we take what we have (based on implementation logic)
        # But usually we visualize k lines if possible.

        for dist, neighbor in k_nearest:
            line = Line(
                target_dot.get_center(),
                neighbor.get_center(),
                color=WHITE,
                stroke_width=1,
            )
            line.z_index = -1
            lines.append(line)

            label = (
                MathTex(f"{dist:.2f}", font_size=16)
                .move_to(line.get_center())
                .shift(UP * 0.15)
            )
            label.z_index = 1
            labels.append(label)
            vals.append(dist)

        self.play(*[Create(line) for line in lines])
        self.play(*[Write(lbl) for lbl in labels])
        self.wait(0.5)

        # --- New Visualization: Move labels to merge ---

        avg = np.mean(vals) if vals else 0.0

        # Formula (Simplified)
        formula = MathTex(rf"\tilde{{f}} = {avg:.2f}", font_size=24).move_to(
            target_dot.get_center()
        )
        formula.z_index = 2

        # Animate distances moving towards target AND turn target yellow
        original_color = target_dot.get_color()
        self.play(
            *[
                label.animate.move_to(target_dot.get_center()).scale(0.1).set_opacity(0)
                for label in labels
            ],
            *[FadeOut(line) for line in lines],
            target_dot.animate.set_color(YELLOW),
            run_time=1.0,
        )

        self.play(formula.animate.next_to(target_dot, UP, buff=0.2), run_time=0.5)

        # Wait with yellow color
        self.wait(1)

        self.play(
            FadeOut(formula),
            target_dot.animate.set_color(original_color),
        )
