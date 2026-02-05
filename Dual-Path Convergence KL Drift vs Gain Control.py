#!/usr/bin/env python3
"""
Dual-Path Convergence: KL Drift vs Gain Control

Demonstrates two optimization regimes for minimizing variational free energy:
1. Representation Purification - geometric drift correction
2. Sensitivity Annealing - adaptive gain control reduction
"""

import random
import math
import tkinter as tk
from typing import List, Tuple, Dict, Any

# ========== Configuration ==========
STEPS: int = 100
LEARNING_RATE: float = 0.08
TRUE_MEAN: float = 2.0
TRUE_STD: float = 1.0
SAMPLE_SIZE: int = 80

# Display settings
WINDOW_WIDTH: int = 850
WINDOW_HEIGHT: int = 700
PLOT_HEIGHT: int = 160
UPDATE_DELAY: int = 50  # milliseconds between steps

# ========== Math Utilities ==========
def mean(values: List[float]) -> float:
    """Calculate arithmetic mean of a list of values."""
    return sum(values) / len(values) if values else 0.0

def variance(values: List[float], mean_val: float) -> float:
    """Calculate sample variance given values and their mean."""
    if len(values) < 2:
        return 1.0
    return sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)

def kl_divergence(m1: float, s1: float, m2: float, s2: float) -> float:
    """
    KL divergence between two Gaussian distributions N(m1,s1²) || N(m2,s2²).
    
    Args:
        m1: Mean of first distribution
        s1: Standard deviation of first distribution
        m2: Mean of second distribution
        s2: Standard deviation of second distribution
    
    Returns:
        KL divergence value (always >= 0)
    """
    if s1 <= 0 or s2 <= 0:
        return 0.0
    return math.log(s2/s1) + (s1*s1 + (m1-m2)**2)/(2*s2*s2) - 0.5

# ========== Simulation State ==========
class SimulationState:
    """Manages the dual-path optimization simulation state and updates."""
    
    def __init__(self) -> None:
        """Initialize simulation with both optimization paths at origin."""
        # Model parameters
        self.theta_purify: float = 0.0
        self.theta_anneal: float = 0.0
        self.sigma_purify: float = 1.0
        self.sigma_anneal: float = 1.0
        self.alpha: float = 1.0
        
        # History tracking
        self.kl_purify: List[float] = []
        self.kl_anneal: List[float] = []
        self.loss_purify: List[float] = []
        self.loss_anneal: List[float] = []
        self.alpha_history: List[float] = []
        
    def step(self, step_num: int) -> None:
        """
        Execute one simulation step with both optimization regimes.
        
        Args:
            step_num: Current step number (1-indexed)
        """
        # Generate data samples from true distribution
        data: List[float] = [
            random.gauss(TRUE_MEAN, TRUE_STD) 
            for _ in range(SAMPLE_SIZE)
        ]
        sample_mean: float = mean(data)
        sample_var: float = variance(data, sample_mean)
        sample_std: float = math.sqrt(sample_var)
        
        # Compute variational free-energy gradients
        # F = E_q[log q - log p] where q is our model, p is true distribution
        grad_purify_mean: float = self.theta_purify - sample_mean
        grad_anneal_mean: float = self.theta_anneal - sample_mean
        
        # Regime 1: Representation purification (geometric correction)
        # Direct gradient descent on KL divergence
        self.theta_purify -= LEARNING_RATE * grad_purify_mean
        
        # Regime 2: Sensitivity annealing (gain control)
        # Modulated gradient descent with exponentially decaying gain
        self.theta_anneal -= LEARNING_RATE * self.alpha * grad_anneal_mean
        self.alpha = max(0.001, self.alpha * 0.985)  # Exponential decay
        
        # Track information-theoretic metrics (no arbitrary clipping)
        kl_p: float = kl_divergence(
            self.theta_purify, self.sigma_purify, 
            TRUE_MEAN, TRUE_STD
        )
        kl_a: float = kl_divergence(
            self.theta_anneal, self.sigma_anneal,
            TRUE_MEAN, TRUE_STD
        )
        
        # Track prediction loss (mean squared error)
        loss_p: float = mean([(x - self.theta_purify)**2 for x in data])
        loss_a: float = mean([(x - self.theta_anneal)**2 for x in data])
        
        # Update histories
        self.kl_purify.append(kl_p)
        self.kl_anneal.append(kl_a)
        self.loss_purify.append(loss_p)
        self.loss_anneal.append(loss_a)
        self.alpha_history.append(self.alpha)

# ========== Visualization Components ==========
class PlotCanvas:
    """Encapsulates a single plot canvas with rendering logic."""
    
    def __init__(
        self, 
        parent: tk.Frame, 
        title: str, 
        labels: List[str], 
        colors: List[str]
    ) -> None:
        """
        Create a plot canvas.
        
        Args:
            parent: Parent Tkinter frame
            title: Plot title
            labels: Series labels for legend
            colors: Hex color codes for each series
        """
        self.title: str = title
        self.labels: List[str] = labels
        self.colors: List[str] = colors
        
        # Create frame
        self.frame: tk.Frame = tk.Frame(parent, bg="#ffffff")
        self.frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Title
        title_label: tk.Label = tk.Label(
            self.frame,
            text=title,
            font=("Arial", 11, "bold"),
            bg="#ffffff",
            fg="#334155"
        )
        title_label.pack(anchor="w")
        
        # Canvas
        self.canvas: tk.Canvas = tk.Canvas(
            self.frame,
            width=WINDOW_WIDTH-40,
            height=PLOT_HEIGHT,
            bg="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1"
        )
        self.canvas.pack(pady=(5, 0))
    
    def draw(self, data_series: List[List[float]]) -> None:
        """
        Render multiple data series on the canvas.
        
        Args:
            data_series: List of data series to plot (same length)
        """
        self.canvas.delete("all")
        
        if not data_series or not data_series[0]:
            return
        
        # Canvas dimensions
        width: int = WINDOW_WIDTH - 40
        height: int = PLOT_HEIGHT
        padding: int = 30
        plot_width: int = width - 2 * padding
        plot_height: int = height - 2 * padding
        
        # Compute value range across all series
        all_values: List[float] = [v for series in data_series for v in series]
        if not all_values:
            return
        
        max_val: float = max(all_values)
        min_val: float = min(all_values)
        value_range: float = max_val - min_val
        
        # Prevent division by zero
        if value_range < 1e-9:
            value_range = 1.0
        
        # Draw coordinate system
        self._draw_axes(width, height, padding)
        self._draw_grid(width, height, padding, plot_height)
        
        # Draw data series
        self._draw_series(
            data_series, width, height, padding, 
            plot_width, plot_height, min_val, value_range
        )
        
        # Draw annotations
        self._draw_labels(
            width, height, padding, 
            max_val, min_val, len(data_series[0])
        )
        
        # Draw legend
        if len(self.labels) > 1:
            self._draw_legend(width, padding)
    
    def _draw_axes(self, width: int, height: int, padding: int) -> None:
        """Draw x and y axes."""
        # X-axis
        self.canvas.create_line(
            padding, height - padding,
            width - padding, height - padding,
            fill="#94a3b8", width=1
        )
        # Y-axis
        self.canvas.create_line(
            padding, padding,
            padding, height - padding,
            fill="#94a3b8", width=1
        )
    
    def _draw_grid(
        self, 
        width: int, 
        height: int, 
        padding: int, 
        plot_height: int
    ) -> None:
        """Draw horizontal grid lines."""
        for i in range(5):
            y: float = padding + (plot_height * i / 4)
            self.canvas.create_line(
                padding, y,
                width - padding, y,
                fill="#e2e8f0", width=1, dash=(2, 2)
            )
    
    def _draw_series(
        self,
        data_series: List[List[float]],
        width: int,
        height: int,
        padding: int,
        plot_width: int,
        plot_height: int,
        min_val: float,
        value_range: float
    ) -> None:
        """Draw smooth line plots for each data series."""
        for series, color in zip(data_series, self.colors):
            if len(series) < 2:
                continue
            
            points: List[float] = []
            for i, value in enumerate(series):
                x: float = padding + (plot_width * i / (STEPS - 1))
                y: float = height - padding - (
                    (value - min_val) / value_range * plot_height
                )
                y = max(padding, min(height - padding, y))
                points.extend([x, y])
            
            if len(points) >= 4:
                self.canvas.create_line(
                    points,
                    fill=color,
                    width=2,
                    smooth=True
                )
    
    def _draw_labels(
        self,
        width: int,
        height: int,
        padding: int,
        max_val: float,
        min_val: float,
        step_count: int
    ) -> None:
        """Draw axis labels and current values."""
        # X-axis label (step counter)
        self.canvas.create_text(
            width - padding - 5, height - padding + 15,
            text=f"{step_count}/{STEPS}",
            font=("Arial", 8),
            fill="#64748b"
        )
        
        # Y-axis max
        self.canvas.create_text(
            padding - 5, padding,
            text=f"{max_val:.2f}",
            font=("Arial", 8),
            anchor="e",
            fill="#64748b"
        )
        
        # Y-axis min
        self.canvas.create_text(
            padding - 5, height - padding,
            text=f"{min_val:.2f}",
            font=("Arial", 8),
            anchor="e",
            fill="#64748b"
        )
    
    def _draw_legend(self, width: int, padding: int) -> None:
        """Draw legend for multiple series."""
        legend_x: int = width - padding - 120
        legend_y: int = padding + 10
        
        for label, color in zip(self.labels, self.colors):
            self.canvas.create_line(
                legend_x, legend_y,
                legend_x + 25, legend_y,
                fill=color, width=2
            )
            self.canvas.create_text(
                legend_x + 30, legend_y,
                text=label,
                font=("Arial", 9),
                anchor="w",
                fill="#475569"
            )
            legend_y += 18

# ========== Main Application Window ==========
class SimulationWindow:
    """Main application window managing simulation and visualization."""
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the simulation window.
        
        Args:
            root: Tkinter root window
        """
        self.root: tk.Tk = root
        self.root.title("Dual-Path Convergence: KL Drift vs Gain Control")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#ffffff")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create UI layout
        self._create_layout()
        
        # Initialize simulation
        self.state: SimulationState = SimulationState()
        self.current_step: int = 0
        self.running: bool = True
        
        # Start simulation
        self.root.after(100, self.run_step)
    
    def _create_layout(self) -> None:
        """Create the main UI layout with header, status panel, and plots."""
        # Main container
        main_frame: tk.Frame = tk.Frame(self.root, bg="#ffffff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        # Header
        header: tk.Label = tk.Label(
            main_frame,
            text="Dual-Path Convergence: KL Drift vs Gain Control",
            font=("Arial", 16, "bold"),
            bg="#ffffff",
            fg="#1e293b"
        )
        header.pack(pady=(0, 15))
        
        # Status panel
        self._create_status_panel(main_frame)
        
        # Create plots
        self.kl_plot: PlotCanvas = PlotCanvas(
            main_frame,
            "Information Drift (KL Divergence)",
            ["Purification", "Annealing"],
            ["#dc2626", "#2563eb"]
        )
        
        self.loss_plot: PlotCanvas = PlotCanvas(
            main_frame,
            "Prediction Loss (MSE)",
            ["Purification", "Annealing"],
            ["#dc2626", "#2563eb"]
        )
        
        self.alpha_plot: PlotCanvas = PlotCanvas(
            main_frame,
            "Sensitivity Parameter α (Gain Control)",
            ["Alpha"],
            ["#16a34a"]
        )
    
    def _create_status_panel(self, parent: tk.Frame) -> None:
        """Create the status panel with real-time metrics."""
        status_frame: tk.Frame = tk.Frame(
            parent, bg="#f8fafc", relief=tk.RIDGE, bd=1
        )
        status_frame.pack(fill=tk.X, pady=(0, 15))
        
        status_inner: tk.Frame = tk.Frame(status_frame, bg="#f8fafc")
        status_inner.pack(padx=15, pady=10)
        
        # Step counter
        self.step_label: tk.Label = tk.Label(
            status_inner,
            text="Step: 0/100",
            font=("Arial", 11, "bold"),
            bg="#f8fafc",
            fg="#0f172a"
        )
        self.step_label.grid(row=0, column=0, padx=20, sticky="w")
        
        # Metric labels
        self.theta_p_label: tk.Label = tk.Label(
            status_inner,
            text="θ_purify: 0.000",
            font=("Consolas", 10),
            bg="#f8fafc",
            fg="#dc2626"
        )
        self.theta_p_label.grid(row=0, column=1, padx=20)
        
        self.theta_a_label: tk.Label = tk.Label(
            status_inner,
            text="θ_anneal: 0.000",
            font=("Consolas", 10),
            bg="#f8fafc",
            fg="#2563eb"
        )
        self.theta_a_label.grid(row=0, column=2, padx=20)
        
        self.alpha_label: tk.Label = tk.Label(
            status_inner,
            text="α: 1.000",
            font=("Consolas", 10),
            bg="#f8fafc",
            fg="#16a34a"
        )
        self.alpha_label.grid(row=0, column=3, padx=20)
    
    def update_display(self) -> None:
        """Update all visualizations and status metrics."""
        # Update status labels
        self.step_label.config(text=f"Step: {self.current_step}/{STEPS}")
        self.theta_p_label.config(
            text=f"θ_purify: {self.state.theta_purify:.3f}"
        )
        self.theta_a_label.config(
            text=f"θ_anneal: {self.state.theta_anneal:.3f}"
        )
        self.alpha_label.config(text=f"α: {self.state.alpha:.3f}")
        
        # Update plots
        self.kl_plot.draw([self.state.kl_purify, self.state.kl_anneal])
        self.loss_plot.draw([self.state.loss_purify, self.state.loss_anneal])
        self.alpha_plot.draw([self.state.alpha_history])
    
    def run_step(self) -> None:
        """Execute one simulation step and schedule the next."""
        if not self.running or self.current_step >= STEPS:
            if self.current_step >= STEPS:
                self.print_summary()
            return
        
        self.current_step += 1
        self.state.step(self.current_step)
        self.update_display()
        
        # Schedule next step
        self.root.after(UPDATE_DELAY, self.run_step)
    
    def print_summary(self) -> None:
        """Print final summary statistics to console."""
        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print("\nTwo optimization regimes for variational free energy:")
        print("  1) Purification: Geometric KL drift correction")
        print("  2) Annealing: Adaptive gain control reduction")
        print(f"\nFinal Results (Target: μ={TRUE_MEAN:.4f}, σ={TRUE_STD:.4f}):")
        print(f"  Purification θ: {self.state.theta_purify:.4f}")
        print(f"  Annealing θ:    {self.state.theta_anneal:.4f}")
        print(f"  Final KL (Purify): {self.state.kl_purify[-1]:.6f}")
        print(f"  Final KL (Anneal): {self.state.kl_anneal[-1]:.6f}")
        print(f"  Final α: {self.state.alpha_history[-1]:.6f}")
        print(f"{'='*60}\n")
    
    def on_closing(self) -> None:
        """Handle window close event."""
        self.running = False
        self.root.destroy()

# ========== Application Entry Point ==========
def main() -> None:
    """Initialize and run the simulation application."""
    root: tk.Tk = tk.Tk()
    app: SimulationWindow = SimulationWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()