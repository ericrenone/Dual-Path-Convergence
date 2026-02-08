# Dual-Path Lambda Learning  
### Functional Adaptive Convergence via Drift Minimization and Gain Control

This repository implements a **minimal functional learning system** that compares two fundamentally different optimization dynamics:

• Standard gradient drift minimization  
• Adaptive gain-controlled learning with internal sensitivity decay  

The system is built entirely from **immutable lambda-style state transitions**, turning learning into a clean dynamical process rather than imperative parameter mutation.

Convergence is evaluated using **information-theoretic drift (KL divergence)** alongside prediction error.


---

## 🔍 Core Concept

Learning is modeled as a deterministic state machine:

Each update step:

1. Observes noisy data sampled from a true underlying distribution  
2. Computes an error signal relative to the current internal belief  
3. Applies pure functional transitions to produce the next state  

Two learning pathways operate in parallel:

### Path A — Drift Minimization

A classic gradient descent dynamic that directly adjusts the internal representation toward observed data.

### Path B — Adaptive Gain Control

A coupled system where:

• The representation is updated  
• A sensitivity (gain) parameter scales learning intensity  
• The gain decays over time, stabilizing convergence  

## This creates a self-regulating learning process.

## Information requires a carrier. Path A is the carrier (the signal), and Path B is the information (the meaning/stability extracted from that signal).

---

## 🧠 Functional Architecture

All learning behavior is expressed as stateless transformations:

• No in-place parameter mutation  
• No optimizer objects  
• No hidden side effects  

State evolution is entirely driven by composable transition functions.

This mirrors functional dynamical systems used in theoretical ML and neuromorphic models.

---

## 📊 What Is Measured

At every step the system tracks:

• Information drift between learned and true distributions  
• Prediction error on sampled data  
• Internal gain dynamics  

This directly measures **how accurately the model infers reality**, not just loss reduction.

---

## 🔬 Key Experimental Observations

Across runs, adaptive gain control consistently:

✅ Converges faster  
✅ Produces smoother learning curves  
✅ Resists noise-induced oscillations  
✅ Stabilizes long-term inference  

While drift-only learning overshoots and fluctuates under noise.

---

## 🧪 Main Takeaways

### 1. Learning benefits from internal control dynamics

Coupling representation updates with a self-regulating sensitivity parameter dramatically improves stability.

---

### 2. Adaptive behavior emerges from simple rules

Second-order style optimization effects arise without complex math or heavy algorithms.

---

### 3. Information convergence matters more than loss alone

The system does not just minimize error — it actively aligns inferred structure with ground truth.

---

### 4. Functional state transitions are sufficient for learning systems

Complex learning behavior can emerge from pure transformations.

---

## 🎯 Scientific Relevance

This project connects ideas from:

• Variational inference  
• Predictive coding  
• Adaptive optimization  
• Control-theoretic learning  
• Functional programming models of computation  

In a minimal, interpretable simulation.

---

## 📈 Visualization

A real-time GUI displays:

• Information drift comparison  
• Prediction error trajectories  
• Gain parameter decay  

Allowing intuitive inspection of learning dynamics.

## Conclusion

The simulation demonstrates that adaptive gain learning preserves convergence speed while significantly reducing stochastic variance, producing smooth, repeatable, and noise-robust trajectories.


