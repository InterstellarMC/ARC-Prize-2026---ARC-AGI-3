# ARC Prize 2026 - ARC-AGI-3 Agent

This repository contains my submission for the [ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3). 

## 🤖 Agent Strategy: Multi-Strategy Graph Explorer
Unlike standard LLM or CNN-based approaches, this agent utilizes a **State Graph** approach to solve AGI tasks:

* **Graph Mapping**: Tracks unique grid states (hashed) as nodes and actions as directed edges.
* **BFS Exploration**: Uses Breadth-First Search to find the shortest path to "untried" state frontiers.
* **Click Solver**: Analyzes color clusters and background colors to prioritize coordinate-based actions (Actions 6 & 7).
* **Probing Phase**: Initially tests all available simple actions to build a local transition model before deep exploration.

## 📁 Repository Structure
* `agents/my_agent.py`: The core implementation of the `MyAgent` class, `StateGraph`, and `ClickSolver`.
* `notebooks/`: Contains the Kaggle notebook used for submission.

## 🛠️ Installation & Usage
To use this agent locally within the ARC-AGI-3 environment:
1. Clone the official ARC-AGI-3-Agents repository.
2. Place `my_agent.py` in the `agents/templates/` directory.
3. Update `agents/__init__.py` to include `MyAgent`.

## ⚙️ Requirements
* numpy
* arc-agi
* arcengine
