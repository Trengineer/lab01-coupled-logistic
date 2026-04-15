# Lab 01 — Two Competing Species: Coupled Logistic Maps

Modelling Complex Systems, Spring 2026

## Model

Two populations with densities $x_n$ and $y_n$ evolve according to:

$$x_{n+1} = (1 - \varepsilon)\, r_1\, x_n(1 - x_n) + \varepsilon\, r_2\, y_n(1 - y_n)$$

$$y_{n+1} = (1 - \varepsilon)\, r_2\, y_n(1 - y_n) + \varepsilon\, r_1\, x_n(1 - x_n)$$

## Repository Structure

```
├── src/
│   ├── coupled_map.py          # Shared map definition and utilities
│   ├── periodicPoints.py       # Exercise 2: periodic orbits (period 2 & 3)
│   ├── phasePortrait.py        # Exercise 3: phase plane plots
│   └── bifurcationCascade.py   # Exercise 4: bifurcation diagram
├── figures/
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-user>/lab01-coupled-logistic.git
cd lab01-coupled-logistic
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
cd src
python periodicPoints.py
python phasePortrait.py
python bifurcationCascade.py
```

Figures are saved to `../figures/`.

## Task Division

| Exercise | File | Assignee |
|---|---|---|
| 1. Fixed points (analytical) | report | — |
| 2. Periodic orbits | `periodicPoints.py` | — |
| 3. Phase portraits | `phasePortrait.py` | — |
| 4. Bifurcation cascade | `bifurcationCascade.py` | — |

## Git Workflow

1. Pull before starting work: `git pull`
2. Work on your own file / branch
3. Commit with descriptive messages
4. Push and open a PR if using branches
