import numpy as np
from scipy.stats import poisson, skellam
from scipy.integrate import quad
from scipy.special import iv
from arrival import simulate_poisson_process_v2

# Match model
tmax = 90
rate_a = 0.02
rate_b = 0.03
nt1 = lambda t: poisson(rate_a * t)
nt2 = lambda t: poisson(rate_b * t)
nt = lambda t: poisson((rate_a + rate_b) * t)

# Simulate both processes independently
M = int(1e4)
arrivals_team_a = simulate_poisson_process_v2(rate_a, 0, tmax, M)
arrivals_team_b = simulate_poisson_process_v2(rate_b, 0, tmax, M)

# Extract information about goals
goals_a = np.array([len(arrivals_team_a[m]) for m in range(M)])
goals_b = np.array([len(arrivals_team_b[m]) for m in range(M)])
total_goals = goals_a + goals_b

# 1. Probability that no goals are scored
p1 = nt(tmax).pmf(0)
p1_sim = np.mean(total_goals == 0)
print(f"[Theoretical] Probability of no goals: {p1:.4f}")
print(f"[Simulated] Probability of no goals: {p1_sim:.4f}")
print("-----")

# 2. Probability of at least two goals
p2 = nt(tmax).sf(1)
p2_sim = np.mean(total_goals >= 2)
print(f"[Theoretical] Probability of at least two goals: {p2:.4f}")
print(f"[Simulated] Probability of at least two goals: {p2_sim:.4f}")
print("-----")

# 3. Probability of A=1, B=2
p3 = nt1(tmax).pmf(1) * nt2(tmax).pmf(2)
p3_sim = np.mean((goals_a == 1) & (goals_b == 2))
print(f"[Theoretical] Probability of A=1, B=2: {p3:.4f}")
print(f"[Simulated] Probability of A=1, B=2: {p3_sim:.4f}")
print("-----")

# 4. Probability of draw
p4 = skellam.pmf(0, rate_a * tmax, rate_b * tmax)
p4_bis = np.exp(-tmax * (rate_a + rate_b)) * iv(0, 2 * tmax * np.sqrt(rate_a * rate_b))
p4_sim = np.mean(goals_a == goals_b)
if np.isclose(p4, p4_bis):
    print(f"[Theoretical] Probability of draw: {p4:.4f}")
    print(f"[Simulated] Probability of draw: {p4_sim:.4f}")
    print("-----")

# 5. Probability of B scoring the first goal
p5 = quad(lambda t: 0.03 * np.exp(-0.05 * t), 0, 90)[0]
p5_sim = np.mean([1 if arrivals_team_b[m].size
                  and (not arrivals_team_a[m].size
                      or arrivals_team_b[m][0] < arrivals_team_a[m][0])
                  else 0 for m in range(M)])
print(f"[Theoretical] Probability of B scoring the first goal: {p5:.4f}")
print(f"[Simulated] Probability of B scoring the first goal: {p5_sim:.4f}")
