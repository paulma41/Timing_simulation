import numpy as np
import matplotlib.pyplot as plt

# ----- Paramètres généraux -----

V = np.array([4.0, 2.0, 1.0])          # V_1, V_2, V_3
dt_actions = np.array([12.0, 6.0, 3.0])  # durées dt_1, dt_2, dt_3

M = 10.0        # biais d'action fixé
T = 30.0        # longueur de l'intervalle
rng = np.random.default_rng(0)  # RNG reproductible


# ----- Softmax avec p0 = 1/4 à b = 0 -----

def logits_and_probs(a, b, V=V, M=M):
    """
    Modèle corrigé :
    - base : s_i^base = a V_i
    - s0(a) choisi pour avoir p0 = 1/4 quand b = 0
    - pour un b quelconque : s_i = a V_i + b M, s0 = s0(a)

    Retourne logits (s0, s1, s2, s3) et probs p_j.
    """
    # Logits de base des actions (sans biais M)
    logits_base_actions = a * V  # shape (3,)

    # Somme S(a) = sum_i exp(a V_i)
    S = np.exp(logits_base_actions).sum()

    # Calibrage : pour b=0, on impose p0 = 1/4
    # => e^{s0(a)} = S/3
    s0 = np.log(S / 3.0)

    # Ajout du biais M sur les actions seulement
    logits_actions = logits_base_actions + b * M  # shape (3,)

    # Logits complets [s0, s1, s2, s3]
    logits = np.concatenate(([s0], logits_actions))

    # Softmax stable numériquement
    max_logit = logits.max()
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()

    return logits, probs


# ----- Simulation d'un intervalle de temps -----

def simulate_one_interval(probs, dt_actions=dt_actions, T=T, rng=rng):
    """
    Simule le comportement de l'agent sur un intervalle [0, T].

    Règles :
        - action 0 (inaction) : t -> t + 1
        - action i (1,2,3)   : t -> t + dt_i

    Retourne le nombre d'occurrences de chaque action.
    """
    t = 0.0
    counts = np.zeros(4, dtype=int)

    while t < T:
        # Tirage d'une action selon les probabilités
        action = rng.choice(4, p=probs)
        counts[action] += 1

        # Mise à jour du temps
        if action == 0:
            t += 1.0
        else:
            t += dt_actions[action - 1]

    return counts


# ----- Estimation des fréquences moyennes sur une grille de (a, b) -----

def estimate_frequencies(a_values, b_values, n_episodes=1000):
    """
    Pour chaque (a, b) :
      - simule n_episodes intervalles de longueur T
      - renvoie la fréquence moyenne des actions.

    freq[a_idx, b_idx, k] ≈ nombre moyen d'occurrences
    de l'action k sur un intervalle de longueur T.
    """
    freq = np.zeros((len(a_values), len(b_values), 4), dtype=float)

    for ia, a in enumerate(a_values):
        for ib, b in enumerate(b_values):
            _, probs = logits_and_probs(a, b)
            total_counts = np.zeros(4, dtype=float)

            for _ in range(n_episodes):
                total_counts += simulate_one_interval(probs)

            freq[ia, ib, :] = total_counts / n_episodes

    return freq


# ----- Exemple d'utilisation -----

if __name__ == "__main__":
    # Grille de (a, b) à explorer
    a_values = np.linspace(0.0, 1.0, 21)   # 21 valeurs de a
    b_values = np.linspace(0.0, 1.0, 21)    # 21 valeurs de b

    # Estimation des fréquences moyennes
    freq = estimate_frequencies(a_values, b_values, n_episodes=500)

    # Fréquence par unité de temps
    freq_per_time = freq / T
    norm = plt.Normalize(vmin=freq_per_time.min(), vmax=freq_per_time.max())

    # Grilles pour les plots
    A, B = np.meshgrid(a_values, b_values, indexing='ij')

    titles = [
        "Inaction (p0)",
        "Action 1 (V=4, dt=12)",
        "Action 2 (V=2, dt=6)",
        "Action 3 (V=1, dt=3)"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    for k, ax in enumerate(axes):
        im = ax.pcolormesh(A, B, freq_per_time[:, :, k],
                           shading='auto', norm=norm)
        ax.set_xlabel("a")
        if k == 0:
            ax.set_ylabel("b")
        ax.set_title(titles[k])

    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.025, pad=0.03)
    cbar.set_label("fréquence / unité de temps")

    plt.show()
