# Chernoff bound (binary decision)

On compare deux modeles avec densites $p_0(y)$ et $p_1(y)$, et des prior egaux $1/2$.
L'erreur minimale (decision bayesienne optimale) s'ecrit :

$$
P_{err} = \tfrac{1}{2} \int \min(p_0(y), p_1(y))\,dy
$$

Pour tout $s \in [0,1]$, on a l'inegalite :

$$
\min(a,b) \le a^{1-s} b^s
$$

En l'appliquant sous l'integrale :

$$
P_{err} \le \tfrac{1}{2} \int p_0(y)^{1-s} p_1(y)^s\,dy
$$

En minimisant sur $s$ (equivalent a maximiser l'opposant du log), on obtient la borne de Chernoff :

$$
P_{err} \le \tfrac{1}{2} \exp(-C)
$$

ou

$$
C = \max_{s \in [0,1]} \; -\log \int p_0(y)^{1-s} p_1(y)^s\,dy
$$

Ce $C$ est le **Chernoff information**. Plus $C$ est grand, plus les distributions sont separables.

## Cas gaussien (formule utilisee dans le code)

Si $p_0, p_1$ sont gaussiennes $\mathcal{N}(\mu_0, \Sigma_0)$ et $\mathcal{N}(\mu_1, \Sigma_1)$ :

$$
C_s = \tfrac{1}{2}\Big( s(1-s)\,\Delta\mu^T \Sigma_s^{-1} \Delta\mu
+ \log|\Sigma_s| - (1-s)\log|\Sigma_0| - s\log|\Sigma_1| \Big)
$$

avec $\Delta\mu = \mu_1 - \mu_0$ et $\Sigma_s = (1-s)\Sigma_0 + s\Sigma_1$. On prend :

$$
C = \max_{s \in [0,1]} C_s
$$

### Derivation rapide (cas gaussien)

On part de :

$$
I(s) = \int p_0(y)^{1-s} p_1(y)^s \, dy
$$

avec
$$
p_k(y)=\frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
\exp\left(-\tfrac12 (y-\mu_k)^T \Sigma_k^{-1} (y-\mu_k)\right).
$$

Alors
$$
p_0(y)^{1-s} p_1(y)^s
= (2\pi)^{-d/2}\,|\Sigma_0|^{-(1-s)/2}|\Sigma_1|^{-s/2}
\exp\left(-\tfrac12 Q_s(y)\right),
$$
ou
$$
Q_s(y)=(1-s)(y-\mu_0)^T\Sigma_0^{-1}(y-\mu_0)+s(y-\mu_1)^T\Sigma_1^{-1}(y-\mu_1).
$$

En completant le carre, on obtient une gaussienne equivalente de covariance
$$
\Sigma_s = \big((1-s)\Sigma_0^{-1} + s\Sigma_1^{-1}\big)^{-1},
$$
et un terme quadratique en $\Delta\mu=\mu_1-\mu_0$. L'integrale se calcule alors :

$$
I(s) = (2\pi)^{-d/2}\,|\Sigma_0|^{-(1-s)/2}|\Sigma_1|^{-s/2}\,
|\Sigma_s|^{1/2}\,
\exp\left(-\tfrac12 s(1-s)\Delta\mu^T \Sigma_s^{-1} \Delta\mu\right).
$$

En prenant $C_s=-\log I(s)$ et en rearrangeant, on retrouve :

$$
C_s = \tfrac12\Big( s(1-s)\Delta\mu^T\Sigma_s^{-1}\Delta\mu
 + \log|\Sigma_s| - (1-s)\log|\Sigma_0| - s\log|\Sigma_1| \Big).
$$


## Pourquoi comparer deux a deux (pairwise)

La quantite cible est multi-modeles :

$$
p(\hat e=1\mid u)=1-\int_{\mathcal Y} \max_m \big[p(m)\,p(y\mid m,u)\big] \,dy.
$$

Le terme $\max_m$ rend le calcul exact difficile. Une strategie classique consiste a **borner** l'erreur multi-classe par une somme (ou un max) d'erreurs **binaires** : si chaque paire de modeles est bien separable, alors l'erreur globale est controlee. Techniquement, on applique une borne de type *union bound* sur les evenements de confusion entre paires. Cela transforme un probleme multi-modeles en une collection de comparaisons deux a deux, ce qui permet d'utiliser la borne de Chernoff pour chaque paire.

Plus explicitement, si $m^*$ est le vrai modele, l'erreur multi-classe est l'evenement
$$
E = \{\hat m(y) \ne m^*\} = \bigcup_{j\ne m^*} E_j,
$$
ou $E_j$ = "on choisit $j$ alors que $m^*$ est vrai". Par l'inegalite de Boole (union bound),
$$
P(E) \le \sum_{j\ne m^*} P(E_j).
$$
Donc si chaque probabilite d'erreur binaire $P(E_j)$ est petite, leur somme (et donc l'erreur totale) l'est aussi. C'est ce qui justifie le recours aux bornes pairwise.
