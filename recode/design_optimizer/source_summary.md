Voici un résumé **de la partie Methods**, en mettant les notations au propre puis en déroulant la chaîne de raisonnement (principalement les équations 1–17). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 1) Notations (posées proprement)

- **Espace des modèles** : \(\mathcal M=\{m_1,\dots,m_{\bar M}\}\) avec un **a priori sur les modèles** \(p(m)\).
- **Design / manipulation expérimentale** : \(u\in\mathcal U\) (p.ex. séquence de stimulations).
- **Données observées** : \(y\in\mathcal Y\).
- **Paramètres latents** d’un modèle \(m\) : \(\vartheta\) (leur nature dépend du modèle, p.ex. paramètres neuraux/hémodynamiques en DCM).

### Vraisemblances, priors, evidence
- Vraisemblance : \(p(y\mid \vartheta,m,u)\).
- Prior paramètres : \(p(\vartheta\mid m,u)\) (souvent \(u\) n’entre pas dans le prior, mais ils laissent la dépendance possible).
- **Evidence / vraisemblance marginale** (aussi “model evidence”) :
\[
p(y\mid m,u)=\int p(y\mid \vartheta,m,u)\,p(\vartheta\mid m,u)\,d\vartheta.
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

- **Posterior sur les modèles** :
\[
p(m\mid y,u)=\frac{p(m)\,p(y\mid m,u)}{p(y\mid u)},\qquad
p(y\mid u)=\sum_{m\in\mathcal M} p(m)\,p(y\mid m,u).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 2) Décision Bayésienne et erreur de sélection

Ils formalisent la **sélection de modèle** comme une décision \(\hat m(y)\in\mathcal M\), avec une **perte 0–1** (erreur si on ne choisit pas le “vrai” modèle générateur \(m\)) :
\[
e(m,\hat m)=\begin{cases}
1 & \text{si }\hat m\neq m\\
0 & \text{sinon.}
\end{cases}
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

La règle optimale (au sens du risque posterior) est le **MAP** :
\[
\hat m(y)=\arg\min_{\hat m}\ \mathbb E_{p(m\mid y,u)}[e(m,\hat m)]
=\arg\max_{m\in\mathcal M} p(m\mid y,u).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

La **proba d’erreur conditionnelle** (étant donné \(y,u\) et en appliquant la règle optimale) devient :
\[
P_e(y,u)=p(\hat e=1\mid y,u)=1-p(\hat m(y)\mid y,u)=1-\max_m p(m\mid y,u).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 3) Objectif d’optimal design : minimiser l’erreur *attendue* avant d’observer \(y\)

Comme \(y\) n’est pas encore observé au moment de choisir \(u\), ils définissent le **design risk** comme l’erreur moyenne sur \(y\) tiré de la prédiction marginale \(p(y\mid u)\). Le **design optimal** :
\[
u^\star=\arg\min_u \ \mathbb E_{p(y\mid u)}[\hat e]
=\arg\min_u\ p(\hat e=1\mid u).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

Avec
\[
p(\hat e=1\mid u)=\int_{\mathcal Y} p(\hat e=1\mid y,u)\,p(y\mid u)\,dy
=1-\int_{\mathcal Y}\max_m \big[p(m)\,p(y\mid m,u)\big]\,dy.
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

Point clé : cette quantité est **difficile** à calculer (intégrale + max sur les modèles), donc ils proposent une **borne information-théorique**.

---

## 4) Borne de Chernoff via divergence de Jensen–Shannon

Ils introduisent une fonction \(b(u)\) qui borne (au-dessus et au-dessous) le taux d’erreur de sélection :
\[
\frac{1}{4(\bar M-1)}\,b(u)^2 \ \le\ p(\hat e=1\mid u)\ \le\ \frac12\,b(u),
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

avec
\[
b(u)=H(p(m)) - D_{\mathrm{JS}}(u),
\]
où \(H(p(m))\) est l’entropie de Shannon de l’a priori sur les modèles (indépendante de \(u\)). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

La **divergence de Jensen–Shannon** (pondérée par \(p(m)\)) entre les prédictifs \(p(y\mid m,u)\) est :
\[
D_{\mathrm{JS}}(u)
=H\!\Big(\sum_{m} p(m)p(y\mid m,u)\Big)\ -\ \sum_{m} p(m)\,H\!\big(p(y\mid m,u)\big),
\]
et équivalemment
\[
D_{\mathrm{JS}}(u)=\sum_m p(m)\,D_{\mathrm{KL}}\!\Big(p(y\mid m,u)\ \Big\|\ \sum_{m'}p(m')p(y\mid m',u)\Big).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

**Conséquence directe** : comme \(H(p(m))\) ne dépend pas de \(u\), **minimiser** \(b(u)\) revient à **maximiser** \(D_{\mathrm{JS}}(u)\), donc à choisir un design qui rend les **prédictions des modèles aussi séparées que possible**. ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 5) Classe “nonlinear Gaussian” et approximation Laplace–Chernoff

Ils se restreignent ensuite à une classe de modèles génératifs “gaussiens non linéaires” :
\[
\begin{cases}
p(y\mid \vartheta,m,u)=\mathcal N\big(g_m(\vartheta,u),\,Q_m\big)\\
p(\vartheta\mid m)=\mathcal N(\mu_m,\,R_m),
\end{cases}
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

où :
- \(g_m(\vartheta,u)\) est la **mapping d’observation** (non linéaire),
- \(Q_m\) covariance du bruit,
- \(\mu_m,R_m\) moments du prior sur \(\vartheta\). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

Dans ce cadre, ils donnent une **approximation analytique** de la borne (via développement de Taylor/Laplace, détails en suppléments), appelée **Laplace–Chernoff risk** :
\[
b_{\mathrm{LC}}(u)
=H(p(m))
+\frac12\left(
\sum_m p(m)\log|\tilde Q_m(u)|
-\log\left|\sum_m p(m)\big(\Delta g_m\Delta g_m^\top+\tilde Q_m(u)\big)\right|
\right).
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

avec
\[
\Delta g_m = g_m(\mu_m,u)\ -\ \sum_{m'}p(m')\,g_{m'}(\mu_{m'},u),
\]
\[
\tilde Q_m(u)=Q_m+\left.\frac{\partial g_m}{\partial \vartheta}\right|_{\mu_m}\! R_m \left.\frac{\partial g_m}{\partial \vartheta}\right|_{\mu_m}^{\!\top}.
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

**Interprétation** (mathématique) :
- \(\Delta g_m\Delta g_m^\top\) capture la **séparation des moyennes prédictives** entre modèles (au voisinage des priors).
- \(\tilde Q_m(u)\) est une **covariance effective** : bruit \(Q_m\) + incertitude induite par le prior via la jacobienne de \(g_m\).
- Le terme \(\log|\cdot|\) compare “volume” des incertitudes individuelles vs mélange pondéré : c’est une forme fermée (approx.) de séparabilité informationnelle. ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 6) Cas simple à 2 modèles : forme “SNR-like”

Quand \(\bar M=2\), modèles équiprobables et covariances effectives égales \(\tilde Q_1(u)=\tilde Q_2(u)=\tilde Q(u)\), ils obtiennent :
\[
b_{\mathrm{LC}}(u)
=1-\frac12\log\!\left(\frac14\,\frac{\big(g_1(\mu_1,u)-g_2(\mu_2,u)\big)^2}{\tilde Q(u)}+1\right),
\]
ce qui ressemble à une mesure de **résolution de contraste** (distance de moyennes normalisée par la variance, puis log). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

## 7) Lien avec l’optimal design classique (GLM, C-optimalité)

Ils montrent ensuite que, dans le cas particulier **linéaire gaussien** (GLM) :
\[
y=X(u)\vartheta+\varepsilon,
\] ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

le critère Bayésien se relie aux critères fréquentistes d’efficacité (du type C-optimalité). En particulier, ils rappellent une mesure d’efficacité pour un contraste \(c\) :
\[
\zeta(u)=\frac{1}{\sigma^2\,c^\top\big(X(u)^\top X(u)\big)^{-1}c},
\]
et dérivent une forme simplifiée de \(b_{\mathrm{LC}}(u)\) sous priors gaussiens i.i.d., puis montrent que **dans la limite “priors plats”** (\(\alpha^2/\sigma^2\to\infty\)), minimiser \(b_{\mathrm{LC}}(u)\) revient à **maximiser** l’efficacité classique \(\zeta(u)\). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

---

### Synthèse en une ligne
La méthode construit un **objectif d’optimal design pour la comparaison Bayésienne de modèles** : (i) l’objectif exact est la **proba d’erreur de sélection attendue** \(p(\hat e=1\mid u)\), (ii) on la **borne** via une quantité liée à \(D_{\mathrm{JS}}(u)\), (iii) on obtient une **approximation fermée** \(b_{\mathrm{LC}}(u)\) pour des modèles non linéaires gaussiens, exploitable numériquement pour optimiser \(u\). ([journals.plos.org](https://journals.plos.org/ploscompbiol/article/file?id=10.1371%2Fjournal.pcbi.1002280&type=printable))

Si tu veux, je peux aussi te réécrire \(b_{\mathrm{LC}}(u)\) sous une forme “moments du prior prédictif” (moyenne/covariance des \(p(y\mid m,u)\)), car c’est exactement la forme que tu manipules avec tes bornes (Chernoff / Laplace) dans ton projet.
