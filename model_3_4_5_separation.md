# Separation analytique des modeles 3, 4 et 5 (action_avg)

Ce document resume comment separer analytiquement les modeles 3/4/5 (kernel=action_avg) et propose des leviers de design pour maximiser la separabilite.

## Rappel: quels modeles sont 3, 4, 5

Dans `test_multi_model.py`, `build_six_models()` construit 6 modeles:
- Modeles 0..2: kernel `event_weighted` avec update `continuous`, `event`, `action`.
- Modeles 3..5: kernel `action_avg` avec update `continuous`, `event`, `action`.

Ici on se concentre sur les modeles 3,4,5:
- M3 = action_avg + update=continuous
- M4 = action_avg + update=event
- M5 = action_avg + update=action

## Forme analytique (action_avg)

Pour une action de type $k$ avec effort $(e, t_e)$ et reward $(r, t_r)$, le noyau action_avg fait la moyenne sur les types:

$h(t) = h_0 + \frac{1}{K} \sum_{k=1}^K \left[ \sum_{(e,t_e) \in E_k} w_e \, e \, \gamma^{t_k(t)-t_e} \, I(t_e < t_k(t)) + \sum_{(r,t_r) \in R_k} w_r \, r \, \gamma^{t_k(t)-t_r} \, I(t_r < t_k(t)) \right] + w_t \, t$

avec $K = K_{types}$ et $I(\cdot)$ l'indicateur.

La difference entre M3/M4/M5 vient uniquement du choix du temps "snappe" $t_k(t)$ pour chaque type:

- M3 (continuous): $t_k(t) = t$
- M4 (event): $t_k(t) = \max\{t_e, t_r \le t\}$ pour le type $k$, sinon $t_k(t) = t$
- M5 (action): $t_k(t) = \max\{t_r \le t\}$ pour le type $k$, sinon $t_k(t) = t_0$ (sentinel), avec $t_0 < \min(t)$

Le sentinel $t_0$ fait que, s'il n'y a pas encore de reward du type $k$, alors toutes les contributions de ce type sont nulles (les indicateurs sont faux).

## Leviers analytiques pour separer M3, M4, M5

### 1) Placer des temps de mesure ENTRE Eff et Rew d'un type

Si on choisit un $t$ tel que $t_e < t < t_r$ pour un type $k$:
- M4 (event) utilise $t_k(t) = t_e$ (dernier event), donc l'effort contribue a plein ($\gamma^0$).
- M5 (action) utilise encore le sentinel si aucun reward n'est passe, donc contribution nulle pour ce type.

On obtient un contraste clair:

$h_4(t) - h_5(t) \approx \frac{1}{K} \, w_e \, e$

C'est le levier le plus "propre" pour separer M4 vs M5.

### 2) Mesures tres APRES le dernier event

Pour un $t$ bien apres le dernier event/reward du type $k$:
- M3 calcule avec $t_k(t) = t$ (decroissance plus forte).
- M4/M5 snappe sur le dernier event/reward, donc $t_k(t) < t$ et la decroissance est plus faible.

Cela cree une difference:

$h_3(t) - h_4(t) = \frac{1}{K} \sum_k \sum_{ev} w \, v \, (\gamma^{t-t_{ev}} - \gamma^{t_k-t_{ev}})$

Comme $t > t_k$ et $0 < \gamma < 1$, on a $\gamma^{t-t_{ev}} < \gamma^{t_k-t_{ev}}$, donc $h_4(t)$ est plus grand en amplitude.

Levier utile pour separer M3 vs M4/M5.

### 3) Fenetres longues Eff -> Rew (grands deltas)

Augmenter $\Delta_k = t_r - t_e$ pour un type augmente la largeur de la fenetre ou:
- M4 a une contribution d'effort
- M5 reste encore au sentinel (pas de reward)

Cela agrandit le domaine temporel ou M4 et M5 sont differents.

### 4) Activer UN type a la fois (eviter l'annulation par la moyenne)

action_avg moyenne sur $K$ types, ce qui dilue le contraste par un facteur $1/K$.
Pour maximiser la separabilite:
- regrouper les actions d'un seul type dans une fenetre,
- placer des mesures dans cette fenetre,
- eviter des actions simultanees de types differents a ce moment-la.

Cela evite que les contributions d'autres types (souvent de signe oppose) "annulent" le signal.

## Conseils pratiques (a impact analytique direct)

1) Forcer des mesures juste apres Eff (ex: $t = t_e + \epsilon$) avant Rew pour chaque type.
2) Ajouter quelques mesures tres tardives apres le dernier event.
3) Allonger le delai Eff -> Rew pour au moins un type afin de creer un large intervalle "Eff-only".
4) Segmenter le temps par type (un type a la fois) pour limiter la moyenne et garder un signal net.

Ces leviers n'alterent pas la definition des modeles, uniquement le design (choix des temps et timing des actions), donc la comparaison reste valide.
