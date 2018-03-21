# One-dimensional (or Earth Mover's Distance (EMD)) optimal transport implementation for $L^2$ distance

Implementation of the optimal transport distance for densities $f(x)$ and $g(y)$ for $x,y \in \mathbb{R}$ and the $L^2$ distance.

In this case the $L^2$ distance is strictly convex so can use the northwest corner method, which has complexity O(m + n - 1).
