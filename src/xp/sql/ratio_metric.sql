-- Ratio metric sum(numerator) / sum(denominator) per variant, with its delta-method variance.
-- {numerator} and {denominator} are column names from the metric catalog, checked against
-- an identifier pattern before they are put into the query.
--
-- With per-user X (numerator) and Y (denominator), R = mean(X) / mean(Y) and
--   Var(R) ~= (1/n) * [ var(X)/mu_y^2 - 2 mu_x cov(X,Y)/mu_y^3 + mu_x^2 var(Y)/mu_y^4 ].
-- The user, not the lesson, is the unit of randomisation, so lessons of one user are not
-- independent; this formula accounts for that, a naive binomial variance would not.
WITH stats AS (
    SELECT
        variant,
        count(*) AS n,
        avg({numerator}) AS mu_x,
        avg({denominator}) AS mu_y,
        var_samp({numerator}) AS var_x,
        var_samp({denominator}) AS var_y,
        covar_samp({numerator}, {denominator}) AS cov_xy
    FROM user_metrics
    GROUP BY variant
)

SELECT
    variant,
    n,
    mu_x / mu_y AS ratio,
    (
        var_x / mu_y ^ 2
        - 2 * mu_x * cov_xy / mu_y ^ 3
        + mu_x ^ 2 * var_y / mu_y ^ 4
    ) / n AS ratio_variance
FROM stats
ORDER BY variant
