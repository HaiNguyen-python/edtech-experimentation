-- Per-variant summary of every numeric per-user column: n, mean, sample variance.
-- These three numbers are all a Welch t-test needs, so the user-level table never has to
-- leave the database for plain mean metrics.
WITH long AS (
    UNPIVOT (
        SELECT * EXCLUDE (user_id, platform, exposed_at) FROM user_metrics
    )
    ON COLUMNS(* EXCLUDE (variant))
    INTO NAME metric VALUE value
)

SELECT
    variant,
    metric,
    count(*) AS n,
    avg(value) AS mean,
    var_samp(value) AS variance
FROM long
GROUP BY variant, metric
ORDER BY metric, variant
