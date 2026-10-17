-- What an analyst would have seen on each day of a monitored experiment: for look day d,
-- the users whose analysis window has ended by start + d days, per variant.
-- Parameters: $start, $window_days, $first_day, $last_day. {metric} is a catalog column.
WITH looks AS (
    SELECT range AS day FROM range(CAST($first_day AS BIGINT), CAST($last_day AS BIGINT) + 1)
)

SELECT
    l.day,
    m.variant,
    count(*) AS n,
    avg(m.{metric}) AS mean,
    var_samp(m.{metric}) AS variance
FROM looks AS l
INNER JOIN user_metrics AS m
    ON m.exposed_at + to_days(CAST($window_days AS INTEGER))
        <= $start + to_days(CAST(l.day AS INTEGER))
GROUP BY l.day, m.variant
ORDER BY l.day, m.variant
