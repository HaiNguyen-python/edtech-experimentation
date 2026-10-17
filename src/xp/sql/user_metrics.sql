-- One row per exposed user of one experiment.
-- Parameters: $experiment, $window_days, $pre_days, $as_of (only users whose whole analysis
-- window has ended by $as_of are included, so a user is never counted with half a window).
WITH first_exposure AS (
    -- The exposure log has one row per visit: keep each user's first exposure and its variant.
    SELECT
        user_id,
        min(exposed_at) AS exposed_at,
        arg_min(variant, exposed_at) AS variant
    FROM exposures
    WHERE experiment = $experiment
    GROUP BY user_id
),

cohort AS (
    SELECT f.user_id, f.variant, f.exposed_at, u.platform
    FROM first_exposure AS f
    INNER JOIN users AS u USING (user_id)
    WHERE f.exposed_at + to_days(CAST($window_days AS INTEGER)) <= $as_of
),

user_events AS (
    -- Events from the pre-period [exposure - pre_days, exposure) and the analysis window
    -- [exposure, exposure + window_days). Nothing before exposure counts as an outcome.
    SELECT
        c.user_id,
        e.event_name,
        e.event_time,
        e.event_time >= c.exposed_at AS in_window
    FROM cohort AS c
    INNER JOIN events AS e
        ON e.user_id = c.user_id
        AND e.event_time >= c.exposed_at - to_days(CAST($pre_days AS INTEGER))
        AND e.event_time < c.exposed_at + to_days(CAST($window_days AS INTEGER))
),

per_user AS (
    SELECT
        user_id,
        count(*) FILTER (WHERE in_window AND event_name = 'lesson_complete') AS lessons_completed,
        count(*) FILTER (WHERE in_window AND event_name = 'lesson_start') AS lessons_started,
        count(DISTINCT CAST(event_time AS DATE))
            FILTER (WHERE in_window AND event_name = 'lesson_complete') AS active_days,
        count(*) FILTER (WHERE in_window AND event_name = 'notif_unsubscribe') > 0 AS unsubscribed,
        count(*) FILTER (WHERE in_window AND event_name = 'app_crash') > 0 AS crashed,
        count(*) FILTER (WHERE NOT in_window AND event_name = 'lesson_complete') AS pre_lessons
    FROM user_events
    GROUP BY user_id
)

-- Users with no events at all still count (with zeros): dropping them would bias the metric.
SELECT
    c.user_id,
    c.variant,
    c.platform,
    c.exposed_at,
    coalesce(p.lessons_completed, 0) AS lessons_completed,
    coalesce(p.lessons_started, 0) AS lessons_started,
    coalesce(p.lessons_completed, 0) * 7.0 / $window_days AS lessons_completed_per_week,
    coalesce(p.active_days, 0) AS active_days,
    CAST(coalesce(p.unsubscribed, false) AS INTEGER) AS unsubscribed,
    CAST(coalesce(p.crashed, false) AS INTEGER) AS crashed,
    coalesce(p.pre_lessons, 0) * 7.0 / $pre_days AS pre_lessons_completed_per_week
FROM cohort AS c
LEFT JOIN per_user AS p USING (user_id)
ORDER BY c.user_id
