-- Users per variant and platform, used by the sample ratio mismatch (SRM) check and the
-- platform-mix balance check. Counted on distinct exposed users, not on exposure rows.
SELECT platform, variant, count(*) AS users
FROM user_metrics
GROUP BY platform, variant
ORDER BY platform, variant
