-- Shot-level contextual features using CTEs, joins, and window functions.
-- Expects tables: games, events, shots (base columns populated by ingestion).

DROP TABLE IF EXISTS shots_with_context;

CREATE TABLE shots_with_context AS
WITH ev AS (
    SELECT
        game_id,
        event_id,
        sort_order,
        period,
        game_seconds,
        type_desc_key,
        team_id,
        situation_code,
        LAG(type_desc_key) OVER (PARTITION BY game_id ORDER BY sort_order) AS prev_event_type,
        LAG(team_id) OVER (PARTITION BY game_id ORDER BY sort_order) AS prev_team_id,
        LAG(game_seconds) OVER (PARTITION BY game_id ORDER BY sort_order) AS prev_game_seconds,
        LEAD(type_desc_key) OVER (PARTITION BY game_id ORDER BY sort_order) AS next_event_type,
        ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY sort_order) AS event_seq
    FROM
        events
),
shot_rows AS (
    SELECT
        s.*,
        g.home_team_id,
        g.away_team_id,
        g.home_defending_side AS game_home_defending_side,
        e.prev_event_type,
        e.prev_team_id,
        e.prev_game_seconds,
        e.next_event_type,
        e.event_seq,
        (s.game_seconds - e.prev_game_seconds) AS time_since_prev_event_seconds
    FROM
        shots s
        JOIN games g ON g.game_id = s.game_id
        JOIN ev e ON e.game_id = s.game_id
        AND e.event_id = s.event_id
)
SELECT
    sr.*,
    CASE
        WHEN sr.prev_event_type IN ('shot-on-goal', 'goal', 'missed-shot')
        AND sr.prev_team_id = sr.team_id
        AND sr.time_since_prev_event_seconds IS NOT NULL
        AND sr.time_since_prev_event_seconds <= 3 THEN 1
        ELSE 0
    END AS rebound_flag,
    CASE
        WHEN sr.prev_event_type IN ('shot-on-goal', 'goal', 'missed-shot')
        AND sr.prev_team_id = sr.team_id THEN 1
        ELSE 0
    END AS same_team_prev_shot_flag,
    CASE
        WHEN sr.prev_event_type IN ('faceoff', 'takeaway', 'giveaway', 'hit', 'blocked-shot')
        AND sr.time_since_prev_event_seconds IS NOT NULL
        AND sr.time_since_prev_event_seconds < 4 THEN 1
        ELSE 0
    END AS rush_flag_heuristic,
    CASE
        WHEN LENGTH(sr.situation_code) = 4 THEN SUBSTR(sr.situation_code, 1, 1)
        ELSE NULL
    END AS situation_digit_1,
    CASE
        WHEN LENGTH(sr.situation_code) = 4 THEN SUBSTR(sr.situation_code, 2, 1)
        ELSE NULL
    END AS situation_digit_2,
    CASE
        WHEN LENGTH(sr.situation_code) = 4 THEN SUBSTR(sr.situation_code, 3, 1)
        ELSE NULL
    END AS situation_digit_3,
    CASE
        WHEN LENGTH(sr.situation_code) = 4 THEN SUBSTR(sr.situation_code, 4, 1)
        ELSE NULL
    END AS situation_digit_4
FROM
    shot_rows sr;

CREATE INDEX IF NOT EXISTS idx_swc_season_date ON shots_with_context (season, game_date);
CREATE INDEX IF NOT EXISTS idx_swc_shooter ON shots_with_context (shooter_id);
