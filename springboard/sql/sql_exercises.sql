-- OK..so this engine seems to require " " to recognize formatting changes in names. OK.
OK, it is Postgres.

-- ILIKE is like ignorecase. Again I like PostGres

SELECT
 year       AS "Year"
,month      AS "Month"
,month_name AS "Month name"
,south      AS "South"
,west       AS "West"
,midwest    AS "Midwest"
,northeast  AS "Northeast"
FROM tutorial.us_housing_units

