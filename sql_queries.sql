-- Run against the SQLite launches table created in outputs/spacex.sqlite
SELECT DISTINCT LaunchSite FROM launches ORDER BY LaunchSite;
SELECT * FROM launches WHERE LaunchSite LIKE 'CCA%' LIMIT 5;
SELECT LaunchSite, SUM(PayloadMass) AS total_payload_kg FROM launches GROUP BY LaunchSite;
SELECT BoosterVersion, AVG(PayloadMass) AS average_payload_kg FROM launches GROUP BY BoosterVersion;
SELECT Class, COUNT(*) AS count FROM launches GROUP BY Class;
SELECT BoosterVersion, PayloadMass, LaunchSite, Date FROM launches
 WHERE PayloadMass=(SELECT MAX(PayloadMass) FROM launches);
SELECT Year, COUNT(*) AS launch_count, AVG(Class) AS success_rate FROM launches GROUP BY Year ORDER BY Year;
