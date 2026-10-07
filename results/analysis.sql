
WITH item_totals AS (
 SELECT order_id, SUM(price) AS gmv, SUM(freight_value) AS freight
 FROM order_items GROUP BY order_id
), payment_totals AS (
 SELECT order_id, SUM(payment_value) AS payment_total
 FROM payments GROUP BY order_id
)
SELECT COUNT(*) AS orders, SUM(it.gmv) AS gmv, SUM(it.freight) AS freight,
       SUM(pt.payment_total) AS payments
FROM orders o
LEFT JOIN item_totals it ON it.order_id=o.order_id
LEFT JOIN payment_totals pt ON pt.order_id=o.order_id
WHERE o.order_status='delivered'
;

SELECT i.seller_id, COUNT(DISTINCT i.order_id) AS orders,
 SUM(i.price) AS delivered_gmv
 FROM order_items i JOIN orders o ON i.order_id=o.order_id
 WHERE o.order_status='delivered' GROUP BY i.seller_id
 ORDER BY delivered_gmv DESC LIMIT 10;
