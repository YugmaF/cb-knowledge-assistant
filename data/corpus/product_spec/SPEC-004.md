---
doc_id: SPEC-004
title: Real-time Fraud Alerts
department: risk
document_type: product_spec
access_level: internal
created_date: 2026-02-23
tags: [fraud, alerts, product-spec, push-notifications, fraud-engine]
---

## Product Summary

Real-time Fraud Alerts notifies Commercial Bank customers within seconds when a card or account transaction looks suspicious, and lets the customer confirm or reject the transaction from the mobile banking app. Real-time Fraud Alerts reduces fraud losses and false declines. The product owner is Grace Mwangi, fraud product manager in the risk department.

## Alert Triggers

Real-time Fraud Alerts are triggered by the fraud-engine risk score. A transaction with a medium risk score is approved and the customer receives an informational alert. A transaction with a high risk score is held for up to 5 minutes and the customer is asked to confirm it. Typical triggers are card transactions in a new country, rapid transactions at several merchants (velocity rules), a first transfer to a new payee above 100,000, and logins from a new device followed by a transfer.

## Alert Delivery Time

The target for Real-time Fraud Alerts is delivery to the customer's phone within 10 seconds of the transaction for 95 percent of alerts, and within 30 seconds for 99 percent. Alerts are sent as high-priority push notifications. If the push notification is not opened within 60 seconds for a held transaction, an SMS is sent as a fallback.

## Customer Response

In Real-time Fraud Alerts the customer taps "Yes, this was me" or "No, block my card". If the customer confirms, the held transaction is released and similar transactions at that merchant are not alerted again for 24 hours. If the customer rejects, the card is blocked immediately, the transaction is declined, and the fraud operations team calls the customer. If there is no response within 5 minutes, a held card transaction is declined and a held transfer stays pending for fraud operations review.

## Architecture

Real-time Fraud Alerts uses the fraud-engine, which scores each transaction synchronously for payments-api and the cardnet-gateway-adapter within 80 ms. Alerts and customer responses are published to the payment-events Kafka topic, consumed by the notification service and the fraud case management system. Fraud rules and thresholds are configuration: every fraud rule change must pass simulation against the previous 24 hours of transactions and a canary, as required by the Change Management Policy.

## Privacy

Real-time Fraud Alerts messages show only the merchant name, amount and the last four digits of the card. Alerts never include full card numbers or account balances. Fraud rules are classified confidential and are not described in customer communications.

## Success Metrics

Real-time Fraud Alerts success is measured by a 25 percent reduction in card fraud losses, a false positive rate below 2 percent of alerted transactions, and customer response to held-transaction alerts within 2 minutes on average.
