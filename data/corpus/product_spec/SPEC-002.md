---
doc_id: SPEC-002
title: Card Tokenization
department: payments
document_type: product_spec
access_level: internal
created_date: 2025-10-13
tags: [cards, tokenization, product-spec, wallets, security]
---

## Product Summary

Card Tokenization replaces the real card number (PAN) of a Commercial Bank debit or credit card with a unique token for use in mobile wallets, merchant card-on-file and in-app payments. Card Tokenization reduces fraud because a stolen token cannot be used outside the device or merchant it was issued for. Card Tokenization is delivered by payments engineering with the CardNet Gateway token service.

## Customer Experience

Customers add a Commercial Bank card to a mobile wallet or merchant from the mobile banking app with one tap ("Add to wallet"), or by entering the card at the merchant. For Card Tokenization, the customer confirms with an OTP sent to their registered mobile number unless the request comes from inside the authenticated mobile banking app. Customers can see every active token for each card in the app, with the device or merchant name, and can suspend or delete any token without blocking the physical card.

## Token Types

Card Tokenization supports three token types. Device tokens are bound to one phone or wearable and used for contactless payments. Merchant card-on-file tokens are bound to one merchant for subscriptions and one-click checkout. In-app tokens are used by the mobile banking app itself for bill payments by card.

## Token Lifecycle

A Card Tokenization token is provisioned through the CardNet Gateway token service after the bank approves the request. When a physical card is reissued or replaced, all its tokens are automatically updated to the new card without customer action. When a card is closed, all tokens are deleted. Tokens not used for 18 months are suspended.

## Security Requirements

Card Tokenization must never store the full card number outside the card vault. Token-to-PAN mapping is held by the CardNet Gateway token service; the bank keeps only the token reference and last four digits. Cryptogram validation for token transactions is performed in the HSM cluster. Provisioning decisions use the fraud-engine score; requests with a high risk score require call-centre verification.

## Integration

Card Tokenization adds three APIs to the cardnet-gateway-adapter: token provisioning approval, token lifecycle events and token transaction authorisation. Token lifecycle events are published to the payment-events Kafka topic so the mobile app and the fraud-engine see changes in real time.

## Non-Functional Requirements

Card Tokenization provisioning must complete within 15 seconds at p95. Token transaction authorisation must add no more than 50 ms to authorisation latency. The token service integration inherits tier 1 availability, and its TLS certificates are covered by automated rotation.

## Success Metrics

Card Tokenization success is measured by the share of card-not-present transactions using tokens (target 60 percent within 12 months), a 30 percent reduction in card-not-present fraud losses, and provisioning success above 95 percent.
