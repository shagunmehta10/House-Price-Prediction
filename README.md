# HouseAI — Frontend

A React + Vite + Tailwind frontend for HouseAI, an ML-powered property price
prediction platform. Black / yellow / white theme, matching the provided design.

## Pages

| Route        | Access   | Purpose                                             |
|--------------|----------|------------------------------------------------------|
| `/`          | Public   | Landing page (hero, features)                        |
| `/login`     | Public   | Log in                                                |
| `/signup`    | Public   | Create account                                        |
| `/predict`   | Private  | Prediction form → predicted price                     |
| `/dashboard` | Private  | Stats + charts (trend line, predictions by location)  |
| `/history`   | Private  | Table of past predictions, with delete                |
| `/favorites` | Private  | Saved properties, with remove                         |
| `/profile`   | Private  | Edit profile + change password                        |
| Logout       | —        | Button in the navbar, clears the session              |

"Private" routes redirect to `/login` if you're not authenticated (see
`src/components/ProtectedRoute.jsx`).

## Getting started

```bash
npm install
cp .env.example .env   # then set VITE_API_BASE_URL to your backend
npm run dev
```

## Connecting your backend

All API calls live in **`src/api/client.js`**. It's an axios instance that:

- Prefixes every request with `VITE_API_BASE_URL` (defaults to `/api`).
- Attaches `Authorization: Bearer <token>` automatically once you're logged in
  (token is read from `localStorage.houseai_token`).
- Clears the stored session on any `401` response.

Expected endpoints (adjust paths/payloads in `client.js` to match your API):

```
POST   /auth/login              { email, password }        -> { token, user }
POST   /auth/signup             { name, email, password }  -> { token, user }
POST   /auth/logout
GET    /auth/me                                             -> user

POST   /predict                 { location, area, bedrooms, bathrooms, ... }
                                                              -> { predicted_price, confidence }
GET    /predictions/history                                 -> [ { id, location, area, bedrooms,
                                                                     predicted_price, created_at } ]
DELETE /predictions/history/:id

GET    /favorites                                            -> [ { id, location, area, bedrooms,
                                                                      bathrooms, predicted_price } ]
POST   /favorites                { ...propertyFields }
DELETE /favorites/:id

GET    /dashboard/summary                                    -> { total_predictions, average_price,
                                                                     favorites_count, locations_covered }
GET    /dashboard/trends                                     -> { trends: [{month, avgPrice}],
                                                                     by_location: [{location, predictions}] }

GET    /profile                                               -> { name, email, phone }
PUT    /profile                  { name, email, phone }
POST   /profile/change-password  { current_password, new_password }
```

If a call fails, pages show an inline error state instead of crashing — the
Dashboard also falls back to sample data so the UI is inspectable before your
backend is wired up.

## Theme

Colors are defined in `tailwind.config.js` under `ink` (near-black) and `gold`
(the accent yellow), plus white as the base. Adjust the hex values there to
retune the palette globally.

## Build

```bash
npm run build   # outputs to dist/
npm run preview
```
