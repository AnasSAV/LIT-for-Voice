# Frontend (React + TypeScript + Vite)

## Overview
This is the React 18 + TypeScript frontend for LIT for Voice. It uses Vite, Tailwind CSS, shadcn/ui (Radix UI), TanStack Query, Plotly.js, and WaveSurfer.js.

## Prerequisites
- Node.js v18+
- npm (or bun)

## Quick Start
```bash
# From repo root
cd Frontend

# Install deps
npm install

# Start dev server (http://localhost:8080)
npm run dev
```

## Scripts
- `npm run dev` — Start Vite dev server
- `npm run build` — Build for production
- `npm run build:dev` — Build with development mode
- `npm run lint` — Run ESLint
- `npm run preview` — Preview production build

## Environment Variables
- API base URL is read in `src/lib/api/datasets.ts`.
  - Default: `http://localhost:8000`
  - Override via Vite env var:
    ```
    # Frontend/.env
    VITE_API_BASE=http://localhost:8000
    ```

## Local Development Notes
- Vite dev server is configured at port 8080 in `vite.config.ts`.
- Backend CORS (`Backend/app/main.py`) allows origins: `http://localhost:8080` and `http://localhost:5173`.
- Fetch calls include credentials by default in dataset APIs; ensure backend cookies (SameSite/Domain) match your setup when testing across domains.

## Directory Structure
```
src/
├── components/
│   ├── analysis/
│   ├── audio/
│   ├── layout/
│   ├── panels/
│   ├── ui/
│   └── visualization/
├── hooks/
├── lib/
│   └── api/
├── pages/
├── App.tsx
├── main.tsx
└── index.css
```

## Backend Integration
- Dataset API client: `src/lib/api/datasets.ts`
  - `GET /datasets` — list datasets
  - `GET /datasets/active` — active dataset
  - `POST /datasets/select` — set active dataset
  - `GET /datasets/files?limit&offset` — list `.wav` files
  - `GET /datasets/file?relpath[&id]` — stream a wav file
- See full endpoint docs in `../README.md` and `../WARP.md`.

## Common Issues
- Port conflicts: change Vite port in `vite.config.ts` if 8080 is occupied.
- CORS/cookies: if using a non-default host/port, set backend cookie settings in `Backend/app/core/settings.py` (e.g., `COOKIE_SAMESITE=none` for cross-site over HTTPS).
