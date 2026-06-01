# AGENTS.md

## Backend Development Style

When adding new backend features, use a modular FastAPI structure instead of putting business logic directly in `main.py`.

`main.py` should only be the application entry point. It should create the FastAPI app, configure middleware, and register routers with `app.include_router(...)`.

Follow this separation of responsibilities:

```text
backend/app/
├── crud/                  # Database create/read/update/delete logic
│   └── <module>.py
├── models/                # Database models / ORM models
│   └── <module>.py
├── routers/               # API routes, grouped by feature/module
│   └── <module>.py
├── schemas/               # Pydantic request/response models
│   └── <module>.py
├── services/              # Business logic and integrations
│   └── <module>.py
├── utils/                 # Shared helper functions
├── config/                # Configuration, database, cache settings
│   ├── db_conf.py
│   └── cache_conf.py
└── main.py                # Application entry point
```

For a new feature, create files by feature name. For example, a recommendation feedback feature should be split like this:

```text
backend/app/
├── routers/
│   └── recommendation_feedback.py
├── schemas/
│   └── feedback.py
└── services/
    └── recommendation_feedback.py
```

Guidelines:

- Put route definitions in `routers/<feature>.py`.
- Put request and response validation models in `schemas/<feature>.py`.
- Put business logic, payload construction, external service calls, and reusable helpers in `services/<feature>.py`.
- Put database operations in `crud/<feature>.py` when persistence is needed.
- Put database table/model definitions in `models/<feature>.py` when persistence is needed.
- Keep `main.py` clean. Do not add feature-specific endpoint logic directly in `main.py`.
- Register new routers from `main.py` using `app.include_router(...)`.
- Keep each module focused on one feature area, similar to:

```text
FirstNews_backend/
├── crud/
│   ├── favorite.py
│   ├── history.py
│   ├── news.py
│   └── users.py
├── models/
│   ├── favorite.py
│   ├── history.py
│   ├── news.py
│   └── users.py
├── routers/
│   ├── favorite.py
│   ├── history.py
│   ├── news.py
│   └── users.py
├── schemas/
│   ├── favorite.py
│   ├── history.py
│   ├── news.py
│   └── users.py
├── utils/
├── config/
│   ├── db_conf.py
│   └── cache_conf.py
├── main.py
└── test_main.http
```
