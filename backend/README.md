# Smart Developer Backend
Demo backend for serving the current Smart Developer algorithm pipeline and static frontend.

## Run
From repo root:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8002
```

Then open:

```text
http://localhost:8001
```

If the existing frontend build is not wired to the new API yet, use:
```text
http://localhost:8001/demo
```

## API
* `GET /api/health`
* `POST /api/retrieve-sites`
* `POST /api/feedback`
* `POST /api/report-jobs`
* `GET /api/report-jobs/{report_id}`
* `GET /api/report-jobs/{report_id}/pdf`
* `GET /api/report-jobs/{report_id}/markdown`