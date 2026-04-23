# Django API for Content Moderation

This Django project exposes your content moderation logic as:

- A REST API endpoint for programmatic access (API selling)
- A simple web form for manual moderation

## Features

- REST API: `/api/moderate/` (POST, returns moderation result)
- Web form: `/moderate/` (submit text, see result)
- API key authentication for API endpoint
- Uses your existing moderation logic from `src.moderate`

## Quickstart

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run migrations:
   ```bash
   python manage.py migrate
   ```
3. Create a superuser (for admin/API key management):
   ```bash
   python manage.py createsuperuser
   ```
4. Start the server:
   ```bash
   python manage.py runserver
   ```
5. Test the API:
   - POST to `/api/moderate/` with `{ "text": "your text" }` and your API key
   - Or visit `/moderate/` in your browser
