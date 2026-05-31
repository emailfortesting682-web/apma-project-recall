# Streamlit Cloud + Supabase Deployment

Use this setup when the client needs a direct web URL and persistent MVP data.

## 1. Create Supabase Project

1. Go to Supabase and create a free project.
2. Open SQL Editor.
3. Run this schema:

```sql
create table if not exists public.apma_memories (
  memory_id text primary key,
  name text not null,
  records jsonb not null default '[]'::jsonb,
  embeddings jsonb,
  metadata jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.apma_kv (
  key text primary key,
  value jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);
```

## 2. Collect Secrets

From Supabase Project Settings > API:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

From OpenAI:

- `OPENAI_API_KEY`

Keep the service role key private. Add it only in Streamlit Cloud secrets, never in GitHub.

## 3. Streamlit Cloud Settings

Deploy with:

- Repository: your GitHub repo
- Branch: `main`
- Main file path: `project_recall_ai/apma_app.py`

Add secrets:

```toml
OPENAI_API_KEY = "sk-..."
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
```

## 4. Behavior

When Supabase secrets are present, the app stores:

- Users in `apma_kv`
- Manual entry config in `apma_kv`
- Summary templates in `apma_kv`
- Memories and embeddings in `apma_memories`

When Supabase secrets are missing, the app falls back to local `data/` files for development.

## 5. Client Testing Flow

1. Deploy the app on Streamlit Community Cloud.
2. Open the generated `.streamlit.app` URL.
3. Create one test user.
4. Upload a small CSV/Excel file.
5. Run semantic search.
6. Export a report.
7. Share the URL and test credentials with the client.
