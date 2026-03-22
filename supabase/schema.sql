-- ═══════════════════════════════════════════════════════════════════
-- Supabase Schema for RAG Interview Platform
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ═══════════════════════════════════════════════════════════════════

-- 1. Documents — tracks uploaded PDFs
create table if not exists documents (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    session_id text unique not null,
    filename text not null,
    original_name text,
    qdrant_collection text not null,
    page_count int,
    chunk_count int,
    created_at timestamptz default now()
);

-- 2. Chat messages — persistent chat history per document
create table if not exists chat_messages (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    document_id uuid references documents(id) on delete cascade not null,
    role text not null check (role in ('user', 'ai')),
    content text not null,
    created_at timestamptz default now()
);

-- 3. Interview results — final reports from LangGraph interviews
create table if not exists interview_results (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    document_id uuid references documents(id) on delete cascade not null,
    thread_id text not null,
    questions_asked int,
    scores jsonb default '[]',
    topic_scores jsonb default '{}',
    final_report text,
    difficulty_level text,
    created_at timestamptz default now()
);

-- 4. Quiz results — quiz attempts and scores
create table if not exists quiz_results (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    document_id uuid references documents(id) on delete cascade not null,
    topic text,
    questions jsonb not null,
    score numeric,
    total_questions int,
    created_at timestamptz default now()
);

-- 5. Activity log — tracks user actions for "welcome back" context
create table if not exists activities (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade not null,
    action text not null,
    metadata jsonb default '{}',
    created_at timestamptz default now()
);

-- 6. User profiles — cross-session memory for personalized interviews
create table if not exists user_profiles (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade unique not null,
    topics_covered jsonb default '[]',        -- ["machine learning", "neural networks", ...]
    weak_areas jsonb default '[]',            -- ["backpropagation", "regularization", ...]
    strong_areas jsonb default '[]',          -- ["data preprocessing", "model evaluation", ...]
    score_history jsonb default '[]',         -- [{date, score, topic, docName}, ...]
    performance_summary text,                 -- LLM-generated narrative summary
    total_interviews int default 0,
    avg_score numeric default 0,
    last_session_at timestamptz,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- ═══════════════════════════════════════════════════════════════════
-- Row Level Security (RLS) — users can only access their own data
-- ═══════════════════════════════════════════════════════════════════

alter table documents enable row level security;
alter table chat_messages enable row level security;
alter table interview_results enable row level security;
alter table quiz_results enable row level security;
alter table activities enable row level security;
alter table user_profiles enable row level security;

-- Documents
create policy "Users can view own documents"
    on documents for select using (auth.uid() = user_id);
create policy "Users can insert own documents"
    on documents for insert with check (auth.uid() = user_id);
create policy "Users can delete own documents"
    on documents for delete using (auth.uid() = user_id);

-- Chat messages
create policy "Users can view own chat messages"
    on chat_messages for select using (auth.uid() = user_id);
create policy "Users can insert own chat messages"
    on chat_messages for insert with check (auth.uid() = user_id);

-- Interview results
create policy "Users can view own interview results"
    on interview_results for select using (auth.uid() = user_id);
create policy "Users can insert own interview results"
    on interview_results for insert with check (auth.uid() = user_id);

-- Quiz results
create policy "Users can view own quiz results"
    on quiz_results for select using (auth.uid() = user_id);
create policy "Users can insert own quiz results"
    on quiz_results for insert with check (auth.uid() = user_id);

-- Activities
create policy "Users can view own activities"
    on activities for select using (auth.uid() = user_id);
create policy "Users can insert own activities"
    on activities for insert with check (auth.uid() = user_id);

-- User profiles
create policy "Users can view own profile"
    on user_profiles for select using (auth.uid() = user_id);
create policy "Users can insert own profile"
    on user_profiles for insert with check (auth.uid() = user_id);
create policy "Users can update own profile"
    on user_profiles for update using (auth.uid() = user_id);

-- ═══════════════════════════════════════════════════════════════════
-- Indexes for performance
-- ═══════════════════════════════════════════════════════════════════

create index if not exists idx_documents_user_id on documents(user_id);
create index if not exists idx_documents_session_id on documents(session_id);
create index if not exists idx_chat_messages_document_id on chat_messages(document_id);
create index if not exists idx_chat_messages_user_id on chat_messages(user_id);
create index if not exists idx_interview_results_user_id on interview_results(user_id);
create index if not exists idx_quiz_results_user_id on quiz_results(user_id);
create index if not exists idx_activities_user_id on activities(user_id);
create index if not exists idx_user_profiles_user_id on user_profiles(user_id);
