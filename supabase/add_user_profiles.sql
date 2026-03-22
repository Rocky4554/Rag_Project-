-- Migration: Add user_profiles table for cross-session interview memory
-- Run this in Supabase SQL Editor if you already have the base schema

create table if not exists user_profiles (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references auth.users(id) on delete cascade unique not null,
    topics_covered jsonb default '[]',
    weak_areas jsonb default '[]',
    strong_areas jsonb default '[]',
    score_history jsonb default '[]',
    performance_summary text,
    total_interviews int default 0,
    avg_score numeric default 0,
    last_session_at timestamptz,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

alter table user_profiles enable row level security;

create policy "Users can view own profile"
    on user_profiles for select using (auth.uid() = user_id);
create policy "Users can insert own profile"
    on user_profiles for insert with check (auth.uid() = user_id);
create policy "Users can update own profile"
    on user_profiles for update using (auth.uid() = user_id);

create index if not exists idx_user_profiles_user_id on user_profiles(user_id);
