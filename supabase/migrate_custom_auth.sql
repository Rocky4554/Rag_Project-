-- ═══════════════════════════════════════════════════════════════════
-- Migration: Custom JWT Auth (replaces Supabase Auth)
-- Run this in Supabase SQL Editor
-- ═══════════════════════════════════════════════════════════════════

-- 1. Users table
create table if not exists users (
    id uuid default gen_random_uuid() primary key,
    email text unique not null,
    password_hash text not null,
    name text,
    created_at timestamptz default now()
);

-- 2. Password reset OTPs
create table if not exists password_resets (
    id uuid default gen_random_uuid() primary key,
    user_id uuid references users(id) on delete cascade not null,
    otp text not null,
    expires_at timestamptz not null,
    used boolean default false,
    created_at timestamptz default now()
);

-- Indexes
create index if not exists idx_users_email on users(email);
create index if not exists idx_password_resets_user_id on password_resets(user_id);

-- 3. Drop old FK constraints on auth.users and recreate pointing to public.users
-- Documents
alter table documents drop constraint if exists documents_user_id_fkey;
alter table documents add constraint documents_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- Chat messages
alter table chat_messages drop constraint if exists chat_messages_user_id_fkey;
alter table chat_messages add constraint chat_messages_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- Interview results
alter table interview_results drop constraint if exists interview_results_user_id_fkey;
alter table interview_results add constraint interview_results_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- Quiz results
alter table quiz_results drop constraint if exists quiz_results_user_id_fkey;
alter table quiz_results add constraint quiz_results_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- Activities
alter table activities drop constraint if exists activities_user_id_fkey;
alter table activities add constraint activities_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- User profiles
alter table user_profiles drop constraint if exists user_profiles_user_id_fkey;
alter table user_profiles add constraint user_profiles_user_id_fkey
    foreign key (user_id) references users(id) on delete cascade;

-- 4. Disable RLS on all tables (we handle auth in Express middleware now)
alter table users disable row level security;
alter table password_resets disable row level security;
alter table documents disable row level security;
alter table chat_messages disable row level security;
alter table interview_results disable row level security;
alter table quiz_results disable row level security;
alter table activities disable row level security;
alter table user_profiles disable row level security;
