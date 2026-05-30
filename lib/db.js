import { supabaseAdmin } from './supabase.js';
import { serverLog } from './logger.js';

// ═══════════════════════════════════════════════════════════════════
// DOCUMENTS
// ═══════════════════════════════════════════════════════════════════

export async function saveDocument({ userId, sessionId, filename, originalName, qdrantCollection, chunkCount }) {
    const { data, error } = await supabaseAdmin
        .from('documents')
        .insert({
            user_id: userId,
            session_id: sessionId,
            filename,
            original_name: originalName,
            qdrant_collection: qdrantCollection,
            chunk_count: chunkCount
        })
        .select()
        .single();
    if (error) throw error;
    return data;
}

export async function getDocumentBySessionId(sessionId) {
    const { data, error } = await supabaseAdmin
        .from('documents')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();
    if (error) throw error;
    return data;
}

export async function getUserDocuments(userId) {
    const { data, error } = await supabaseAdmin
        .from('documents')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });
    if (error) throw error;
    return data;
}

export async function deleteDocument(documentId, userId) {
    // Delete related records first (cascade manually)
    await supabaseAdmin.from('chat_messages').delete().eq('document_id', documentId).eq('user_id', userId);
    await supabaseAdmin.from('interview_results').delete().eq('document_id', documentId).eq('user_id', userId);
    await supabaseAdmin.from('quiz_results').delete().eq('document_id', documentId).eq('user_id', userId);

    const { data, error } = await supabaseAdmin
        .from('documents')
        .delete()
        .eq('id', documentId)
        .eq('user_id', userId)
        .select()
        .single();
    if (error) throw error;
    return data;
}

// ═══════════════════════════════════════════════════════════════════
// CHAT MESSAGES
// ═══════════════════════════════════════════════════════════════════

export async function saveChatMessage({ userId, documentId, role, content }) {
    const { data, error } = await supabaseAdmin
        .from('chat_messages')
        .insert({ user_id: userId, document_id: documentId, role, content })
        .select()
        .single();
    if (error) throw error;
    return data;
}

export async function getChatHistory(documentId, limit = 20) {
    const { data, error } = await supabaseAdmin
        .from('chat_messages')
        .select('role, content, created_at')
        .eq('document_id', documentId)
        .order('created_at', { ascending: true })
        .limit(limit);
    if (error) throw error;
    return data || [];
}

// ═══════════════════════════════════════════════════════════════════
// INTERVIEW RESULTS
// ═══════════════════════════════════════════════════════════════════

export async function saveInterviewResult({ userId, documentId, threadId, questionsAsked, scores, topicScores, finalReport, difficultyLevel }) {
    const { data, error } = await supabaseAdmin
        .from('interview_results')
        .insert({
            user_id: userId,
            document_id: documentId,
            thread_id: threadId,
            questions_asked: questionsAsked,
            scores,
            topic_scores: topicScores,
            final_report: finalReport,
            difficulty_level: difficultyLevel
        })
        .select()
        .single();
    if (error) throw error;
    return data;
}

export async function getInterviewResults(userId, documentId) {
    let query = supabaseAdmin
        .from('interview_results')
        .select('*, documents(session_id, original_name, filename)')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });
    if (documentId) query = query.eq('document_id', documentId);
    const { data, error } = await query;
    if (error) throw error;
    return data || [];
}

// ═══════════════════════════════════════════════════════════════════
// QUIZ RESULTS
// ═══════════════════════════════════════════════════════════════════

export async function saveQuizResult({ userId, documentId, topic, questions, score, totalQuestions }) {
    const { data, error } = await supabaseAdmin
        .from('quiz_results')
        .insert({
            user_id: userId,
            document_id: documentId,
            topic,
            questions,
            score,
            total_questions: totalQuestions
        })
        .select()
        .single();
    if (error) throw error;
    return data;
}

export async function getQuizResults(userId, documentId) {
    let query = supabaseAdmin
        .from('quiz_results')
        .select('*, documents(session_id, original_name, filename)')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });
    if (documentId) query = query.eq('document_id', documentId);
    const { data, error } = await query;
    if (error) throw error;
    return data || [];
}

// ═══════════════════════════════════════════════════════════════════
// ACTIVE INTERVIEW SESSIONS
//
// Tracks in-progress interviews so they can be resumed after
// a page reload or network drop. The LangGraph checkpointer
// (PostgresSaver) keeps the full node state; this table stores
// the thread_id needed to reconnect to that checkpoint.
//
// Required Supabase migration (run once):
//   CREATE TABLE IF NOT EXISTS active_interviews (
//     id           uuid DEFAULT gen_random_uuid() PRIMARY KEY,
//     session_id   text NOT NULL,
//     user_id      uuid REFERENCES auth.users(id) ON DELETE CASCADE,
//     thread_id    text NOT NULL,
//     max_questions int NOT NULL DEFAULT 5,
//     questions_asked int NOT NULL DEFAULT 0,
//     current_question text DEFAULT '',
//     created_at   timestamptz DEFAULT now(),
//     updated_at   timestamptz DEFAULT now(),
//     UNIQUE(session_id)
//   );
// ═══════════════════════════════════════════════════════════════════

export async function upsertActiveInterview({ sessionId, userId, threadId, maxQuestions, questionsAsked = 0, currentQuestion = '' }) {
    const { error } = await supabaseAdmin
        .from('active_interviews')
        .upsert({
            session_id: sessionId,
            user_id: userId || null,
            thread_id: threadId,
            max_questions: maxQuestions,
            questions_asked: questionsAsked,
            current_question: currentQuestion,
            updated_at: new Date().toISOString(),
        }, { onConflict: 'session_id' });
    if (error) serverLog.warn({ err: error.message, sessionId }, 'upsertActiveInterview failed');
}

export async function getActiveInterview(sessionId) {
    const { data, error } = await supabaseAdmin
        .from('active_interviews')
        .select('*')
        .eq('session_id', sessionId)
        .maybeSingle();
    if (error) {
        serverLog.warn({ err: error.message, sessionId }, 'getActiveInterview failed');
        return null;
    }
    return data;
}

export async function clearActiveInterview(sessionId) {
    const { error } = await supabaseAdmin
        .from('active_interviews')
        .delete()
        .eq('session_id', sessionId);
    if (error) serverLog.warn({ err: error.message, sessionId }, 'clearActiveInterview failed');
}

// ═══════════════════════════════════════════════════════════════════
// ACTIVITY LOG
// ═══════════════════════════════════════════════════════════════════

export async function logActivity({ userId, action, metadata = {} }) {
    const { error } = await supabaseAdmin
        .from('activities')
        .insert({ user_id: userId, action, metadata });
    if (error) serverLog.error({ err: error.message, action }, 'Activity log error');
}

export async function getRecentActivity(userId, limit = 10) {
    const { data, error } = await supabaseAdmin
        .from('activities')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(limit);
    if (error) throw error;
    return data || [];
}

// ═══════════════════════════════════════════════════════════════════
// USER PROFILES — cross-session interview memory
// ═══════════════════════════════════════════════════════════════════

export async function getUserProfile(userId) {
    const { data, error } = await supabaseAdmin
        .from('user_profiles')
        .select('*')
        .eq('user_id', userId)
        .single();
    if (error && error.code !== 'PGRST116') throw error; // PGRST116 = no rows
    return data;
}

export async function upsertUserProfile(userId, updates) {
    const { data, error } = await supabaseAdmin
        .from('user_profiles')
        .upsert({
            user_id: userId,
            ...updates,
            updated_at: new Date().toISOString()
        }, { onConflict: 'user_id' })
        .select()
        .single();
    if (error) throw error;
    return data;
}
