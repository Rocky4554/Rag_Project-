import { createLLMWithFallback } from "../llm.js";
import { getUserProfile, upsertUserProfile } from "../db.js";
import { serverLog } from "../logger.js";

const llm = createLLMWithFallback({ provider: "groq", temperature: 0.3, maxTokens: 1000 });

/**
 * After an interview completes, extract insights and merge into the user's profile.
 * This gives the AI memory of the user's strengths/weaknesses across sessions.
 *
 * @param {string} userId
 * @param {object} interviewResult - { questionsAsked, scores, topicScores, finalReport, difficultyLevel }
 * @param {string} docName - Original document name for context
 */
export async function updateUserProfileAfterInterview(userId, interviewResult, docName) {
    try {
        // 1. Fetch existing profile (may be null for first interview)
        const existing = await getUserProfile(userId);

        // 2. Ask LLM to extract structured insights from the interview
        const insights = await extractInsights(interviewResult, existing, docName);

        // 3. Merge insights into profile
        const existingTopics = existing?.topics_covered || [];
        const existingWeak = existing?.weak_areas || [];
        const existingStrong = existing?.strong_areas || [];
        const existingScoreHistory = existing?.score_history || [];
        const existingTotalInterviews = existing?.total_interviews || 0;
        const existingAvgScore = existing?.avg_score || 0;

        // Deduplicate arrays by merging new + existing
        const mergedTopics = deduplicateMerge(existingTopics, insights.topics_covered);
        const mergedWeak = deduplicateMerge(
            existingWeak.filter(w => !insights.strong_areas.includes(w)), // Remove from weak if now strong
            insights.weak_areas
        );
        const mergedStrong = deduplicateMerge(
            existingStrong.filter(s => !insights.weak_areas.includes(s)), // Remove from strong if now weak
            insights.strong_areas
        );

        // Calculate new average score
        const currentScore = insights.avg_score_this_session;
        const newTotal = existingTotalInterviews + 1;
        const newAvg = ((existingAvgScore * existingTotalInterviews) + currentScore) / newTotal;

        // Append to score history (keep last 50)
        const newScoreEntry = {
            date: new Date().toISOString(),
            score: currentScore,
            topic: docName || 'unknown',
            difficulty: interviewResult.difficultyLevel || 'medium',
            questionsAsked: interviewResult.questionsAsked
        };
        const mergedScoreHistory = [...existingScoreHistory, newScoreEntry].slice(-50);

        // 4. Upsert profile
        await upsertUserProfile(userId, {
            topics_covered: mergedTopics,
            weak_areas: mergedWeak,
            strong_areas: mergedStrong,
            score_history: mergedScoreHistory,
            performance_summary: insights.performance_summary,
            total_interviews: newTotal,
            avg_score: Math.round(newAvg * 10) / 10,
            last_session_at: new Date().toISOString()
        });

        serverLog.info({ userId, totalInterviews: newTotal, avgScore: newAvg.toFixed(1) }, 'User profile updated after interview');

    } catch (err) {
        serverLog.error({ err: err.message, userId }, 'Failed to update user profile after interview');
    }
}

/**
 * Use LLM to extract structured insights from an interview result.
 */
async function extractInsights(interviewResult, existingProfile, docName) {
    const existingContext = existingProfile
        ? `Previous profile:
- Topics covered: ${JSON.stringify(existingProfile.topics_covered || [])}
- Known weak areas: ${JSON.stringify(existingProfile.weak_areas || [])}
- Known strong areas: ${JSON.stringify(existingProfile.strong_areas || [])}
- Total past interviews: ${existingProfile.total_interviews || 0}
- Average score: ${existingProfile.avg_score || 'N/A'}`
        : 'This is the user\'s first interview.';

    const prompt = `Analyze this interview result and extract structured insights.

Document: ${docName || 'Unknown'}
Questions asked: ${interviewResult.questionsAsked || 0}
Difficulty: ${interviewResult.difficultyLevel || 'medium'}
Topic scores: ${JSON.stringify(interviewResult.topicScores || {})}
Final report: ${(interviewResult.finalReport || '').substring(0, 1500)}

${existingContext}

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "topics_covered": ["topic1", "topic2"],
  "weak_areas": ["specific area where user struggled"],
  "strong_areas": ["specific area where user excelled"],
  "avg_score_this_session": 7.5,
  "performance_summary": "Brief 2-3 sentence summary combining past + current performance. Mention trends (improving/declining) if there's history."
}`;

    try {
        const response = await llm.invoke(prompt);
        const text = response.content || response.text || '';

        // Extract JSON from response (handle markdown code blocks)
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
            serverLog.warn('Profile updater: LLM response was not valid JSON, using fallback');
            return fallbackInsights(interviewResult);
        }

        const parsed = JSON.parse(jsonMatch[0]);
        return {
            topics_covered: Array.isArray(parsed.topics_covered) ? parsed.topics_covered : [],
            weak_areas: Array.isArray(parsed.weak_areas) ? parsed.weak_areas : [],
            strong_areas: Array.isArray(parsed.strong_areas) ? parsed.strong_areas : [],
            avg_score_this_session: typeof parsed.avg_score_this_session === 'number' ? parsed.avg_score_this_session : 5,
            performance_summary: parsed.performance_summary || 'No summary available.'
        };
    } catch (err) {
        serverLog.warn({ err: err.message }, 'Profile updater: LLM extraction failed, using fallback');
        return fallbackInsights(interviewResult);
    }
}

/**
 * Fallback when LLM fails: extract what we can directly from scores.
 */
function fallbackInsights(interviewResult) {
    const topicScores = interviewResult.topicScores || {};
    const topics = Object.keys(topicScores);
    const weak = [];
    const strong = [];
    let totalScore = 0;
    let count = 0;

    for (const [topic, score] of Object.entries(topicScores)) {
        const numScore = typeof score === 'number' ? score : parseFloat(score) || 5;
        totalScore += numScore;
        count++;
        if (numScore < 5) weak.push(topic);
        else if (numScore >= 7) strong.push(topic);
    }

    return {
        topics_covered: topics,
        weak_areas: weak,
        strong_areas: strong,
        avg_score_this_session: count > 0 ? totalScore / count : 5,
        performance_summary: `Completed interview with ${interviewResult.questionsAsked || 0} questions at ${interviewResult.difficultyLevel || 'medium'} difficulty.`
    };
}

/**
 * Merge two arrays, deduplicating by lowercase value.
 */
function deduplicateMerge(existing, incoming) {
    const seen = new Set(existing.map(s => s.toLowerCase()));
    const result = [...existing];
    for (const item of incoming) {
        if (!seen.has(item.toLowerCase())) {
            seen.add(item.toLowerCase());
            result.push(item);
        }
    }
    return result;
}

/**
 * Fetch a user's profile formatted as context for injection into interview prompts.
 * Returns a string or null if no profile exists.
 */
export async function getUserProfileContext(userId) {
    try {
        const profile = await getUserProfile(userId);
        if (!profile || profile.total_interviews === 0) return null;

        const parts = [];
        parts.push(`[USER HISTORY: ${profile.total_interviews} past interview(s), avg score: ${profile.avg_score}/10]`);

        if (profile.strong_areas?.length > 0) {
            parts.push(`Strong areas: ${profile.strong_areas.join(', ')}`);
        }
        if (profile.weak_areas?.length > 0) {
            parts.push(`Weak areas (focus more here): ${profile.weak_areas.join(', ')}`);
        }
        if (profile.topics_covered?.length > 0) {
            parts.push(`Previously covered topics: ${profile.topics_covered.join(', ')}`);
        }
        if (profile.performance_summary) {
            parts.push(`Performance summary: ${profile.performance_summary}`);
        }

        return parts.join('\n');
    } catch (err) {
        serverLog.warn({ err: err.message, userId }, 'Failed to fetch user profile context');
        return null;
    }
}
