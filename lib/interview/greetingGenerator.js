/**
 * Enhanced greeting system with timezone, personalization, variety
 */

const greetingTemplates = {
    morning: [
        "{time}! Ready to ace this? Let's go, {name}.",
        "{time}! Fresh start ahead. You've got this, {name}.",
        "{time}, {name}! Time to shine.",
        "{time}! Let's make this count, {name}.",
    ],
    afternoon: [
        "{time}! Momentum's on your side, {name}. Keep it up.",
        "{time}, {name}. Let's push through this.",
        "{time}! You're in the zone, {name}.",
        "{time}! One more sprint, {name}.",
    ],
    evening: [
        "{time}, {name}! Final stretch. Let's finish strong.",
        "{time}! Almost there, {name}. You've got the energy.",
        "{time}, {name}. Close out strong.",
        "{time}! Last round, {name}. Bring it.",
    ]
};

const moodModifiers = {
    confident: "You're sharp today",
    focused: "Let's lock in",
    energetic: "Bring the energy",
    calm: "Take your time",
    determined: "Let's do this"
};

/**
 * Generate personalized greeting based on timezone, mood, difficulty
 * @param {string} userName - User's name
 * @param {string} timezone - Client timezone (e.g., "America/New_York")
 * @param {string} userProfileContext - Profile context string
 * @param {string} difficulty - Interview difficulty
 * @returns {object} { greeting, timeOfDay }
 */
export function generateGreeting(userName, timezone, userProfileContext, difficulty) {
    // Get user's local time
    const now = new Date();
    const localTime = new Date(now.toLocaleString('en-US', { timeZone: timezone }));
    const hours = localTime.getHours();

    // Determine time of day
    let timeOfDay, timeGreeting;
    if (hours < 12) {
        timeOfDay = 'morning';
        timeGreeting = `Good morning`;
    } else if (hours < 17) {
        timeOfDay = 'afternoon';
        timeGreeting = `Good afternoon`;
    } else {
        timeOfDay = 'evening';
        timeGreeting = `Good evening`;
    }

    // Extract mood from profile context (simple heuristic)
    let mood = 'determined';
    if (userProfileContext) {
        if (userProfileContext.includes('confident')) mood = 'confident';
        else if (userProfileContext.includes('focused')) mood = 'focused';
        else if (userProfileContext.includes('energetic')) mood = 'energetic';
        else if (userProfileContext.includes('calm')) mood = 'calm';
    }

    // Adjust template based on difficulty
    let templates = greetingTemplates[timeOfDay];
    if (difficulty === 'hard' && templates.length > 1) {
        templates = templates.slice(0, Math.ceil(templates.length / 2)); // Pick motivational ones
    }

    // Pick random template
    const template = templates[Math.floor(Math.random() * templates.length)];

    // Build greeting
    const displayName = userName && userName !== 'there' ? userName : '';
    const greeting = template
        .replace('{time}', timeGreeting)
        .replace('{name}', displayName)
        .replace('{mood}', moodModifiers[mood] || moodModifiers.determined)
        .trim();

    return {
        greeting,
        timeOfDay,
        timeGreeting,
        mood,
        difficulty
    };
}

/**
 * Fallback: simple greeting if timezone fails
 */
export function generateSimpleGreeting(userName, difficulty) {
    const h = new Date().getHours();
    const timeGreeting = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
    const displayName = userName && userName !== 'there' ? userName : '';

    const templates = [
        `${timeGreeting}, ${displayName}. Let's begin.`,
        `${timeGreeting}! Ready to go, ${displayName}?`,
        `${timeGreeting}, ${displayName}. Let's do this.`,
    ];

    return {
        greeting: templates[Math.floor(Math.random() * templates.length)],
        timeGreeting,
        difficulty
    };
}
