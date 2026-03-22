# Frontend Implementation Plan — RAG Interview Platform

## Tech Stack

- **Framework**: Next.js 15 (App Router, JavaScript — no TypeScript)
- **Styling**: Tailwind CSS v4 + shadcn/ui components
- **Animations**: Framer Motion + CSS 3D transforms
- **API Layer**: React Query (TanStack Query v5) + Axios
- **State Management**: Redux Toolkit (global auth, session, theme)
- **Icons**: Lucide React
- **Audio Visualization**: Web Audio API for waveform animations
- **Theme**: next-themes (dark/light mode)
- **Fonts**: Inter (body) + JetBrains Mono (code/mono)
- **Deployment**: Vercel

---

## Project Structure

```
rag-frontend/
├── public/
│   ├── images/                    # Hero images, feature illustrations
│   └── fonts/
├── src/
│   ├── app/
│   │   ├── layout.js             # Root layout (providers, theme, fonts)
│   │   ├── page.js               # Landing page (public)
│   │   ├── globals.css           # Tailwind imports + custom CSS vars
│   │   ├── (auth)/
│   │   │   ├── login/page.js
│   │   │   └── signup/page.js
│   │   └── (dashboard)/
│   │       ├── layout.js         # Dashboard shell (sidebar + topbar)
│   │       ├── page.js           # Dashboard home (document library + feature grid)
│   │       ├── upload/page.js    # Upload new document page
│   │       ├── chat/page.js      # Chat with PDF
│   │       ├── quiz/page.js      # Quiz feature
│   │       ├── summary/page.js   # Voice summary
│   │       ├── interview/page.js # Voice interview
│   │       ├── history/page.js   # Past results & activity
│   │       └── profile/page.js   # User profile & settings
│   ├── components/
│   │   ├── ui/                   # shadcn/ui components (button, card, dialog, etc.)
│   │   ├── landing/
│   │   │   ├── Hero.jsx
│   │   │   ├── BentoGrid.jsx
│   │   │   ├── FeatureShowcase.jsx
│   │   │   ├── HowItWorks.jsx
│   │   │   ├── Testimonials.jsx
│   │   │   └── Footer.jsx
│   │   ├── auth/
│   │   │   ├── LoginForm.jsx
│   │   │   ├── SignupForm.jsx
│   │   │   └── AuthGuard.jsx
│   │   ├── dashboard/
│   │   │   ├── Sidebar.jsx
│   │   │   ├── Topbar.jsx
│   │   │   ├── DocumentCard.jsx
│   │   │   ├── FeatureGrid.jsx
│   │   │   └── ActiveDocument.jsx
│   │   ├── upload/
│   │   │   ├── DropZone.jsx
│   │   │   ├── UploadProgress.jsx
│   │   │   └── ProcessingSteps.jsx
│   │   ├── chat/
│   │   │   ├── ChatContainer.jsx
│   │   │   ├── ChatMessage.jsx
│   │   │   ├── ChatInput.jsx
│   │   │   ├── StreamingText.jsx
│   │   │   └── ThinkingIndicator.jsx
│   │   ├── quiz/
│   │   │   ├── QuizSetup.jsx
│   │   │   ├── QuestionCard.jsx
│   │   │   ├── QuizProgress.jsx
│   │   │   ├── QuizResults.jsx
│   │   │   └── ScoreAnimation.jsx
│   │   ├── summary/
│   │   │   ├── SummaryView.jsx
│   │   │   ├── AudioPlayer.jsx
│   │   │   └── WaveformVisualizer.jsx
│   │   ├── interview/
│   │   │   ├── InterviewSetup.jsx
│   │   │   ├── InterviewActive.jsx
│   │   │   ├── TranscriptPanel.jsx
│   │   │   ├── VoiceWave.jsx
│   │   │   ├── SpeakingIndicator.jsx
│   │   │   ├── ListeningIndicator.jsx
│   │   │   └── InterviewReport.jsx
│   │   └── shared/
│   │       ├── PageTransition.jsx
│   │       ├── LoadingSpinner.jsx
│   │       ├── AnimatedCounter.jsx
│   │       ├── GlowCard.jsx
│   │       ├── GradientText.jsx
│   │       ├── ParticleBackground.jsx
│   │       ├── ThemeToggle.jsx
│   │       └── EmptyState.jsx
│   ├── hooks/
│   │   ├── useAuth.js
│   │   ├── useSession.js
│   │   ├── useStreamingText.js
│   │   ├── useAudioVisualizer.js
│   │   ├── useSocket.js
│   │   ├── useLiveKit.js
│   │   └── useScrollToBottom.js
│   ├── lib/
│   │   ├── api.js                # Axios instance with interceptors
│   │   ├── socket.js             # Socket.io client singleton
│   │   ├── livekit.js            # LiveKit client helpers
│   │   └── utils.js              # Shared utility functions
│   ├── store/
│   │   ├── index.js              # Redux store config
│   │   ├── authSlice.js          # Auth state (user, tokens)
│   │   ├── sessionSlice.js       # Active document/session state
│   │   └── themeSlice.js         # Theme preferences
│   └── providers/
│       ├── QueryProvider.jsx     # React Query provider
│       ├── ReduxProvider.jsx     # Redux provider
│       ├── ThemeProvider.jsx     # next-themes provider
│       └── SocketProvider.jsx    # Socket.io context provider
└── next.config.js
```

---

## Detailed Implementation Steps

---

### PHASE 1: Project Setup & Configuration

#### Step 1.1 — Initialize Next.js Project

```bash
npx create-next-app@latest rag-frontend --js --app --tailwind --eslint --src-dir --import-alias "@/*"
cd rag-frontend
```

#### Step 1.2 — Install Dependencies

```bash
# UI
npx shadcn@latest init
npm install framer-motion lucide-react next-themes

# API & State
npm install @tanstack/react-query axios @reduxjs/toolkit react-redux

# Socket & LiveKit
npm install socket.io-client livekit-client

# Utilities
npm install clsx tailwind-merge class-variance-authority
```

#### Step 1.3 — Install shadcn/ui Components

```bash
npx shadcn@latest add button card dialog input label tabs badge
npx shadcn@latest add dropdown-menu avatar separator scroll-area
npx shadcn@latest add progress toast sonner skeleton sheet
npx shadcn@latest add radio-group select slider switch tooltip
```

#### Step 1.4 — Configure Environment

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:3000/api
NEXT_PUBLIC_SOCKET_URL=http://localhost:3000
NEXT_PUBLIC_LIVEKIT_URL=wss://your-livekit-instance.livekit.cloud
```

#### Step 1.5 — Setup Providers

Create the root layout with all providers wrapped:

```
layout.js
└── ReduxProvider
    └── QueryProvider
        └── ThemeProvider
            └── SocketProvider
                └── {children}
                └── <Toaster />
```

#### Step 1.6 — Configure Tailwind

In `globals.css`, define CSS custom properties for both themes:

```css
:root {
  --background: 0 0% 100%;
  --foreground: 240 10% 3.9%;
  --primary: 262 83% 58%;          /* Purple accent */
  --accent: 190 95% 39%;           /* Teal secondary */
  --glow: 262 83% 58% / 0.15;     /* Glow effect color */
}

.dark {
  --background: 240 10% 3.9%;
  --foreground: 0 0% 98%;
  --primary: 263 70% 50%;
  --glow: 263 70% 50% / 0.2;
}
```

Design tokens: use a purple-to-teal gradient as the primary brand identity across the entire app. Every accent, glow, border highlight should use this gradient palette.

---

### PHASE 2: Landing Page

#### Step 2.1 — Hero Section (`components/landing/Hero.jsx`)

Full-viewport hero with:
- **Animated gradient background** — slowly shifting purple-to-teal mesh gradient using CSS animation
- **3D floating elements** — use framer-motion `useMotionValue` + `useTransform` for parallax book/document icons that respond to mouse movement
- **Headline**: "Turn Any Document Into an Interactive Learning Experience" — use `GradientText` component with animated gradient
- **Subheadline**: Brief description of the platform
- **Two CTAs**: "Get Started Free" (primary, glowing button with pulse animation) and "Watch Demo" (secondary, outline)
- **Hero illustration**: Animated SVG or 3D composition showing a document transforming into quiz questions, chat bubbles, and voice waves
- **Framer Motion**: Stagger children entrance with `variants` — headline slides up first, then subtext, then CTAs, then illustration fades in

#### Step 2.2 — Bento Grid Features (`components/landing/BentoGrid.jsx`)

4-column bento grid layout (responsive: 1 col mobile, 2 col tablet, 4 col desktop). Each card is a `GlowCard` — on hover, a soft gradient glow follows the cursor (use `onMouseMove` to track position, render a radial gradient at that position).

**Grid layout (varying sizes for visual interest):**

```
┌──────────────┬──────────┐
│  Chat with   │  Voice   │
│  PDF (2x1)   │ Summary  │
├───────┬──────┼──────────┤
│ Quiz  │ Voice│ Document │
│       │Inter-│ Library  │
│       │ view │ (1x1)    │
└───────┴──────┴──────────┘
```

Each card contains:
- Icon (Lucide) with gradient background
- Feature title
- One-line description
- Micro-animation on hover (e.g., chat card shows typing dots, quiz card flips a question mark, interview card shows pulsing voice wave, summary card shows audio wave)
- All cards use `whileHover={{ scale: 1.02, y: -4 }}` with `transition={{ type: "spring" }}`

#### Step 2.3 — How It Works (`components/landing/HowItWorks.jsx`)

Horizontal stepper with 3 steps, connected by animated lines:
1. **Upload** — drag-and-drop icon with floating page animation
2. **AI Processes** — brain/gear icon with spinning animation
3. **Learn & Practice** — graduation cap with confetti burst on scroll-into-view

Use `useInView` from framer-motion to trigger animations when section scrolls into viewport. Each step animates in sequentially with a 0.2s delay between them.

#### Step 2.4 — Feature Showcase (`components/landing/FeatureShowcase.jsx`)

Alternating left-right sections (like Stripe's marketing pages). Each feature gets a full section:

1. **Chat with PDF** — Show a mockup of the chat interface with streaming text animation playing in a loop
2. **Smart Quiz** — Show a question card flipping to reveal the answer with a score counter animating up
3. **Voice Summary** — Show an audio player with animated waveform bars pulsing
4. **AI Interview** — Show a split-screen mockup with voice wave on left and streaming transcript on right

Each section: image/mockup on one side, text + bullet points on the other. Use `useInView` to slide in from the corresponding side.

#### Step 2.5 — Auth Section on Landing Page

Inline auth section before the footer (or modal triggered by CTA):
- Tab toggle: Login | Sign Up
- Clean form with email + password (+ name for signup)
- Social login buttons (Google) — styled but can be placeholder
- "Or continue with email" divider
- Animated form transition between login/signup tabs using framer-motion `AnimatePresence`

#### Step 2.6 — Footer (`components/landing/Footer.jsx`)

Dark-themed footer with:
- Logo + tagline on the left
- Link columns: Product, Resources, Company
- Social icons row
- "Built with AI" badge with subtle glow
- Gradient line separator at the top of footer
- Background: subtle dot pattern or grid pattern

---

### PHASE 3: Auth Pages

#### Step 3.1 — Login Page (`app/(auth)/login/page.js`)

- Split layout: left half = decorative (gradient mesh + floating 3D shapes + app preview), right half = form
- On mobile: form only, decorative hidden
- Form fields: email, password, "Remember me" checkbox, "Forgot password?" link
- Submit button with loading state (spinner inside button)
- "Don't have an account? Sign up" link below
- On successful login: redirect to dashboard with a smooth page transition
- Store tokens in Redux + localStorage (via Redux middleware)
- Auto-refresh token on expiry using Axios interceptor

#### Step 3.2 — Signup Page (`app/(auth)/signup/page.js`)

- Same split layout as login
- Form fields: name, email, password, confirm password
- Password strength indicator (animated bar that fills and changes color: red → yellow → green)
- Submit → call `POST /api/auth/signup` → auto-login → redirect to dashboard
- "Already have an account? Login" link

#### Step 3.3 — Auth Guard (`components/auth/AuthGuard.jsx`)

- Wrapper component for `(dashboard)` layout
- Checks Redux auth state on mount
- If no token: redirect to `/login`
- If token exists: validate with `GET /api/auth/profile`
- Show skeleton loading while validating
- Handle token refresh automatically

---

### PHASE 4: Dashboard Layout

#### Step 4.1 — Sidebar (`components/dashboard/Sidebar.jsx`)

Collapsible sidebar (expanded on desktop, collapsed/sheet on mobile):
- **Top**: App logo + name (collapses to just icon)
- **Navigation items** (with Lucide icons + labels):
  - Dashboard (home)
  - Upload Document
  - Chat with PDF
  - Quiz
  - Voice Summary
  - AI Interview
  - History
  - separator
  - Profile
- **Active indicator**: animated gradient pill that slides to the active item using `layout` animation from framer-motion
- **Bottom**: Theme toggle (sun/moon with rotation animation) + user avatar dropdown
- **Collapse button**: chevron icon that rotates 180deg on toggle
- On mobile: use shadcn `Sheet` component, slides in from left

#### Step 4.2 — Topbar (`components/dashboard/Topbar.jsx`)

- Left: breadcrumb showing current page
- Center: Active document indicator — pill showing current document name + "Change" button. If no document selected, show "Select a document to begin" with pulsing border
- Right: notification bell (for future use) + user avatar + theme toggle
- Mobile: hamburger menu + active document name

#### Step 4.3 — Page Transitions (`components/shared/PageTransition.jsx`)

Wrap every dashboard page in this component:

```jsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  exit={{ opacity: 0, y: -20 }}
  transition={{ duration: 0.3, ease: "easeInOut" }}
>
  {children}
</motion.div>
```

Use `AnimatePresence` in the dashboard layout with `mode="wait"` for smooth page switches.

---

### PHASE 5: Document Upload & Library

#### Step 5.1 — Document Upload Strategy

**Upload happens ONCE from a dedicated upload page.** After uploading, the document becomes the "active document" and all features (chat, quiz, summary, interview) operate on it. Users can switch active documents from the dashboard or topbar.

**Why this approach:**
- Avoids re-uploading for every feature
- Document processing (chunking + embedding) takes time — do it once
- Users build a library of documents over time
- Clean separation: upload is one action, using features is another

#### Step 5.2 — Dashboard Home (`app/(dashboard)/page.js`)

**If user has no documents:**
- Large empty state with illustration
- "Upload your first document" CTA button (animated entrance)

**If user has documents:**
- **Active Document Section** (top):
  - Large card showing current active document (name, upload date, chunk count)
  - "Change Document" button opens document picker dialog

- **Feature Grid** (below active document):
  - 4 cards in a 2x2 grid (responsive: 1 col mobile):
    - Chat with PDF — message bubble icon, "Ask questions about your document"
    - Quiz — brain icon, "Test your knowledge"
    - Voice Summary — headphones icon, "Listen to a summary"
    - AI Interview — mic icon, "Practice with AI interviewer"
  - Each card: icon, title, description, "Start" button
  - Cards are disabled (grayed out with "Upload a document first" tooltip) if no active document
  - On hover: card lifts up, icon gets a glow, subtle gradient border appears

- **Recent Activity** (below features):
  - Timeline-style list showing recent actions (loaded from `GET /api/history/activity`)
  - Each item: icon + description + relative timestamp
  - Framer motion staggered entrance

#### Step 5.3 — Upload Page (`app/(dashboard)/upload/page.js`)

**Three-phase upload experience:**

**Phase 1 — Drop Zone:**
- Large dashed border area (full-width, ~300px height)
- Drag-and-drop support with visual feedback:
  - Default: dashed border, upload icon, "Drag & drop your PDF here or click to browse"
  - Drag over: border goes solid + purple glow, background shifts, icon scales up
  - File selected: shows filename + size + "Upload" button
- Accepted format indicator: "PDF files up to 20MB"
- Framer motion: icon floats up/down gently in idle state

**Phase 2 — Processing Steps (after upload starts):**
- Vertical stepper showing pipeline progress:
  1. "Uploading document..." — progress bar
  2. "Extracting text..." — animated text icon
  3. "Chunking content..." — splitting animation
  4. "Creating embeddings..." — brain/neural icon with pulse
  5. "Storing in vector database..." — database icon with checkmark
- Each step: starts gray, turns blue when active (with spinner), turns green when complete
- Use framer-motion to animate each step's entrance and completion
- Fake progress since backend doesn't stream progress — estimate timings based on file size

**Phase 3 — Success:**
- Confetti animation (framer-motion)
- "Document ready!" with animated checkmark
- Auto-set as active document
- "Start Chatting" / "Take a Quiz" quick-action buttons

#### Step 5.4 — Document Picker Dialog

- Triggered from topbar "Change" button or dashboard
- Modal showing all user's documents (from `GET /api/history/documents`)
- Each document row: name, upload date, chunk count, "Select" button
- Search/filter by name
- Currently active document highlighted
- "Upload New" button at the top

---

### PHASE 6: Chat with PDF

#### Step 6.1 — Chat Page Layout (`app/(dashboard)/chat/page.js`)

Full-height chat layout (fills available space minus sidebar/topbar):

```
┌─────────────────────────────┐
│ Active Document: resume.pdf │  ← small header bar
├─────────────────────────────┤
│                             │
│   Chat messages area        │  ← scrollable, takes all available space
│   (auto-scroll to bottom)   │
│                             │
├─────────────────────────────┤
│ [  Type your question... 🔍]│  ← fixed input at bottom
└─────────────────────────────┘
```

#### Step 6.2 — Chat Messages (`components/chat/ChatMessage.jsx`)

**User messages:**
- Right-aligned, primary color background, white text
- Rounded bubble shape
- Small avatar on right
- Entrance animation: slide in from right + fade

**AI messages:**
- Left-aligned, muted background (card color)
- AI avatar/icon on left
- **Streaming text effect**: characters appear one by one using `useStreamingText` hook
- While streaming: show a blinking cursor at the end
- After streaming completes: cursor disappears
- Entrance animation: slide in from left + fade
- Support markdown rendering (bold, lists, code blocks) using a lightweight markdown renderer

**Thinking indicator** (`components/chat/ThinkingIndicator.jsx`):
- Shows while waiting for API response (before streaming starts)
- Three bouncing dots animation (like iMessage typing indicator)
- Left-aligned like an AI message
- Framer motion: dots bounce with staggered delay

#### Step 6.3 — Chat Input (`components/chat/ChatInput.jsx`)

- shadcn `Input` styled as a chat input
- Send button (arrow-up icon) — disabled when empty, primary color when has text
- Send on Enter key (Shift+Enter for newline)
- Disabled state while AI is responding (with subtle pulse animation on the input border)
- Character count indicator (optional, near max limit)

#### Step 6.4 — Auto-scroll (`hooks/useScrollToBottom.js`)

- Auto-scroll to bottom when new messages arrive
- If user scrolls up manually, stop auto-scrolling
- Show "Scroll to bottom" floating button when user is scrolled up
- Button has a bounce animation and shows unread message count

#### Step 6.5 — Streaming Text Hook (`hooks/useStreamingText.js`)

```js
// Simulates streaming by revealing text character by character
// Input: full text string (received from API)
// Output: { displayedText, isStreaming }
// Speed: ~30ms per character, with slight randomization for natural feel
```

The API returns the full answer at once. This hook creates the visual streaming effect on the frontend by gradually revealing the text. This gives users immediate feedback and feels more natural than waiting for the full response.

#### Step 6.6 — Chat History Restoration

On page mount:
1. Check if session has chat history: `GET /api/chat/history/:sessionId`
2. If yes, render all past messages instantly (no streaming effect for old messages)
3. New messages get the streaming effect

---

### PHASE 7: Quiz Feature

#### Step 7.1 — Quiz Setup (`components/quiz/QuizSetup.jsx`)

Clean setup card:
- **Topic input** (optional): text input with placeholder "e.g., Chapter 3, Machine Learning basics"
- **Number of questions**: slider or radio group (3, 5, 7, 10)
- **"Generate Quiz" button** — primary, full-width
- Animated entrance for the form elements (stagger)

#### Step 7.2 — Quiz Loading State

When generating:
- Full-page overlay with centered animation
- Animated brain icon with orbiting question marks (framer-motion)
- Progress text cycling through: "Analyzing document...", "Crafting questions...", "Almost ready..."
- Estimated time indicator
- Cancel button (subtle)

#### Step 7.3 — Question Display (`components/quiz/QuestionCard.jsx`)

One question at a time (not all at once):

```
┌─────────────────────────────────┐
│ Question 3 of 10                │  ← progress indicator
│ ████████░░░░░░░░░░░░            │  ← progress bar
├─────────────────────────────────┤
│                                 │
│ What is the primary purpose     │  ← question text (large)
│ of backpropagation in neural    │
│ networks?                       │
│                                 │
│ ○ A) To initialize weights     │  ← answer options
│ ○ B) To calculate gradients    │     (radio group, each in a card)
│ ○ C) To normalize inputs       │
│ ○ D) To reduce overfitting     │
│                                 │
│           [Next Question →]     │  ← navigation
└─────────────────────────────────┘
```

- **Question entrance animation**: card slides in from the right, previous card exits to the left (AnimatePresence)
- **Option selection**: selected option gets a colored border + fill animation
- **Option hover**: subtle lift + border highlight
- Keyboard navigation: 1-4 keys select options, Enter confirms

#### Step 7.4 — Quiz Progress (`components/quiz/QuizProgress.jsx`)

Top bar showing:
- "Question X of Y" text
- Animated progress bar (smooth width transition)
- Timer (optional, for competitive mode)
- Dot indicators for each question (filled = answered, hollow = unanswered, current = pulsing)

#### Step 7.5 — Quiz Results (`components/quiz/QuizResults.jsx`)

After submitting all answers:

**Reveal animation sequence:**
1. Screen fades to a focused state
2. Large animated score counter (0 → actual score, counting up with `AnimatedCounter`)
3. Score circle/ring that fills up (like a circular progress bar)
4. Color based on score: green (>80%), yellow (50-80%), red (<50%)
5. Confetti burst if score > 80%
6. Grade label: "Excellent!", "Good Job!", "Keep Practicing!"

**Below the score:**
- Expandable list of all questions with:
  - Question text
  - User's answer (highlighted green if correct, red if wrong)
  - Correct answer (shown if user was wrong)
  - Brief explanation
- Each item uses `Accordion` from shadcn
- "Retake Quiz" and "Try Different Topic" buttons

---

### PHASE 8: Voice Summary

#### Step 8.1 — Summary Page (`app/(dashboard)/summary/page.js`)

Two-section layout:

**Top: Generate Section**
- "Generate Voice Summary" button (large, centered, with microphone icon)
- If already generated: show the summary text + audio player
- Loading state: animated gradient skeleton + "Summarizing your document..." text

**Bottom: Audio Player & Summary Text**

#### Step 8.2 — Audio Player (`components/summary/AudioPlayer.jsx`)

Custom-styled audio player (not browser default):

```
┌─────────────────────────────────────┐
│  ▶ ────────●──────────── 2:34/5:12  │  ← play/pause + seekbar + time
│                                     │
│  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  │  ← waveform visualization
│  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  │
│                                     │
│  🔊 ────────────── 1x              │  ← volume + speed control
└─────────────────────────────────────┘
```

- Play/pause button with icon morph animation (play → pause smooth transition)
- Seek bar with gradient fill
- Time display (current / total)
- Volume slider
- Speed control: 0.75x, 1x, 1.25x, 1.5x, 2x

#### Step 8.3 — Waveform Visualizer (`components/summary/WaveformVisualizer.jsx`)

When audio is playing:
- Use Web Audio API's `AnalyserNode` to get frequency data
- Render ~40-60 vertical bars that react to the audio frequencies
- Bars use gradient colors (purple → teal)
- Smooth animation: bars grow/shrink with CSS transitions or framer-motion
- When paused: bars settle to a flat line with gentle wave animation
- When idle (not playing): show static waveform silhouette

#### Step 8.4 — Summary Text Display

Below the player:
- Full summary text in a readable card
- Text highlights along with audio playback (if feasible) — or just static display
- Copy button to copy summary text
- Markdown rendering for bullet points, headers

---

### PHASE 9: AI Voice Interview

#### Step 9.1 — Interview Setup (`components/interview/InterviewSetup.jsx`)

Setup card:
- **Number of questions**: slider (3, 5, 7, 10)
- **Difficulty preference**: radio group (Easy, Medium, Hard, Adaptive)
- **Microphone test**: small "Test Mic" button that records 2 seconds and plays back
- **"Start Interview" button** — large, primary, with mic icon
- Tips section: "Speak clearly", "Take your time", "You can say 'skip' to skip a question"

#### Step 9.2 — Interview Active Layout (`components/interview/InterviewActive.jsx`)

Split-screen layout:

```
┌──────────────────┬──────────────────┐
│                  │                  │
│   VOICE PANEL    │  TRANSCRIPT      │
│                  │  PANEL           │
│  [Voice Wave]    │                  │
│                  │  AI: Question 1  │
│  AI Speaking...  │  You: Answer...  │
│  ── or ──        │  AI: Question 2  │
│  Listening...    │  You: Answer...  │
│                  │                  │
│  Q3 of 5 | Med   │                  │
│                  │                  │
├──────────────────┴──────────────────┤
│  [🔴 End Interview]                │
└─────────────────────────────────────┘
```

On mobile: stack vertically — voice panel on top (shorter), transcript below (scrollable).

#### Step 9.3 — Voice Wave Animation (`components/interview/VoiceWave.jsx`)

Central voice visualization:

**When AI is speaking:**
- Large circular waveform animation (concentric rings pulsing outward)
- Gradient colors: purple → teal
- Status text: "AI is speaking..." with animated dots
- Rings pulse in sync with TTS audio (use Web Audio API analyser on the received audio)
- Subtle glow effect behind the circle

**When AI is thinking:**
- Rings contract to center, pulse slowly
- Status text: "Thinking..." with animated dots
- Muted colors (gray tones)

**When listening to user:**
- Different wave pattern — bars or a circular EQ that responds to microphone input
- Status text: "Listening..." or "Your turn to speak"
- Green accent color
- Microphone icon in the center of the circle

**When user is speaking:**
- Wave animation intensifies based on mic input volume
- Status text: "Speaking..."
- Show live volume meter

#### Step 9.4 — Transcript Panel (`components/interview/TranscriptPanel.jsx`)

Real-time transcript display:
- Auto-scrolling container
- **AI messages**: Left-aligned, purple/primary accent border on left, "AI Interviewer" label
- **User messages**: Right-aligned, green/teal accent border on right, "You" label
- New messages appear with streaming text effect
- Each message has a subtle entrance animation (fade + slide)
- Timestamps on each message (optional, toggleable)
- Auto-scroll to bottom (same hook as chat)
- Socket.io events drive real-time updates:
  - `transcript_final` → add new message
  - `ai_feedback` → show score badge on the AI's evaluation
  - `ai_stream` → streaming text for AI responses

#### Step 9.5 — Interview Progress Bar

Thin bar at the top of the voice panel:
- Shows "Question X of Y"
- Animated progress fill
- Difficulty badge: "Easy" / "Medium" / "Hard" with color coding

#### Step 9.6 — End Interview

"End Interview" button:
- Confirmation dialog: "Are you sure? The AI will generate your final report."
- On confirm: shows "Generating report..." animation (loading spinner + text)

Also triggered automatically when the AI finishes all questions (via `interview_done` socket event).

#### Step 9.7 — Interview Report (`components/interview/InterviewReport.jsx`)

Full-page report view:

**Header:**
- "Interview Complete" with animated checkmark
- Overall score with animated counter + circular progress ring
- Grade: "Excellent", "Good", "Needs Improvement" with color

**Score Breakdown:**
- Horizontal bar chart showing per-topic scores
- Each bar animates from 0 to its value on entrance
- Color gradient based on score (red → yellow → green)

**Detailed Feedback:**
- Expandable sections for each question:
  - Question text
  - Your answer (transcript)
  - AI evaluation + score
  - Suggestions for improvement
- Use shadcn `Accordion`

**Actions:**
- "Start New Interview" button
- "Download Report" (generate PDF client-side)
- "Share Results" (copy link)

---

### PHASE 10: History & Profile

#### Step 10.1 — History Page (`app/(dashboard)/history/page.js`)

Tabbed layout (using shadcn `Tabs`):

**Tab 1 — Documents:**
- Grid/list of all uploaded documents
- Each card: filename, upload date, chunk count, actions (select, delete)
- Click to set as active document

**Tab 2 — Interviews:**
- List of past interview results
- Each row: document name, date, score, question count, "View Report" button
- Click opens the full report view

**Tab 3 — Quizzes:**
- List of past quiz attempts
- Each row: document name, topic, score, date, "Review" button

**Tab 4 — Activity:**
- Timeline view of all actions
- Each entry: icon + action description + timestamp
- Infinite scroll (load more on scroll)

#### Step 10.2 — Profile Page (`app/(dashboard)/profile/page.js`)

- User info: name, email, avatar (editable)
- Statistics card: total documents, total interviews, average score, total quizzes
- Theme preference toggle
- Logout button (with confirmation)
- Danger zone: Delete account

---

### PHASE 11: Shared Components & Animations

#### Step 11.1 — GlowCard (`components/shared/GlowCard.jsx`)

Reusable card with cursor-following glow effect:
- Track mouse position relative to card with `onMouseMove`
- Render a radial gradient at the cursor position on a pseudo-element
- Gradient uses `--glow` CSS variable (adapts to theme)
- Smooth transition when mouse enters/leaves

```jsx
// Uses a ref to track mouse position
// Applies: background: radial-gradient(circle at {x}px {y}px, var(--glow), transparent 60%)
// On the card's ::before pseudo-element (via inline style on a div)
```

#### Step 11.2 — GradientText (`components/shared/GradientText.jsx`)

Text with animated gradient:
```jsx
<span className="bg-gradient-to-r from-purple-500 via-teal-400 to-purple-500 bg-[200%_auto] bg-clip-text text-transparent animate-gradient">
  {children}
</span>
```

Add to Tailwind config:
```js
animation: { gradient: "gradient 3s linear infinite" }
keyframes: { gradient: { "0%, 100%": { backgroundPosition: "0% center" }, "50%": { backgroundPosition: "100% center" } } }
```

#### Step 11.3 — LoadingSpinner (`components/shared/LoadingSpinner.jsx`)

Multiple variants:
- `spinner` — rotating circle with gradient
- `dots` — three bouncing dots
- `pulse` — pulsing circle
- `brain` — brain icon with orbiting particles (for AI operations)

Prop: `variant`, `size`, `text` (optional loading message below)

#### Step 11.4 — AnimatedCounter (`components/shared/AnimatedCounter.jsx`)

Animates a number from 0 to target value:
- Use framer-motion's `useMotionValue` + `useTransform` + `animate`
- Duration: proportional to the value (larger numbers take slightly longer)
- Easing: `easeOut`
- Optional: prefix (e.g., "$"), suffix (e.g., "%", "/10")

#### Step 11.5 — ParticleBackground (`components/shared/ParticleBackground.jsx`)

Subtle floating particles for landing page and auth pages:
- 20-30 small dots floating slowly in random directions
- Dots connect with faint lines when close to each other
- Use Canvas API for performance (not DOM elements)
- Respect `prefers-reduced-motion` — disable animations if user prefers

#### Step 11.6 — Page Transition Wrapper

Every page wrapped with:
```jsx
<motion.div
  initial={{ opacity: 0, y: 12 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
>
```

---

### PHASE 12: Hooks

#### Step 12.1 — `useAuth` Hook

```js
// Returns: { user, isAuthenticated, isLoading, login, signup, logout, refreshToken }
// Reads from Redux auth slice
// login/signup: call API → store tokens in Redux + localStorage → set user
// logout: clear Redux + localStorage → redirect to /login
// Auto-refresh: set up Axios interceptor for 401 responses
```

#### Step 12.2 — `useSession` Hook

```js
// Returns: { sessionId, activeDocument, setActiveDocument, isDocumentLoaded }
// Reads from Redux session slice
// Manages which document is currently active
// Persists active sessionId to localStorage
// On mount: check localStorage for last active session
```

#### Step 12.3 — `useStreamingText` Hook

```js
// Input: fullText (string), speed (ms per char, default 30)
// Output: { displayedText, isStreaming, skipToEnd }
// Reveals text character by character
// skipToEnd: immediately shows full text (for user clicking "skip")
// Respects word boundaries (doesn't cut mid-word)
// Adds slight randomization to speed for natural feel (25-35ms range)
```

#### Step 12.4 — `useAudioVisualizer` Hook

```js
// Input: audioElement (ref to <audio> or AudioContext)
// Output: { frequencyData, isPlaying, volume }
// Creates AnalyserNode from Web Audio API
// Returns Float32Array of frequency data on each animation frame
// Used by WaveformVisualizer and VoiceWave components
```

#### Step 12.5 — `useSocket` Hook

```js
// Returns: { socket, isConnected, on, emit, off }
// Singleton socket.io connection (via context from SocketProvider)
// Auto-reconnect on disconnect
// Registers session on connect
// Cleans up listeners on unmount
```

#### Step 12.6 — `useLiveKit` Hook

```js
// Returns: { room, connect, disconnect, isConnected, localAudioTrack, remoteAudioTrack }
// Manages LiveKit room connection
// Handles token fetching from /api/livekit/token
// Publishes local microphone track
// Subscribes to remote AI audio track
// Returns remote audio track for visualization
```

#### Step 12.7 — `useScrollToBottom` Hook

```js
// Input: containerRef, dependencies (array of values that trigger scroll)
// Output: { scrollToBottom, isAtBottom, showScrollButton }
// Auto-scrolls when new content arrives (if user is at bottom)
// Stops auto-scrolling if user manually scrolls up
// Returns function to manually scroll to bottom
```

---

### PHASE 13: API Layer

#### Step 13.1 — Axios Instance (`lib/api.js`)

```js
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 30000,
});

// Request interceptor: attach auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Response interceptor: handle 401 → refresh token → retry
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Try refresh token
      // If refresh fails: logout + redirect
    }
    return Promise.reject(error);
  }
);

export default api;
```

#### Step 13.2 — API Functions

```js
// auth
export const authApi = {
  signup: (data) => api.post('/auth/signup', data),
  login: (data) => api.post('/auth/login', data),
  refresh: (refreshToken) => api.post('/auth/refresh', { refresh_token: refreshToken }),
  profile: () => api.get('/auth/profile'),
};

// documents
export const documentApi = {
  upload: (file) => {
    const formData = new FormData();
    formData.append('pdf', file);
    return api.post('/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  list: () => api.get('/history/documents'),
};

// chat
export const chatApi = {
  send: (sessionId, question) => api.post('/chat', { sessionId, question }),
  history: (sessionId) => api.get(`/chat/history/${sessionId}`),
};

// quiz
export const quizApi = {
  generate: (sessionId, topic, numQuestions) => api.post('/quiz', { sessionId, topic, numQuestions }),
  results: (documentId) => api.get('/history/quizzes', { params: { documentId } }),
};

// summary
export const summaryApi = {
  generate: (sessionId) => api.post('/summary', { sessionId }),
};

// interview
export const interviewApi = {
  start: (sessionId, maxQuestions) => api.post('/interview/start', { sessionId, maxQuestions }),
  results: (documentId) => api.get('/history/interviews', { params: { documentId } }),
};

// tokens
export const tokenApi = {
  livekit: (sessionId) => api.post('/livekit/token', { sessionId }),
  deepgram: () => api.get('/deepgram/token'),
};

// history
export const historyApi = {
  activity: (limit) => api.get('/history/activity', { params: { limit } }),
};
```

#### Step 13.3 — React Query Hooks

Create custom hooks for each API call:

```js
// Example: useDocuments
export function useDocuments() {
  return useQuery({
    queryKey: ['documents'],
    queryFn: () => documentApi.list().then(r => r.data.documents),
    staleTime: 5 * 60 * 1000,
  });
}

// Example: useSendChat (mutation)
export function useSendChat() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, question }) => chatApi.send(sessionId, question),
    onSuccess: () => queryClient.invalidateQueries(['chatHistory']),
  });
}
```

---

### PHASE 14: Redux Store

#### Step 14.1 — Auth Slice (`store/authSlice.js`)

```js
// State: { user, accessToken, refreshToken, isAuthenticated, isLoading }
// Reducers: setCredentials, setUser, logout, setLoading
// Middleware: persist tokens to localStorage on setCredentials
// Initialize: read from localStorage on store creation
```

#### Step 14.2 — Session Slice (`store/sessionSlice.js`)

```js
// State: { activeSessionId, activeDocument, documents }
// Reducers: setActiveSession, setActiveDocument, clearSession
// Persist: activeSessionId to localStorage
```

---

### PHASE 15: Dark/Light Mode

#### Step 15.1 — Theme Setup

- Use `next-themes` with `attribute="class"` strategy
- Default: system preference
- Toggle: sun/moon icon button with rotation animation
- Transition: 200ms CSS transition on `background-color` and `color` properties on `html` element

#### Step 15.2 — Theme Toggle (`components/shared/ThemeToggle.jsx`)

```jsx
// Moon icon rotates 180deg and morphs to sun icon (and vice versa)
// Use framer-motion AnimatePresence with icon swap
// Subtle scale bounce on click
```

#### Step 15.3 — Dark Mode Design Guidelines

- Dark background: `hsl(240, 10%, 3.9%)` (near-black with slight blue)
- Cards: `hsl(240, 5%, 10%)` (slightly lighter)
- Borders: `hsl(240, 5%, 18%)` (subtle)
- Primary: remains purple/teal gradient (pops on dark)
- Glow effects: more visible and pronounced in dark mode
- Text: white with 90% opacity for body, 60% for muted

---

### PHASE 16: Responsive Design

#### Breakpoints

```
Mobile:  < 640px  — single column, bottom nav, compact UI
Tablet:  640-1024 — two columns, collapsible sidebar
Desktop: > 1024   — full layout, expanded sidebar
```

#### Key Responsive Behaviors

- **Sidebar**: full on desktop → collapsed icons on tablet → sheet/drawer on mobile
- **Interview split-screen**: side-by-side on desktop → stacked on mobile (voice on top, transcript below)
- **Bento grid**: 4 cols → 2 cols → 1 col
- **Chat**: full-width on all sizes, input sticks to bottom
- **Quiz cards**: same layout all sizes (centered, max-width)
- **Audio player**: horizontal on desktop → stacked controls on mobile

---

## Implementation Order (Suggested)

| Priority | Phase | Description |
|----------|-------|-------------|
| 1 | Phase 1 | Project setup, providers, Tailwind config |
| 2 | Phase 14 | Redux store (auth + session) |
| 3 | Phase 13 | API layer (Axios + React Query) |
| 4 | Phase 12 | Core hooks (useAuth, useSession) |
| 5 | Phase 11 | Shared components (GlowCard, LoadingSpinner, etc.) |
| 6 | Phase 3 | Auth pages (login, signup, guard) |
| 7 | Phase 4 | Dashboard layout (sidebar, topbar, transitions) |
| 8 | Phase 5 | Upload page + document library |
| 9 | Phase 6 | Chat with PDF |
| 10 | Phase 7 | Quiz feature |
| 11 | Phase 8 | Voice Summary |
| 12 | Phase 9 | AI Interview (most complex) |
| 13 | Phase 10 | History & Profile |
| 14 | Phase 2 | Landing page (can be done last, it's marketing) |
| 15 | Phase 15-16 | Dark mode polish + responsive testing |
