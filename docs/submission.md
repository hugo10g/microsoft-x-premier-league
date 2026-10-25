# Submission: Off-Ball Value

## Pitch (rule 4.4)

**What I built.** Off-Ball Value is an AI match-intelligence app for the 90% of football played without the ball. It's built on a synthetic, football-realistic league (25 Hz tracking, events and head orientation, with the laws of the game).

**Problem it solves.** Broadcasts and stats show passes and shots, but miss the decoy run, the press and the head check that decide matches. Viewers get a scoreline, not the story.

**What it does.**
- **Ingests** each match live, event by event.
- **Interprets** it into off-ball metrics, pass difficulty and shot speed.
- **Explains** why each moment matters, in commentary and fact-checked match recaps.
- **Renders** it on a 2D pitch, plus a timed overlay feed (JSON) for broadcast graphics.
- **Personalizes** it: Fan or Analyst mode, a favourite club, a player to follow, and nine languages.
- **Proves its numbers:** the simulator hides each player's real talent, and the metrics recover it.

**Microsoft and Azure technology.**
- **Microsoft Foundry** (`gpt-4.1-mini`).
- **Microsoft Agent Framework:** an Analyst, a Narrator and a Persona editor, with a code fact-checker that rejects unsupported claims and sends drafts back.
- **Azure Container Apps**, with GitHub Actions.

## Demo video (rule 4.4: under 2 min, the working app, no third-party marks or music)

Film the live site, **Matchday 2 · Harbourside FC v Redmoor Rovers**, following **Umar Lindqvist**.

**Prepare:**
- Select Matchday 2.
- Set Language to Español, let the Live Match feed translate once, and generate the **Broadcaster** report.
- Reset to English, Fan mode, no player followed, and Live Match at minute 0.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00 | Live Match, click **▶ Play live** | "Ninety percent of football happens off the ball, and none of it reaches the screen. Off-Ball Value changes that, live." |
| 0:12 | Pause at about 32'. Point at the 15:12 press and the 30:26 goal | "Every event is ingested as it happens and explained: this press cut 83 percent of the passing lanes; this goal came from a 32-metre pass rated 54 out of 100." |
| 0:30 | Sidebar: **Analyst**, follow **Umar Lindqvist**, **Español** | "Every fan gets their own match: casual or analyst, their club, their player, their language." |
| 0:42 | **Tactical Playback** → Goals → **30:26**: player ring, speed and distance tags, shot speed | "On the pitch: your player's sprints and distance, and the shot's speed." |
| 0:55 | Live Match: hover **⬇ Overlay feed (JSON)** | "For broadcasters, every moment is a timed overlay cue." |
| 1:02 | **Match Report**: the Spanish report, its ✅ badge, and **Agent trace** | "Agents on Microsoft Agent Framework and Foundry write the recap: an Analyst, a Narrator and a Persona editor, and a fact-checker that sends back anything the data doesn't support." |
| 1:22 | **Player Profile**: the polygon, then the ML price breakdown | "Off-ball work becomes an index, a radar and a price that shows its working." |
| 1:35 | **Model Check** | "And it's proven: the simulator hides each player's talent, and the metrics recover it." |
| 1:47 | The live URL | "Off-Ball Value, live on Azure Container Apps." |

**Before uploading:** check it's under 2:00, has no music or real logos, and shows no personal tabs. Upload to YouTube or Vimeo as public or unlisted.

## Testing (rule 4.8: free access until judging ends)

- **Live app:** https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io. No login. If it has been idle, the first load takes a minute or two.
- **Try:**
  1. **Live Match** → ▶ Play live.
  2. In **Your view**, set a mode, a player and a language.
  3. **Tactical Playback** → a highlight clip.
  4. **Match Report** → Generate report (open the Agent trace).
  5. **Model Check.**
- **Code (public):** https://github.com/hugo10g/microsoft-x-premier-league. The README covers setup, the architecture and how every number is calculated.
