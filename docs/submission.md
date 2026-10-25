# Submission: Off-Ball Value

## Pitch

**Off-Ball Value: explainable match intelligence for the 90% of football played without the ball.**

Broadcasts and stats apps tell you who passed and who shot. They miss the decoy run that pulled a centre-back away, the press that cut every passing lane, and the head check before a winning pass. Off-Ball Value finds that work in 25 Hz tracking data and explains it as the match happens.

**How it works:**
- A synthetic league with the laws of the game and hidden player talents feeds a pipeline.
- The pipeline measures off-ball work. A Model Check proves each metric recovers the talent it targets.
- A live feed then says *why* each moment matters.
- A team of AI agents writes fact-checked reports.
- A "Your view" panel tailors everything to the viewer:
  - Fan or Analyst mode;
  - a favourite club;
  - a player to follow, ringed on the pitch with speed and distance tags;
  - nine languages.
- Broadcasters get a timed overlay feed (JSON).

**Built with:**
- **Microsoft Foundry** (`gpt-4.1-mini`).
- **Microsoft Agent Framework**: a multi-agent workflow.
  - An Analyst picks the story with tools, and a Narrator writes it.
  - A code fact-checker rejects any time, number or player the data doesn't support, and sends the draft back.
  - A Persona editor retells it per audience and language.
- **Azure Container Apps**, scaling to zero; GitHub Actions builds and tests the image.

## Demo video (under 2 minutes, no music or third-party marks)

**The story in one line:** football's most important work happens off the ball. Off-Ball Value finds it in live tracking data, explains it, and shapes it for every viewer. The video follows the challenge's five stages (ingest, interpret, explain, render, personalize), then shows the agents and the proof.

**What to film:** Matchday 2 · Harbourside FC v Redmoor Rovers. It has six goals and a penalty. Follow **Umar Lindqvist**, who scores twice.

### Before you record (about 5 minutes)
1. Open the live site and wait until it has loaded. The first load can take a minute or two.
2. In the sidebar, choose **Matchday 2**.
3. **Pre-warm the Spanish:**
   - set **Language** to **Español**;
   - on **Live Match**, wait for the commentary to translate;
   - then on **Match Report**, choose **Broadcaster** and click **Generate report**, and wait until it's ready.
4. **Reset:**
   - set **Language** back to **English**, **Mode** to **Fan**, and **Follow a player** to **None**;
   - go back to **Live Match**; it starts at minute 0.
5. Use full screen, 1080p, and browser zoom at about 90%. Close other tabs and turn off notifications.

### Script

Captions such as **① INGEST** are optional on-screen text you can add while editing. They help judges tick off the five stages.

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:10 | Live Match at minute 0 | "The decoy run that drags a defender away. The press that cuts every passing lane. Ninety percent of football happens off the ball, and none of it reaches the screen. Until now." |
| 0:10–0:22 | Click **▶ Play live**. Events stream in; the score, stats and rhythm chart update. Caption **① INGEST · ② INTERPRET** | "Off-Ball Value ingests the match as it happens, every pass, press and sprint from 25-frame-a-second tracking, and turns it into live statistics." |
| 0:22–0:36 | At about minute 32, click **⏸ Pause**. Point at the 15:12 press, then the 30:26 goal. Caption **③ EXPLAIN** | "Then it explains why each moment matters. This press cut 83 percent of the passing lanes. This goal came from a 32-metre pass rated 54 out of 100 for difficulty." |
| 0:36–0:48 | Sidebar: **Mode → Analyst**, **Follow a player → Umar Lindqvist**, **Language → Español**. The feed switches to Spanish, with stars on his moments. Caption **⑤ PERSONALIZE** | "Every fan gets their own match. Casual or analyst, your club, your player, your language: nine of them." |
| 0:48–1:04 | **Tactical Playback** → Highlights **Goals** → **30:26 · Umar Lindqvist**. He's ringed with ⚡ speed and distance tags, and the shot shows its speed. Caption **④ RENDER** | "On the pitch, follow your player's sprints and distance, and see the shot's speed in real time. One click replays any key moment." |
| 1:04–1:10 | Back on **Live Match**, hover the **⬇ Overlay feed (JSON)** button | "For broadcasters, every moment is also a timed, machine-readable overlay cue." |
| 1:10–1:30 | **Match Report**: the Spanish Broadcaster report with its ✅ badge. Open **Agent trace** and scroll the steps. Caption **MULTI-AGENT** | "The recap is written by a team of agents on Microsoft Agent Framework and Foundry. An Analyst picks the story, a Narrator writes it, a Persona editor shapes it for the audience, and a code fact-checker sends back anything the data doesn't support." |
| 1:30–1:42 | **Player Profile**: the attribute polygon, then scroll to **How the ML price is calculated** | "Off-ball work becomes an index, a radar for every player, and a transfer price that shows its working." |
| 1:42–1:52 | **Model Check** table | "And it's proven. The simulator hides each player's true talent, and our metrics recover it: scanning at 0.87." |
| 1:52–1:58 | Hold on Live Match with the URL in the address bar. Caption **Microsoft Foundry · Agent Framework · Azure Container Apps** | "Off-Ball Value. Live on Azure, for broadcasters, fans and scouts." |

**Total voice-over:** about 250 words, comfortable at a natural pace. If you run long, cut the 1:04 overlay line first; the README covers it.

### Final checks before uploading
- The video is under 2:00.
- There's no music, or only royalty-free music.
- No real club names or logos appear; every club and player in the app is fictional.
- No other tabs, bookmarks or personal details are visible.
- Upload to YouTube or Vimeo as **Public** or **Unlisted**, then paste the link in the submission form.

## Testing instructions

- **URL:** https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io. No login.
- **First load:** the app scales to zero when idle, so the first visit can take a minute or two to start. After that it's quick.
- **A five-minute tour:**
  1. **Live Match:** press ▶ Play live and watch the commentary, stats and rhythm update. Switch **Mode** to Analyst in the sidebar.
  2. In **Your view**, pick a player under **Follow a player**, and a language.
  3. **Tactical Playback:** click a highlight clip. The followed player is ringed with distance and sprint-speed tags.
  4. **Match Report:** press *Generate report* to see the agents, the fact-check and the agent trace.
  5. **Player Profile** and **Model Check:** the ML price breakdown, and the validation against hidden talents.
- **Overlay feed:** the Live Match page has a ⬇ button that downloads the timed JSON for broadcast graphics.
- **Code:** https://github.com/hugo10g/microsoft-x-premier-league. The README covers local setup, the architecture and the metrics.
