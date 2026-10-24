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

| Time | On screen | Voice-over |
|---|---|---|
| 0:00–0:12 | Live Match at kick-off. Press ▶ Play live | "Football is ninety percent off the ball, and almost none of it is on screen. This is Off-Ball Value." |
| 0:12–0:32 | Commentary scrolls: a goal with its assist and pass difficulty, a press, a scan. Stats and the rhythm chart update | "As the match arrives event by event, it explains *why* moments matter: this press cut eighty percent of the passing lanes; this goal came from a 32-metre pass rated 51 out of 100." |
| 0:32–0:45 | Sidebar: Analyst mode, then follow a player and choose Español | "Every fan gets their own match: analyst or casual, your club, your player, your language." |
| 0:45–1:05 | Tactical Playback: a highlight clip. The followed player is ringed with "⚡ 26 km/h · 412 m"; a shot shows "⚽ 97 km/h" | "On the pitch, follow your player's sprints and distance, and see every shot's speed. Clips for every pillar: scanning, space, pressing and transitions." |
| 1:05–1:25 | Match Report: generate the Broadcaster version in Spanish; open the agent trace | "Four agents on Microsoft Agent Framework and Foundry write the recap. A code fact-checker rejects any invented time, number or player, and sends it back." |
| 1:25–1:40 | Off-Ball Index radar, then the Player Profile price breakdown | "Off-ball work becomes an index, a radar and an ML price that shows its working." |
| 1:40–1:52 | Model Check table | "And it's honest: the simulator hides each player's true talent, and the metrics recover it. Scanning, 0.87." |
| 1:52–1:58 | The live URL | "Live on Azure Container Apps. Off-Ball Value." |

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
