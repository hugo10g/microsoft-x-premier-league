# Off-Ball Value

**AI match intelligence for the 90% of football played without the ball.**

### The problem
Broadcasts and stats follow the ball, but matches are decided away from it: the decoy run that opens space, the press that forces a bad pass, the head check before receiving. Fans never see that work, and no stat captures it.

### What it does: one pipeline, five stages
1. **Ingest:** a synthetic, football-realistic match arrives event by event as it happens: passes, shots, tackles, fouls, cards, possession changes and pressure. 25 Hz player tracking comes with it.
2. **Interpret:** it measures decoy runs, passing lanes cut, press-and-win, transition speed, head checks, pass difficulty and shot speed. These combine into an Off-Ball Index and a machine-learning player price that shows how it was calculated.
3. **Explain:** live commentary says why each moment matters. For example, "this press cut 83% of the passing lanes" or "this goal came from a 32 m pass rated 54/100". Agents write a fact-checked match report. An AI Scout answers questions about the league.
4. **Render:** a live scoreboard and feed, and a 2D tactical pitch with sprint-speed and distance tags. A timed overlay feed (JSON) can drive graphics on a live broadcast.
5. **Personalize:** Fan and Analyst modes are two different experiences. You can pick your club, follow a player (ringed on the pitch, starred in the feed, the focus of the report) and choose one of nine languages.

The simulator hides each player's true talent, and the metrics recover it, so the numbers are proven.

### Microsoft and Azure technology
- **Microsoft Agent Framework:** an Analyst, a Narrator and a Persona editor write each report, and a code fact-checker sends back any claim the data doesn't support.
- **Microsoft Foundry** (gpt-4.1-mini) powers the agents, the AI Scout and the translations.
- **Azure Container Apps** and **Azure Container Registry** host the live app, built with **GitHub Actions**. Covered by 176 automated tests.

### Try it (free, no login)
https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io
1. **Live Match** → ▶ Play live.
2. **Your view** → pick a mode, a player and a language.
3. **Tactical Playback** → open a highlight clip.
4. **Match Report** → Generate report, then open the Agent trace.
5. **Model Check.**

**Demo video:** https://youtube.com/watch?v=O_Za9cgzAP4
**Code:** https://github.com/hugo10g/microsoft-x-premier-league
