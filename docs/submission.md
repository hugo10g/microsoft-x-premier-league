# Off-Ball Value

**AI match intelligence for the 90% of football played without the ball.**

### The problem
Broadcasts and stats follow the ball, but matches are decided away from it: the decoy run that opens space, the press that forces a bad pass, the head check before receiving. Fans never see that work, and no stat captures it.

### What it does
Off-Ball Value plays a synthetic, football-realistic match event by event and explains each moment as it happens. For example: "this press cut 83% of the passing lanes", or "this goal came from a 32 m pass rated 54/100".
- **Measures** decoy runs, passing lanes cut, press-and-win, transition speed and head checks from 25 Hz tracking. These combine into an Off-Ball Index and a machine-learning player price that shows how it was calculated.
- **Explains** each match in a fact-checked report. An Analyst, a Narrator and a Persona editor write it, and a code fact-checker sends back any claim the data doesn't support. An AI Scout answers questions about the league.
- **Personalizes:** Fan or Analyst mode, your club, a player to follow, and nine languages.
- **Serves broadcasters** with a timed overlay feed for on-screen graphics.
- **Proves itself:** the simulator hides each player's true talent, and the metrics recover it.

### Microsoft and Azure technology
Microsoft Foundry (gpt-4.1-mini) · Microsoft Agent Framework · Azure Container Apps · Azure Container Registry · GitHub Actions. Covered by 176 automated tests.

### Try it (free, no login)
https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io
1. **Live Match** → ▶ Play live.
2. **Your view** → pick a mode, a player and a language.
3. **Tactical Playback** → open a highlight clip.
4. **Match Report** → Generate report, then open the Agent trace.
5. **Model Check.**

**Demo video:** https://youtube.com/watch?v=O_Za9cgzAP4
**Code:** https://github.com/hugo10g/microsoft-x-premier-league
