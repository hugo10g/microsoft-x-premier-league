# Submission: Off-Ball Value

## Pitch

Ninety percent of football happens off the ball: the decoy run that opens a lane, the press that forces a bad pass, the head check before receiving. None of it reaches the screen. **Off-Ball Value** is an AI match-intelligence app that makes that work visible, live, for every fan.

It ingests a match event by event and turns 25 Hz tracking into off-ball metrics, pass difficulty and shot speed. Then it explains why each moment matters. A team of agents on **Microsoft Agent Framework** and **Microsoft Foundry** writes the recap. An Analyst, a Narrator and a Persona editor draft it, and a code fact-checker sends back any claim the data doesn't support. Each fan chooses their own view: Fan or Analyst mode, their club, a player to follow, and one of nine languages. Broadcasters get a timed overlay feed for on-screen graphics. It runs on **Azure Container Apps**, and the numbers are proven: the simulator hides each player's real talent, and the metrics recover it.

**How it scores on each judging criterion:**
- **Technological implementation:** a synthetic league with the laws of the game (fouls, cards, offsides, penalties). Its tracking and events become decoy runs, lane cuts, press-and-win, transition speed and scanning. All of it feeds a composite Off-Ball Index and a machine-learning price that shows its working.
- **Agentic design:** a multi-agent recap with a verify-and-revise loop, plus an AI Scout that answers questions from the league data.
- **Real-world impact and production readiness:**
  - It's a public app with no login.
  - It's containerised and built by GitHub Actions.
  - Precomputed scores and shared caches make every page open in under a second.
  - It has 176 automated tests.
- **User experience:** a live scoreboard with no spoilers, commentary that explains each moment, a 2D tactical pitch with highlight clips, and player radars.
- **Fit with the brief:** explainable match intelligence, real-time narratives, automated recaps and personalized experiences for fans, studios and broadcasters.

**Microsoft and Azure technology:** Microsoft Foundry (`gpt-4.1-mini`), Microsoft Agent Framework, Azure Container Apps and Azure Container Registry.

## Demo video

Under 2 minutes, showing the working app, with no music, real logos or club names. Upload it to YouTube or Vimeo as public or unlisted.

Show these in order:
1. Live Match playing.
2. The personal view: a mode, a player to follow and a language.
3. A goal on the Tactical Playback pitch.
4. The overlay feed.
5. A Match Report with its Agent trace.
6. The Player Profile price breakdown.
7. Model Check.

## Testing

- **Live app:** https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io. No login needed.
- **Try:**
  1. **Live Match** → ▶ Play live.
  2. In **Your view**, set a mode, a player and a language.
  3. **Tactical Playback** → a highlight clip.
  4. **Match Report** → Generate report (open the Agent trace).
  5. **Model Check.**
- **Code (public):** https://github.com/hugo10g/microsoft-x-premier-league. The README covers setup, the architecture and how every number is calculated.
