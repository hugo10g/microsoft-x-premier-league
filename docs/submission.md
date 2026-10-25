# Submission: Off-Ball Value

## Pitch

**Off-Ball Value** is an AI match-intelligence app that measures and explains what footballers do without the ball, live, for fans, studios and broadcasters.

Match coverage follows the ball, but most of a player's game happens away from it. Decoy runs, pressing and head checks decide matches, yet no stat captures them. Off-Ball Value turns 25 Hz tracking from a synthetic, football-realistic league into those missing numbers, and AI agents explain them in plain language.

**What it does**
- **Live Match:** follows a match event by event, with a running score, stats, and commentary on why each moment matters.
- **Explainable metrics:** decoy runs, passing lanes cut, press-and-win, transition speed, head checks, pass difficulty and shot speed. They combine into an Off-Ball Index and a machine-learning player price that shows its working.
- **Fact-checked recaps:** an Analyst, a Narrator and a Persona editor, built on Microsoft Agent Framework and Foundry, write each match report. A code fact-checker sends back any claim the data doesn't support. An AI Scout answers questions about the league.
- **Personal:** Fan or Analyst mode, a favourite club, a player to follow, and nine languages.
- **For broadcasters:** a timed overlay feed (JSON) for on-screen graphics.
- **Proven:** the simulator hides each player's true talent, and the metrics recover it.

**Built with:** Microsoft Foundry (`gpt-4.1-mini`), Microsoft Agent Framework, Azure Container Apps, Azure Container Registry and GitHub Actions. Covered by 176 automated tests.

## Demo video script

About 1:40, filmed on **Matchday 2**, so the numbers in shot 1 match what's on screen.

| # | On screen | Say |
|---|---|---|
| 1 | Live Match playing | "Ninety percent of football happens off the ball: the decoy run, the press, the head check. None of it reaches the screen. Off-Ball Value changes that, live. Here, the match arrives event by event, and every moment is explained. This press cut 83 percent of the passing lanes, and this goal came from a 32-metre pass rated 54 out of 100." |
| 2 | Your view: a mode, a player to follow, a language | "Every fan gets their own match. Fan or Analyst mode, their club, a player to follow, and nine languages." |
| 3 | A goal on the Tactical Playback pitch | "On the 2D pitch, the player I follow is ringed, with sprint speed and distance, and you can see the shot's speed." |
| 4 | The overlay feed | "For broadcasters, every moment is also a timed overlay cue they can put on screen." |
| 5 | Match Report and its Agent trace | "The recap is written by a team of agents on Microsoft Agent Framework and Foundry: an Analyst, a Narrator and a Persona editor. A fact-checker sends back anything the data doesn't support, and the trace shows every step." |
| 6 | Player Profile price breakdown | "Off-ball work becomes an index, a radar, and a price that shows its working." |
| 7 | Model Check | "And it's proven. The simulator hides each player's real talent, and the metrics recover it. Off-Ball Value, live on Azure Container Apps." |

## Testing

- **Live app:** https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io. No login needed.
- **Try:**
  1. **Live Match** → ▶ Play live.
  2. In **Your view**, set a mode, a player and a language.
  3. **Tactical Playback** → a highlight clip.
  4. **Match Report** → Generate report (open the Agent trace).
  5. **Model Check.**
- **Code (public):** https://github.com/hugo10g/microsoft-x-premier-league.
