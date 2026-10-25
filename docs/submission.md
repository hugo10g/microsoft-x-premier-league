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

## Testing

- **Live app:** https://off-ball-value.livelyforest-7e8a1161.norwayeast.azurecontainerapps.io. No login needed.
- **Try:**
  1. **Live Match** → ▶ Play live.
  2. In **Your view**, set a mode, a player and a language.
  3. **Tactical Playback** → a highlight clip.
  4. **Match Report** → Generate report (open the Agent trace).
  5. **Model Check.**
- **Code (public):** https://github.com/hugo10g/microsoft-x-premier-league.
  6. **Video:** https://youtube.com/watch?v=O_Za9cgzAP4&feature=youtu.be
