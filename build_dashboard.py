import json

from project_paths import output_path

with output_path("matchups.json").open() as f:
    matchups = json.load(f)
with output_path("team_summary.json").open() as f:
    team_summary = json.load(f)
with output_path("model_results.json").open() as f:
    model_results = json.load(f)

teams = sorted(team_summary.keys())

TEAM_COLORS = {
    "Atlanta Hawks": "#E03A3E", "Boston Celtics": "#007A33", "Brooklyn Nets": "#000000",
    "Charlotte Hornets": "#1D1160", "Chicago Bulls": "#CE1141", "Cleveland Cavaliers": "#860038",
    "Dallas Mavericks": "#00538C", "Denver Nuggets": "#0E2240", "Detroit Pistons": "#C8102E",
    "Golden State Warriors": "#1D428A", "Houston Rockets": "#CE1141", "Indiana Pacers": "#002D62",
    "Los Angeles Clippers": "#C8102E", "Los Angeles Lakers": "#552583", "Memphis Grizzlies": "#5D76A9",
    "Miami Heat": "#98002E", "Milwaukee Bucks": "#00471B", "Minnesota Timberwolves": "#0C2340",
    "New Orleans Pelicans": "#0C2340", "New York Knicks": "#006BB6", "Oklahoma City Thunder": "#007AC1",
    "Orlando Magic": "#0077C0", "Philadelphia 76ers": "#006BB6", "Phoenix Suns": "#1D1160",
    "Portland Trail Blazers": "#E03A3E", "Sacramento Kings": "#5A2D81", "San Antonio Spurs": "#C4CED4",
    "Toronto Raptors": "#CE1141", "Utah Jazz": "#002B5C", "Washington Wizards": "#002B5C",
}

rf = model_results["classification_random_forest"]
baseline = model_results["classification_baseline_home_always_wins"]
reg_home = model_results["regression"]["home_pts"]["random_forest"]
reg_home_baseline = model_results["regression"]["home_pts"]["naive_season_avg_baseline"]

team_options = "\n".join(f'<option value="{t}">{t}</option>' for t in teams)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NBA Game Outcome Predictor</title>
<style>
  :root {{
    --bg: #0b0e14;
    --panel: #131824;
    --panel2: #1a2130;
    --border: #262f42;
    --text: #e8ecf4;
    --muted: #8b95a8;
    --accent: #ff6b35;
    --accent2: #4d9fff;
    --green: #2ecc71;
    --red: #e74c3c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
    background: radial-gradient(circle at top, #111726 0%, var(--bg) 60%);
    color: var(--text);
    padding: 32px 16px 64px;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; }}
  header {{ text-align: center; margin-bottom: 28px; }}
  header h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  header p {{ color: var(--muted); margin: 0; font-size: 14px; }}
  .badges {{ display: flex; gap: 10px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }}
  .badge {{
    background: var(--panel2); border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 14px; font-size: 12.5px; color: var(--muted);
  }}
  .badge b {{ color: var(--text); }}

  .picker {{
    display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px; align-items: center;
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 22px; margin-bottom: 24px;
  }}
  .team-select {{ display: flex; flex-direction: column; gap: 8px; }}
  .team-select label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
  select {{
    background: var(--panel2); color: var(--text); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px; font-size: 15px; font-weight: 600; cursor: pointer;
  }}
  .vs {{ font-weight: 800; color: var(--muted); font-size: 14px; text-align: center; padding-top: 18px; }}

  .result {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
    padding: 26px; margin-bottom: 24px;
  }}
  .result-header {{ text-align: center; margin-bottom: 20px; }}
  .winner-line {{ font-size: 22px; font-weight: 800; margin-bottom: 4px; }}
  .conf-line {{ color: var(--muted); font-size: 13.5px; }}

  .score-row {{ display: flex; align-items: center; justify-content: center; gap: 28px; margin: 20px 0; }}
  .score-side {{ text-align: center; min-width: 140px; }}
  .score-side .team-name {{ font-size: 14px; color: var(--muted); margin-bottom: 6px; }}
  .score-side .pts {{ font-size: 42px; font-weight: 800; }}
  .score-divider {{ font-size: 24px; color: var(--muted); }}

  .prob-bar {{
    height: 10px; border-radius: 999px; background: var(--panel2); overflow: hidden;
    display: flex; margin: 18px 0 8px; border: 1px solid var(--border);
  }}
  .prob-fill {{ height: 100%; }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); }}

  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px;
  }}
  .card h3 {{ margin: 0 0 12px; font-size: 15px; display: flex; align-items: center; gap: 8px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 13.5px; padding: 6px 0; border-bottom: 1px solid var(--border); }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-row span:first-child {{ color: var(--muted); }}
  .stat-row span:last-child {{ font-weight: 700; }}

  .methodology {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 20px 22px;
    font-size: 13px; color: var(--muted); line-height: 1.6;
  }}
  .methodology h3 {{ color: var(--text); font-size: 15px; margin: 0 0 10px; }}
  .methodology table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12.5px; }}
  .methodology td, .methodology th {{ padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }}
  .methodology th {{ color: var(--text); }}
  .methodology code {{ background: var(--panel2); padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🏀 NBA Game Outcome Predictor</h1>
    <p>Pick two teams to see the model's predicted winner and projected final score, based on each team's current form.</p>
    <div class="badges">
      <div class="badge">Win/loss accuracy: <b>{rf['accuracy']*100:.1f}%</b> (vs {baseline['accuracy']*100:.1f}% "home always wins" baseline)</div>
      <div class="badge">Score prediction MAE: <b>±{reg_home['mae']:.1f} pts</b></div>
      <div class="badge">Trained on <b>2,268</b> games, 2023-24 to 2025-26 seasons</div>
    </div>
  </header>

  <div class="picker">
    <div class="team-select">
      <label>Home team</label>
      <select id="homeTeam">{team_options}</select>
    </div>
    <div class="vs">VS</div>
    <div class="team-select">
      <label>Away team</label>
      <select id="awayTeam">{team_options}</select>
    </div>
  </div>

  <div class="result" id="result"></div>

  <div class="cards" id="teamCards"></div>

  <div class="methodology">
    <h3>How this works</h3>
    <p>Every game in the training data is represented only by information known <em>before</em> tip-off: each team's
    season and last-10-game win percentage and scoring margin, home/away-specific record, days of rest and
    back-to-back status, strength of schedule faced so far, current win/loss streak, and head-to-head history against
    this specific opponent this season &mdash; all computed strictly from games prior to the one being predicted, to
    avoid leaking the outcome into the features. A <code>random forest</code> classifier predicts the winner; two
    <code>random forest</code> regressors predict each team's final point total. The model is evaluated on a
    chronological hold-out (the most recent ~20% of games by date), not a random shuffle, since shuffling would let
    the model train on future games to predict the past.</p>
    <table>
      <tr><th>Metric</th><th>Model</th><th>Baseline</th></tr>
      <tr><td>Win/loss accuracy</td><td>{rf['accuracy']*100:.1f}%</td><td>{baseline['accuracy']*100:.1f}% (home team always wins)</td></tr>
      <tr><td>Final score MAE (home team)</td><td>±{reg_home['mae']:.1f} pts</td><td>±{reg_home_baseline['mae']:.1f} pts (team's own season average)</td></tr>
    </table>
    <p><b>Scope and limitations:</b> data was collected from basketball-reference.com game logs for the 2023-24,
    2024-25, and 2025-26 regular seasons (2,268 of ~3,690 possible games &mdash; a handful of monthly schedule pages
    failed to load and were excluded rather than filled in with guessed data). "Current form" for each matchup uses
    each team's most recent rolling stats as of their last played game, not a live in-season update. Predictions
    reflect team-level scoring and win-rate patterns only &mdash; they do not account for injuries, trades, or
    lineup changes.</p>
  </div>
</div>

<script>
const matchups = {json.dumps(matchups)};
const teamSummary = {json.dumps(team_summary)};
const teamColors = {json.dumps(TEAM_COLORS)};

const homeSel = document.getElementById('homeTeam');
const awaySel = document.getElementById('awayTeam');

function init() {{
  homeSel.value = "Boston Celtics";
  awaySel.value = "Denver Nuggets";
  render();
}}

function h2hLine(m, home, away) {{
  const net = m.h2h_net_record_home_perspective;
  const meetings = m.h2h_meetings_this_season;
  const homeWins = (meetings + net) / 2;
  const awayWins = meetings - homeWins;
  return `${{home}} ${{homeWins}}-${{awayWins}} ${{away}} (${{meetings}} game${{meetings === 1 ? '' : 's'}} played)`;
}}

function render() {{
  const home = homeSel.value, away = awaySel.value;
  const resultEl = document.getElementById('result');
  const cardsEl = document.getElementById('teamCards');

  if (home === away) {{
    resultEl.innerHTML = '<p style="text-align:center;color:var(--muted)">Pick two different teams.</p>';
    cardsEl.innerHTML = '';
    return;
  }}

  const key = home + '|' + away;
  const m = matchups[key];
  const homeColor = teamColors[home] || '#4d9fff';
  const awayColor = teamColors[away] || '#ff6b35';
  const homeWinPct = Math.round(m.home_win_prob * 100);
  const awayWinPct = 100 - homeWinPct;
  const winner = m.predicted_winner;

  resultEl.innerHTML = `
    <div class="result-header">
      <div class="winner-line">🏆 Predicted winner: <span style="color:${{winner === home ? homeColor : awayColor}}">${{winner}}</span></div>
      <div class="conf-line">${{homeWinPct}}% chance ${{home}} wins at home &middot; ${{awayWinPct}}% chance ${{away}} wins on the road</div>
      ${{m.h2h_meetings_this_season > 0 ? `<div class="conf-line" style="margin-top:4px">Head-to-head this season: ${{h2hLine(m, home, away)}}</div>` : ''}}
    </div>
    <div class="score-row">
      <div class="score-side">
        <div class="team-name">${{home}} (home)</div>
        <div class="pts" style="color:${{homeColor}}">${{m.predicted_home_pts.toFixed(0)}}</div>
      </div>
      <div class="score-divider">&ndash;</div>
      <div class="score-side">
        <div class="team-name">${{away}} (away)</div>
        <div class="pts" style="color:${{awayColor}}">${{m.predicted_away_pts.toFixed(0)}}</div>
      </div>
    </div>
    <div class="prob-bar">
      <div class="prob-fill" style="width:${{homeWinPct}}%; background:${{homeColor}}"></div>
      <div class="prob-fill" style="width:${{awayWinPct}}%; background:${{awayColor}}"></div>
    </div>
    <div class="prob-labels"><span>${{home}} ${{homeWinPct}}%</span><span>${{away}} ${{awayWinPct}}%</span></div>
  `;

  function teamCard(teamName, color, label) {{
    const s = teamSummary[teamName];
    const record = `${{Math.round(s.season_win_pct * s.games_played)}}-${{s.games_played - Math.round(s.season_win_pct * s.games_played)}}`;
    const streakLabel = s.win_streak > 0 ? `W${{s.win_streak}}` : s.win_streak < 0 ? `L${{Math.abs(s.win_streak)}}` : '-';
    return `
      <div class="card">
        <h3><span class="dot" style="background:${{color}}"></span>${{teamName}} <span style="color:var(--muted);font-weight:400">(${{label}})</span></h3>
        <div class="stat-row"><span>Record (as of ${{s.as_of_date}})</span><span>${{record}}</span></div>
        <div class="stat-row"><span>Win %</span><span>${{(s.season_win_pct*100).toFixed(1)}}%</span></div>
        <div class="stat-row"><span>Point differential</span><span>${{s.season_point_diff > 0 ? '+' : ''}}${{s.season_point_diff.toFixed(1)}}</span></div>
        <div class="stat-row"><span>Pts scored / game</span><span>${{s.season_avg_pts_for.toFixed(1)}}</span></div>
        <div class="stat-row"><span>Pts allowed / game</span><span>${{s.season_avg_pts_against.toFixed(1)}}</span></div>
        <div class="stat-row"><span>Last 10 win %</span><span>${{(s.last10_win_pct*100).toFixed(0)}}%</span></div>
        <div class="stat-row"><span>${{label === 'home' ? 'Win % at home' : 'Win % on road'}}</span><span>${{(s.split_win_pct*100).toFixed(0)}}%</span></div>
        <div class="stat-row"><span>Strength of schedule</span><span>${{(s.strength_of_schedule*100).toFixed(0)}}%</span></div>
        <div class="stat-row"><span>Current streak</span><span>${{streakLabel}}</span></div>
      </div>
    `;
  }}

  cardsEl.innerHTML = teamCard(home, homeColor, 'home') + teamCard(away, awayColor, 'away');
}}

homeSel.addEventListener('change', render);
awaySel.addEventListener('change', render);
init();
</script>
</body>
</html>
"""

with output_path("nba_predictor_dashboard.html").open("w") as f:
    f.write(html)

print("Dashboard written, size:", len(html), "bytes")
