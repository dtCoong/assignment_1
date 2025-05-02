from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup,Comment
import pandas as pd

options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

urls = {
    'standard': "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    'goalkeeping': "https://fbref.com/en/comps/9/keepers/Premier-League-Stats",
    'shooting': "https://fbref.com/en/comps/9/shooting/Premier-League-Stats",
    'passing': "https://fbref.com/en/comps/9/passing/Premier-League-Stats",
    'gca': "https://fbref.com/en/comps/9/gca/Premier-League-Stats",
    'defense': "https://fbref.com/en/comps/9/defense/Premier-League-Stats",
    'possession': "https://fbref.com/en/comps/9/possession/Premier-League-Stats",
    'misc': "https://fbref.com/en/comps/9/misc/Premier-League-Stats"
}

soups = {}

for key, url in urls.items():
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    html = driver.page_source
    soups[key] = BeautifulSoup(html, 'html.parser')
    driver.quit()

def extract_table(soup, table_id):
    table = soup.find('table', id=table_id)
    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            if table_id in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table', id=table_id)
                break
    rows = table.find('tbody').find_all('tr')
    data_rows = [[td.text.strip() for td in row.find_all('td')]
                 for row in rows if not(row.get('class') and ('thead' in row.get('class')))]
    columns = [col['data-stat'] for col in rows[0].find_all('td')]
    return pd.DataFrame(data_rows, columns=columns)
    
players_df = extract_table(soups['standard'],table_id="stats_standard")
keepers_df = extract_table(soups['goalkeeping'], table_id="stats_keeper")
shooting_df = extract_table(soups['shooting'],table_id="stats_shooting")
passing_df = extract_table(soups['passing'],table_id="stats_passing")
gca_df = extract_table(soups['gca'],table_id="stats_gca")
defense_df = extract_table(soups['defense'],table_id="stats_defense")
possession_df = extract_table(soups['possession'],table_id="stats_possession")
misc_df = extract_table(soups['misc'],table_id="stats_misc")

players_df['nationality'] = players_df['nationality'].str.split().str[-1]
players_df['minutes'] = players_df['minutes'].str.replace(',','')
players_df['minutes'] = pd.to_numeric(players_df['minutes'], errors='coerce')
players_df = players_df[players_df['minutes'] > 90]
players_df.drop(columns=[
    'birth_year','minutes_90s','goals_assists','goals_pens','pens_made','pens_att',
    'npxg','npxg_xg_assist','goals_assists_per90','goals_pens_per90',
    'goals_assists_pens_per90','xg_xg_assist_per90','npxg_per90',
    'npxg_xg_assist_per90','matches'
], errors='ignore', inplace=True)
keepers_df = keepers_df[['player','team','gk_goals_against_per90','gk_save_pct','gk_clean_sheets_pct','gk_pens_save_pct']]
shooting_df = shooting_df[['player','team', 'shots_on_target_pct', 'shots_on_target_per90', 'goals_per_shot', 'average_shot_distance']]
passing_df = passing_df[['player','team', 'passes_completed', 'passes_pct', 'passes_total_distance', 
                            'passes_pct_short', 'passes_pct_medium', 'passes_pct_long', 
                            'assisted_shots', 'passes_into_final_third', 'passes_into_penalty_area', 
                            'crosses_into_penalty_area', 'progressive_passes']]
gca_df = gca_df[['player','team','sca','sca_per90','gca','gca_per90']]
defense_df = defense_df[['player','team','tackles','tackles_won','challenges','challenges_lost','blocks','blocked_shots','blocked_passes','interceptions']]
possession_df = possession_df[['player','team','touches','touches_def_pen_area','touches_def_3rd','touches_mid_3rd',
                      'touches_att_3rd','touches_att_pen_area','take_ons','take_ons_won_pct','take_ons_tackled_pct','carries','carries_progressive_distance','progressive_carries','carries_into_final_third','carries_into_penalty_area','miscontrols','dispossessed','passes_received','progressive_passes_received']]
misc_df = misc_df[['player','team','fouls','fouled','offsides','crosses','ball_recoveries','aerials_won','aerials_lost','aerials_won_pct']]
df_final = players_df.merge(keepers_df, how='left', on=['player','team']) \
                      .merge(shooting_df, how='left', on=['player','team']) \
                      .merge(passing_df, how='left', on=['player','team']) \
                      .merge(gca_df, how='left', on=['player','team'])\
                      .merge(defense_df,how='left',on=['player','team'])\
                      .merge(possession_df,how='left',on=['player','team'])\
                      .merge(misc_df,how='left',on=['player','team'])
df_final.fillna('N/a', inplace=True)
df_final_sorted = df_final.loc[df_final['player'].str.split().str[0].argsort()]
df_final_sorted.to_csv("results.csv", index=False, encoding='utf-8-sig')
print(df_final_sorted)
