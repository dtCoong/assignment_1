import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

df = pd.read_csv("results.csv")
teams = sorted(df['team'].unique())
col = [i for i in df.columns][5:]

with open('top_3.txt',"w", encoding='utf-8') as f:
    for i in col:
        tmp = []
        f.write("-"*10 + str(i).replace("_", " ").title() +"-"*10 + '\n')
        for index,row in df.iterrows():
            if row[i]!='N/a':
                tmp.append([f'{row['player']}',f'{row[i]}'])
        tmp.sort(key = lambda k:(float(k[1])))
        f.write("Top 3 highest:\n"+"\n")
        for j in range(-1,-4,-1):
            f.write((tmp[j][0]) +": " + tmp[j][1] + "\n" + "\n")
        f.write("Top 3 lowest:\n" +"\n")
        for j in range(3):
            f.write(tmp[j][0] + ": " +tmp[j][1] + "\n" + "\n")
        f.write("="*30 + "\n" + "\n")

data = []
row = {
    '': 'all'
}
df2 = df
for i in col:
    df2[i] = pd.to_numeric(df2[i], errors='coerce')
df2.fillna(0,inplace = True)
for i in col:
    row[f"Median of {i}"] =  df2[i].median()
    row[f"Mean of {i}"] = df2[i].mean()
    row[f"Std of {i}"] = df2[i].std()
data.append(row)
for team in teams:
    team_df = df2[df2['team'] == team]
    row = {'':team}
    for i in col:
        row[f"Median of {i}"] =  team_df[i].median()
        row[f"Mean of {i}"] = team_df[i].mean()
        row[f"Std of {i}"] = team_df[i].std()
    data.append(row)
data_frame = pd.DataFrame(data)
data_frame.to_csv('results2.csv')

df3= df

os.makedirs("image/all_player", exist_ok=True)
os.makedirs("image/eachTeam", exist_ok=True)
columns = ['shots_on_target_pct','shots_on_target_per90','goals_per_shot','tackles','tackles_won','blocks']
for x in columns:
    df_copy = df3
    tmp = df_copy[x]
    tmp.fillna(0,inplace=True)
    plt.hist(tmp)
    plt.title(f"phân bố của chỉ số {x}")
    plt.xlabel(x)
    plt.ylabel("Frequency")
    plt.savefig(f'image/all_player/statistic_of_{x.replace('/', '_per_')}.png',dpi=300)
    plt.close()
    
for team in teams:
    tmp = df3[df3['team'] == team]
    for x in columns:
        df_copy = tmp[x]
        df_copy.fillna(0,inplace = True)
        plt.hist(df_copy)
        plt.title(f"phân bố của chỉ số {x} đội {team}:")
        plt.xlabel(f'{x} of team {team}')
        plt.ylabel("Frequency")
        plt.savefig(f"image/eachTeam/statistic_of_{x.replace('/', 'per')}_team_{team}.png",dpi = 300)
        plt.close()

df4 = df
dict = {}
for team in teams: dict[team] = 0
for c in col:
    t, m = "", 0
    for index, row in df.iterrows():
        if row[c] != 'N/a':
            if float(row[c]) >= m:
                m = float(row[c])
                t = row['team']
    if t:
        dict[t] += 1
list = sorted(dict.items(), key=lambda x: (-x[1]))
for i in list:
    print(i[0],": ",i[1])
print("the Team performing the best in the 2024-2025 Premier League season is",list[0][0])
