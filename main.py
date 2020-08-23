import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle

with open('newdf3.pkl', 'rb') as f:
    df3 = pickle.load(f)
with open('predictorsscale.pkl', 'rb') as f:
    predictors_scaled = pickle.load(f)
with open('newpredictors.pkl', 'rb') as f:
    predictors_df = pickle.load(f)
with open('train_predictors_val.pkl', 'rb') as f:
    train_predictors_val= pickle.load(f)
with open('newfifa.pkl', 'rb') as f:
    fifa = pickle.load(f)
with open('df3scaled.pkl', 'rb') as f:
    df3scaled = pickle.load(f)
xbr = pickle.load(open('finalxbrmodel.pkl','rb'))


predscale_target=predictors_scaled.columns.tolist()

def player_sim_team(team, position, NUM_RECOM, AGE_upper_bound):
# part 1(recommendation)
    target_cols = predscale_target


# team stats
    team_stats = df3scaled.query('position_group == @position and Club == @team').head(3)[target_cols].mean(axis=0)
    team_stats_np = team_stats.values

# player stats by each position
    ply_stats = df3scaled.query('position_group == @position and Club != @team and Age1 <= @AGE_upper_bound')[
    ['ID'] + target_cols]
    ply_stats_np = ply_stats[target_cols].values
    X = np.vstack((team_stats_np, ply_stats_np))

## KNN
    nbrs = NearestNeighbors(n_neighbors=NUM_RECOM + 1, algorithm='auto').fit(X)
    dist, rank = nbrs.kneighbors(X)


    global indice
    global predicted_players_name
    global predicted_players_value
    global predictions

    indice = ply_stats.iloc[rank[0, 1:]].index.tolist()
    predicted_players_name=df3['Name'].loc[indice,].tolist()
    predicted_players_value=fifa['Value'].loc[indice,].tolist()
    display_df1 = predictors_scaled.loc[indice,]
    playrpredictorss = predictors_df.loc[indice,]
    display_df2 = df3.loc[indice,]
    display_df = fifa.loc[indice,]


#part 2(prediction)
    predictors_anomaly_processed=playrpredictorss[playrpredictorss.index.isin(list(display_df2['ID']))].copy()
    predictors_anomaly_processed['Forward_Skill'] = predictors_anomaly_processed.loc[:,['LS',  'ST', 'RS',  'LW', 'LF', 'CF', 'RF', 'RW']].mean(axis=1)

    predictors_anomaly_processed['Midfield_Skill'] = predictors_anomaly_processed.loc[:,['LAM','CAM','RAM', 'LM', 'LCM', 'CM' ,'RCM', 'RM','LDM', 'CDM', 'RDM']].mean(axis=1)

    predictors_anomaly_processed['Defence_Skill'] = predictors_anomaly_processed.loc[:,['LWB','RWB', 'LB','LCB','CB','RCB','RB']].mean(axis=1)

    predictors_anomaly_processed = predictors_anomaly_processed.drop(['LS',  'ST', 'RS',  'LW', 'LF', 'CF', 'RF', 'RW',
          'LAM','CAM','RAM', 'LM', 'LCM', 'CM' ,'RCM', 'RM','LDM', 'CDM', 'RDM',
          'LWB','RWB', 'LB','LCB','CB','RCB','RB'], axis = 1)

    predictors_anomaly_processed=predictors_anomaly_processed.drop(predictors_anomaly_processed.iloc[:,predictors_anomaly_processed.columns.get_loc('Position_CAM'):predictors_anomaly_processed.columns.get_loc('Position_ST')+1], axis=1)

    predictors_anomaly_processed=predictors_anomaly_processed[train_predictors_val.columns]
    predictors_anomaly_processed[['International Reputation','Real Face']]=predictors_anomaly_processed[['International Reputation','Real Face']].astype('category')

    scaler = StandardScaler()
    predictors_anomaly_processed[predictors_anomaly_processed.select_dtypes(include=['float64','float32','int64','int32'], exclude=['category']).columns] = scaler.fit_transform(predictors_anomaly_processed.select_dtypes(include=['float64','float32','int64','int32'], exclude=['category']))
    predictors_anomaly_processed[predictors_anomaly_processed.select_dtypes(include='category').columns]=predictors_anomaly_processed[predictors_anomaly_processed.select_dtypes(include='category').columns].astype('int')


    predictions = abs(xbr.predict(predictors_anomaly_processed))
    predictions = predictions.astype('int64')

    result=final_pred(NUM_RECOM,predictions,predicted_players_value,predicted_players_name)
    return result





def final_pred(num_of_players,b=[],c=[],d=[]):

    z=[]
    for m in range(0,num_of_players):


        c[m]=((c[m]+b[m])/2)
        z.append({"starting_bid":c[m],"player_name":d[m]})


    return z




app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():


    data=request.args
    team_chosen = data.get('team_chosen')
    postion_chosen= data.get('postion_chosen')
    num_of_players =int( data.get('num_of_players'))
    age_up = int(data.get('age_up'))
    r = player_sim_team(team_chosen,postion_chosen,num_of_players,age_up)
    return render_template('recommend.html',player_list=r)



if __name__ == '__main__':
    app.run()




    #print("postions=side_df,cent_df,cent_md,side_md,cent_fw,side_fw,goalkeep")
    #print("team=any club teams in any of the countries ")
    #print("*********************************************** \n")
    #team_chosen = str(input("Enter the team you are looking for:  \n"))
    #postion_chosen = str(input("Enter the position you are looking for:  \n"))
    #num_of_players = input("Enter the number of similar players you are looking for: \n")
    #age_up = input("Enter the age limit: ")
    #print("***please have some biscuits, it will take some time***")

    #player_sim_team(team_chosen,postion_chosen, int(num_of_players), int(age_up))
    #finalfunction = player_sim_team(team_chosen,postion_chosen, int(num_of_players), int(age_up))
    #pickle.dump(finalfunction, open('finalfunction.pkl', 'wb'))
