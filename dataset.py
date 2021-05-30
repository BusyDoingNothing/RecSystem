import numpy as np
import pandas as pd
from random import randint

def normalization(hoursList):
    max_hour = hoursList.max()
    min_hour = hoursList.min()
    ratings = list()
    # Normalization in range 1-5
    if max_hour == min_hour:
        for hour in hoursList:
            if hour > 40:
                ratings.append(4.5)
            else:
                ratings.append(2.5)
    else:
        for hour in hoursList:
            norm_elm = (5-1)*((hour-min_hour)/(max_hour-min_hour))+1
            ratings.append(np.round(norm_elm,1))
    return ratings


def get_utility_matrix(dataF):    
    uniqUsers = dataF['uID'].unique() 
    uniqGames = dataF['game'].unique()
    hP = list()
    for usr in uniqUsers:
        select_indeces = list(np.where((dataF['uID'] == usr))[0])
        hP = dataF.iloc[select_indeces]['hoursPlayed'].values
        norm_hP = normalization(hP)
        for indx, value in zip(select_indeces, norm_hP):
            dataF.at[indx, 'hoursPlayed'] = value
    dataF = dataF.rename(columns={'hoursPlayed':'ratings'})

    users_items_matrix_df = dataF.pivot_table(index='uID', columns='game', values='ratings').fillna(0)
    users_items_sparse_matrix = users_items_matrix_df.to_numpy()
    
    selected_users = 50
    selected_games = 50
    users_items_sparse_matrix = users_items_sparse_matrix[:selected_users, :selected_games]

    zeros_indx = [indx for indx, row in enumerate(users_items_sparse_matrix) if not row.any()]

    for indx in zeros_indx:
        synthetic_ratings = np.round(np.random.uniform(1,5,[int(selected_games/5)]))
        for x in np.nditer(synthetic_ratings):
            users_items_sparse_matrix[indx][randint(0,selected_games-1)] = x     

    return users_items_sparse_matrix


if __name__ == '__main__':
    
    df = pd.read_csv('datasets/steam_updated.csv')
    utility_mat = get_utility_matrix(df)
    np.save('utility_mat.npy', utility_mat)
    #Human readable data
    np.savetxt('utility_mat.txt', utility_mat)