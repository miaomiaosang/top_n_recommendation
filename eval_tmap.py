'''
Evaluation Metric - Truncated Mean Average Precision (mAP)

F. Aiolli, A Preliminary Study on a Recommender System for the Million Songs Dataset Challenge
Preference Learning: Problems and Applications in AI (PL-12), ECAI-12 Workshop, Montpellier
http://www.ke.tu-darmstadt.de/events/PL-12/papers/08-aiolli.pdf
'''

def ap(SongsPlayed, SongsRecommended, N):

    '''
    Calculate average precision for a given user
    Params:
        SongsPlayed - List of songs played by that user
        SongsRecommended - List of songs recommended to that user
        N - number of top recommendations
    '''
    
    np = len(SongsPlayed)
    nc = 0.0
    mapr_user = 0.0
    for j, s in enumerate(SongsRecommended):
        if j >= N:
            break
        if s in SongsPlayed:
            nc += 1.0
            mapr_user += nc / (j+1)
    mapr_user /= min(np, N)
    return mapr_user

def tmap(users_data, N):

    '''
    Calculates Truncated mAP given the recommendations
    Params:
        users_data - DataFrame with columns: ['UserId': str, 'SongsPlayed': list, 'SongsRecommended': list]
        N - number of top recommendations
    '''

    n_users = len(users_data)
    users_data['aps'] = users_data.apply(lambda x: ap(x['SongsPlayed'], x['SongsRecommended'], N), axis=1)
    tmap = users_data['aps'].sum()
    return tmap/n_users