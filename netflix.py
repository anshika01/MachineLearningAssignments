import argparse
import re
import os
import csv
import math
import collections as coll
import numpy as np

def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries

    Input: filename

    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    
    import csv
    with open(filename, 'r') as f:      
        reader = csv.reader(f) 
        for row in reader:
            movie_id = int(row[0])
            user_id = int(row[1])
            rating = float(row[2])     
            user_ratings.setdefault(user_id, {}).update({movie_id: rating})
            movie_ratings.setdefault(movie_id,{}).update({user_id: rating})
            
    return user_ratings, movie_ratings

def compute_average_user_ratings(user_ratings):
    """ Given a the user_rating dict compute average user ratings

    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = {}
    
    for user,value in user_ratings.items():
        sum = 0
        movie_num=0
        for movieId, rating in value.items():
            sum += float(rating)
            movie_num += 1
        average = sum / movie_num
        ave_ratings[user]=average
    return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users

        Input: d1, d2, (dictionary of user ratings per user) 
            ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    
    movie_1=set()
    movie_2=set()
    for movie,value in d1.items():
        movie_1.add(movie) 
    for movie, value in d2.items():
        movie_2.add(movie) 
    common_movie_rated=movie_1.intersection(movie_2)
    if (len(common_movie_rated)==0):
        return 0.0
    num = 0
    den1 = 0
    den2 = 0
    for common_movie in common_movie_rated:
        num += (float(d1[common_movie])-ave_rat1)*(float(d2[common_movie])-ave_rat2)
        den1 += ((float(d1[common_movie])-ave_rat1)**2)
        den2 += ((float(d2[common_movie]) - ave_rat2) ** 2)
    # When the user gives same rating to all the movies
    try:
        user_sim = num/((den1*den2)**0.5)
    except ZeroDivisionError:
        return 0.0
    return user_sim

def main():
    """
    This function is called from the command line via
    
    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    print train_file, test_file
    
    user_ratings_train, movie_ratings_train=parse_file(train_file)
    ave_ratings=compute_average_user_ratings(user_ratings_train)
    user_train=list(user_ratings_train.keys())
    with open(test_file,'r') as test:
        with open('predictions.txt', 'w') as pred:
            writer = csv.writer(pred)
            prediction=list()
            actual=list()
            for row in csv.reader(test):
                num_sum=0.0
                sim_sum=0.0
                user=int(row[1])
                movie=int(row[0])
                other_users_ratings=movie_ratings_train[movie]
                other_users=other_users_ratings.keys()
                for i in range(len(other_users)):        
                    other_user=other_users[i]
                    similar=compute_user_similarity(user_ratings_train[user],user_ratings_train[other_user],ave_ratings[user],ave_ratings[other_user])
                    num_sum=num_sum+similar*(float(movie_ratings_train[movie][other_user])-float(ave_ratings[other_user]))
                    sim_sum = sim_sum+abs(similar)
                #No similar users
                try:
                    pred_rating=ave_ratings[user]+num_sum/sim_sum
                except ZeroDivisionError:
                    pred_rating=ave_ratings[user]
                prediction.append(pred_rating)
                actual.append(row[2])
                writer.writerow(row+[pred_rating])
                actual_np=np.array(map(float, actual))
                prediction_np=np.array(prediction)
            rmse=np.sqrt(((prediction_np - actual_np)** 2).mean())
            mae=np.absolute(prediction_np - actual_np).mean()
            print "RMSE",round(rmse,4)
            print "MAE",round(mae,4)
                
if __name__ == '__main__':
    main()
    
