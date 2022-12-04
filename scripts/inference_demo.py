#!/usr/bin/env python3
"""Inference demo for RecSys-retail"""

import sys
import logging
import argparse
import pandas as pd
import requests


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--data_path",
        required=False,
        default="https://storage.yandexcloud.net/recsys-retail-input/test.csv",
        help="test transactions dataset store path",
    )
    argparser.add_argument(
        "-h1",
        "--host",
        required=False,
        default="localhost",
        help="host used for inference",
    )
    argparser.add_argument(
        "-d1",
        "--day",
        required=False,
        default=664,
        help="day number: a number from 664 to 684",
    )
    argparser.add_argument(
        "-b",
        "--batch_id",
        required=False,
        default=1,
        help="make inference in one batch (batch_id > 0) or one by one (batch_id=0)",
    )
    argparser.add_argument(
        "-n",
        "--n_users",
        required=False,
        default=2,
        help="the number of users for inference: 'all' or int less than 300",
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="filename to store the output",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading test data...")
    test = pd.read_csv(args.data_path)

    logging.info("Generating user list...")
    df = test.groupby(["day"])["user_id"].unique()
    if args.n_users == "all":
        user_list = df[int(args.day)].tolist()
    else:
        user_list = df[int(args.day)][: int(args.n_users)].tolist()
    logging.info(f"Selected {len(user_list)} users from {len(df[int(args.day)])}")    

    if int(args.batch_id):
        get_batch_recommendations(user_list, args)
    else:
        get_individual_recommendations(user_list, args)


def get_batch_recommendations(user_list, args):
    day_recs = pd.DataFrame(columns=["day", "batch_id", "recommendations"])    
    request_str = (f"http://{args.host}:8000/predict_user_list?batch_id={int(args.batch_id)}")
    r = requests.post(request_str, json={"user_ids": user_list})    
    for user_id in user_list:
        recs = r.json()[1][str(user_id)]
        day_recs.loc[user_id, "day"] = int(args.day)
        day_recs.loc[user_id, "batch_id"] = int(args.batch_id)
        day_recs.loc[user_id, "recommendations"] = recs
        logging.info(f"User {user_id}: {recs}")
    day_recs.index.name = "user_id"
    day_recs.to_csv(args.output)


def get_individual_recommendations(user_list, args):
    day_recs = pd.DataFrame(columns=["day", "batch_id", "recommendations"])
    for user_id in user_list:
        request_str = f"http://{args.host}:8000/predict?user_id={user_id}"
        r = requests.post(request_str, json={"user_id": str(user_id)})        
        recs = r.json()[1][str(user_id)]
        day_recs.loc[user_id, "day"] = int(args.day)
        day_recs.loc[user_id, "batch_id"] = int(args.batch_id)
        day_recs.loc[user_id, "recommendations"] = recs
        logging.info(f"User {user_id}: {recs}")
    day_recs.index.name = "user_id"
    day_recs.to_csv(args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
